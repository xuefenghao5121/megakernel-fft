/*
 * poc/jit-fp64-forward/jit/fft_jit.cpp — precision-parameterized family
 * JIT emitter (fp32 ps path preserved + opt-in fp64 pd path).
 *
 * Copy-then-edit of Trunk src/jit/fft_jit.cpp (family emitter, all ps).
 * This topic precision-parameterizes the SAME butterfly / complex-mul /
 * load-store generation logic to branch ps/pd by element type:
 *
 *   - fp32 (float, ps): unchanged.  Result MUST NOT regress (bit-identical
 *     or within tolerance).
 *   - fp64 (double, pd): NEW.  double complex = 16 bytes; xmm holds 1
 *     double complex (SIMD width halved vs fp32: mulps→mulpd,
 *     addps→addpd, subps→subpd, vhsubps→vhsubpd).  twiddle table
 *     float→double on the same materialization path (MUST NOT build a
 *     second twiddle subsystem).
 *
 * Default fft_jit_kernel_create stays the unwindowed fp32 family.  Opt-in
 * fft_jit_kernel_create_fp64 (draft {#API-FFT-008} / {#BEH-FFT-009}) emits
 * the fp64 forward kernel.  Opt-in fft_jit_kernel_create_gauss_window
 * materializes G (σ=N/8, ∑G=1) at plan time (separate from the twiddle
 * LUT; ARCH-FFT-009 / ARCH-FFT-011) and multiplies x[n]·G[n] on the first
 * load / stage-1, in registers, before the first butterfly
 * ({#ARCH-FFT-015} / {#BEH-FFT-004}).  Execute MUST NOT call exp.
 * Family ceiling is n=0..15, N≤32768 (n=11..15 reuse the n=9..10
 * LUT-address path with a longer table; MUST NOT invent a second twiddle
 * source).
 *
 * Clauses: {#ARCH-FFT-008} (JIT fixed-N execute path), {#ARCH-FFT-009}
 * (twiddle LUT), {#ARCH-FFT-011} (N=2^n family twiddle specialization),
 * {#ARCH-FFT-015} (gauss window LUT + first-stage fuse).
 *
 * CPU requirement: AVX (checked at plan time; failure -> NULL, no
 * fallback).  All loads/stores are unaligned (movups/movupd/vmovups/
 * vmovupd) because the public ABI makes no alignment guarantee
 * (API-FFT-001).  All GPR/XMM/YMM used are caller-saved (SysV AMD64), so
 * no prologue/epilogue / stack frame.
 */

#include "fft_jit.h"

#include "weld_codelets.h"

#include "xbyak/xbyak.h"

#include <cpuid.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <new>
#include <type_traits>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

bool is_pow2(int N) {
    return N >= 1 && (N & (N - 1)) == 0;
}

int ilog2(int N) {
    int n = 0;
    while ((1 << n) < N) {
        ++n;
    }
    return n;
}

/* Generic bit-reversal over `nbits` bits. */
uint32_t bitrev(uint32_t v, int nbits) {
    uint32_t r = 0;
    for (int i = 0; i < nbits; ++i) {
        r = (r << 1) | (v & 1u);
        v >>= 1;
    }
    return r;
}

/* aligned_alloc(64, n) requires n % 64 == 0 (arbitrary N breaks the pow2
 * size assumptions the promoted paths enjoyed) */
static size_t weld_pad64(size_t v) {
    return (v + 63u) & ~63u;
}

bool cpu_has_avx(void) {
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid_count(1, 0, &eax, &ebx, &ecx, &edx)) {
        return false;
    }
    const bool osxsave = (ecx & (1u << 27)) != 0;
    const bool avx = (ecx & (1u << 28)) != 0;
    if (!osxsave || !avx) {
        return false;
    }
    uint32_t lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0u));
    return (lo & 0x6u) == 0x6u;
}

/* FMA3 feature check (CPUID.1:ECX.FMA = bit 12).  The SchedEmitter pd path
 * emits vfmaddsub231pd complex multiplies only when this passes; otherwise
 * it falls back to vmulpd x2 + vhsubpd with the conjugate twiddle derived
 * in registers (vshufpd+vxorpd) from the SINGLE W table (never a second
 * Wn table; [[CON-FFT-002]] ISA ceiling — MUST NOT emit FMA3 without FMA). */
bool cpu_has_fma(void) {
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid_count(1, 0, &eax, &ebx, &ecx, &edx)) {
        return false;
    }
    return (ecx & (1u << 12)) != 0;
}

/* R2 F7 measured plan defaults (unset/auto knob values).  PROVISIONAL
 * (both off) until the F7 A/B matrix (evidence/measure-r2.log) is
 * collected; the final measured winners are pinned here by the
 * implement-r2 hop and the honest comparison lives in the evidence log. */
constexpr bool kNtStoreDefault = false;
constexpr bool kPrefetchDefault = false;

} // namespace

struct fft_jit_kernel {
    Xbyak::CodeGenerator *emitter; /* owns the executable code buffer */
    float *lut;                    /* aligned twiddle LUT (fp32 LUT path) */
    double *lut_d;                 /* aligned twiddle LUT (fp64 LUT path) */
    float *win;                    /* plan-time G[N] (windowed kernels) */
    float *win_dup;                /* G[bitrev(k)] duplicated per complex */
    double *scratch_d;             /* pd Sched path: stage-0 ping buffer */
    float *scratch_f;              /* ps Sched path: stage-0 ping buffer
                                      (topic jit-ps-sched-fuse) */
    double *h_mul_d;               /* topic jit-fp64-conv-fuse: conv weld
                                      natural H table (nullptr otherwise;
                                      value-init keeps forward/fp32 kernels
                                      untouched) */
    double *h_swap_d;              /* topic jit-fp64-conv-fuse: conv weld
                                      pre-swapped H twin (nullptr otherwise) */
    double *weld_arena;            /* topic jit-codelet-weld: t1fv W / TW
                                      sweep tables + codelet stride arrays +
                                      stage descriptors (nullptr otherwise) */
    size_t weld_arena_bytes;
    double *weld_hc;               /* topic jit-codelet-weld: half-B
                                      load-fusion conj(H) 32B slots */
    double *spec_d;                /* topic jit-fp64-linear-conv: OLS weld
                                      natural spectrum transit buffer
                                      (nullptr otherwise; value-init) */
    fft_jit_ols_fn_t fn_ols;       /* topic jit-fp64-linear-conv: OLS
                                      stream entry (P, y, nblocks);
                                      nullptr otherwise (value-init) */
    fft_jit_fn_t fn;
    fft_jit_fn2_t fn2;             /* topic jit-joint-search-batched: batch
                                      * single-core entry (io, stride) */
    fft_jit_fn1c_t fn2c;           /* topic R2 (amend r2): slim single-core
                                      * batch kernel's contiguous
                                      * stride==N-specialized entry */
    int code_size;
    int windowed;
    int precision;                 /* 0 = fp32, 1 = fp64 */
    int batch_single;              /* topic jit-joint-search-batched: 1 =
                                      * windowed-batch single-core kernel */
};

namespace {

/* Emit a vector of reals (float: dd; double: dq) into the instruction
 * stream constant pool. */
template <typename R>
static inline void emit_reals_into(Xbyak::CodeGenerator &cg,
                                   const std::vector<R> &v) {
    for (R f : v) {
        if constexpr (std::is_same_v<R, float>) {
            uint32_t u;
            __builtin_memcpy(&u, &f, sizeof(u));
            cg.dd(u);
        } else {
            uint64_t u;
            __builtin_memcpy(&u, &f, sizeof(u));
            cg.dq(u);
        }
    }
}

/*
 * Emit the pool-containing kernel (n=5..8).  We wrap the emitter's code
 * emission with an explicitly-emitted constant pool appended after ret()
 * for the immediate path.  To keep a single code path,
 * fft_jit_kernel_create builds the twiddle data once and hands it to
 * either a pool or an aligned LUT.  Precision-parameterized: fp32 keeps
 * the ps pool (sign2 + bitrev + stageW + stageWn), fp64 adds a pd pool
 * (sign2pd + bitrev + double stageW + double stageWn).
 */
template <typename R>
class PoolEmitter final : public Xbyak::CodeGenerator {
public:
    PoolEmitter(const std::vector<uint16_t> &bitrevPairs, int nPairs,
                const std::vector<std::vector<R>> &stageW,
                const std::vector<std::vector<R>> &stageWn, int n,
                const float *winDup)
        : Xbyak::CodeGenerator(262144), winDup_(winDup) {
        using namespace Xbyak;
        const int N = 1 << n;

        /* ---------------- body (immediate references via rip labels) ----- */
        emitBody(nPairs, N, n);

        /* ---------------- constant pool (常量池进指令流) ---------------- */
        if constexpr (std::is_same_v<R, float>) {
            align(32);
            L(sign2Lbl_);
            dd(sign2_bits(0));
            dd(sign2_bits(1));
            dd(sign2_bits(2));
            dd(sign2_bits(3));
        } else {
            align(16);
            L(sign2pdLbl_);
            dq(0);
            dq(0x8000000000000000ull);
        }

        align(16);
        L(bitrevLbl_);
        for (int p = 0; p < nPairs; ++p) {
            dw(bitrevPairs[2 * p]);
            dw(bitrevPairs[2 * p + 1]);
        }

        for (int s = 3; s <= n; ++s) {
            align(32);
            L(stageWlbl_[s]);
            emit_reals_into(*this, stageW[s - 3]);
            align(32);
            L(stageWnlbl_[s]);
            emit_reals_into(*this, stageWn[s - 3]);
        }
    }

private:
    void emitBody(int nPairs, int N, int n) {
        using namespace Xbyak;
        if constexpr (std::is_same_v<R, float>) {
            emitBodyFp32(nPairs, N, n);
        } else {
            emitBodyFp64(nPairs, N, n);
        }
    }

    /* ---- fp32 (ps): unchanged from Trunk ---- */
    void emitBodyFp32(int nPairs, int N, int n) {
        using namespace Xbyak;
        /* ---- bit-reversal (immediate: lea rip+label) ---- */
        lea(r8, ptr[rip + bitrevLbl_]);
        if (nPairs > 0) {
            xor_(ecx, ecx);
            Label brLoop;
            L(brLoop);
            movzx(eax, word[r8 + rcx * 4]);
            movzx(edx, word[r8 + rcx * 4 + 2]);
            movq(xmm0, ptr[rdi + rax * 8]);
            movq(xmm1, ptr[rdi + rdx * 8]);
            movq(ptr[rdi + rax * 8], xmm1);
            movq(ptr[rdi + rdx * 8], xmm0);
            add(ecx, 1);
            cmp(ecx, nPairs);
            jb(brLoop);
        }

        /* ---- stage 1 (m=2, h=1): twiddle W_2^0 = 1 ---- */
        if (winDup_ != nullptr) {
            mov(r11, reinterpret_cast<uint64_t>(winDup_));
        }
        mov(rsi, rdi);
        lea(rdx, ptr[rdi + N * 8]);
        Label s1Loop;
        L(s1Loop);
        movups(xmm0, ptr[rsi]);
        if (winDup_ != nullptr) {
            movups(xmm1, ptr[r11]);
            mulps(xmm0, xmm1);
            add(r11, 16);
        }
        movaps(xmm2, xmm0);
        shufps(xmm2, xmm2, 0x4E);
        movaps(xmm3, xmm0);
        addps(xmm3, xmm2);
        subps(xmm0, xmm2);
        movlps(ptr[rsi], xmm3);
        movlps(ptr[rsi + 8], xmm0);
        add(rsi, 16);
        cmp(rsi, rdx);
        jb(s1Loop);

        /* ---- stage 2 (m=4, h=2) ---- */
        if (n >= 2) {
            movaps(xmm7, ptr[rip + sign2Lbl_]);
            mov(rsi, rdi);
            lea(rdx, ptr[rdi + N * 8]);
            Label s2Loop;
            L(s2Loop);
            movups(xmm0, ptr[rsi]);
            movups(xmm1, ptr[rsi + 16]);
            movaps(xmm2, xmm1);
            shufps(xmm2, xmm2, 0xB1);
            xorps(xmm2, xmm7);
            blendps(xmm1, xmm2, 0x0C);
            movaps(xmm3, xmm0);
            addps(xmm3, xmm1);
            subps(xmm0, xmm1);
            movups(ptr[rsi], xmm3);
            movups(ptr[rsi + 16], xmm0);
            add(rsi, 32);
            cmp(rsi, rdx);
            jb(s2Loop);
        }

        /* ---- stages 3..n: AVX 4-butterfly, W/Wn from rip+label pool ---- */
        for (int s = 3; s <= n; ++s) {
            const int h = 1 << (s - 1);
            lea(r8, ptr[rip + stageWlbl_[s]]);
            lea(r9, ptr[rip + stageWnlbl_[s]]);
            mov(rsi, rdi);
            lea(rdx, ptr[rdi + N * 8]);
            Label blockLoop;
            L(blockLoop);
            lea(r10, ptr[rsi + h * 8]);
            xor_(ecx, ecx);
            Label kLoop;
            L(kLoop);
            vmovups(ymm0, ptr[rsi + rcx * 8]);
            vmovups(ymm1, ptr[r10 + rcx * 8]);
            vmovups(ymm2, ptr[r8 + rcx * 8]);
            vmovups(ymm3, ptr[r9 + rcx * 8]);
            vmulps(ymm4, ymm1, ymm2);
            vmulps(ymm5, ymm1, ymm3);
            vhsubps(ymm4, ymm4, ymm5);
            vshufps(ymm4, ymm4, ymm4, 0xD8);
            vaddps(ymm6, ymm0, ymm4);
            vsubps(ymm7, ymm0, ymm4);
            vmovups(ptr[rsi + rcx * 8], ymm6);
            vmovups(ptr[r10 + rcx * 8], ymm7);
            add(ecx, 4);
            cmp(ecx, h);
            jb(kLoop);
            add(rsi, (1 << s) * 8);
            cmp(rsi, rdx);
            jb(blockLoop);
        }
        vzeroupper();
        ret();
    }

    /* ---- fp64 (pd): same butterfly logic, 1 double complex per xmm ---- */
    void emitBodyFp64(int nPairs, int N, int n) {
        using namespace Xbyak;
        /* ---- bit-reversal: complex index ×16 bytes (shl 4) ---- */
        lea(r8, ptr[rip + bitrevLbl_]);
        if (nPairs > 0) {
            xor_(ecx, ecx);
            Label brLoop;
            L(brLoop);
            movzx(eax, word[r8 + rcx * 4]);
            movzx(edx, word[r8 + rcx * 4 + 2]);
            shl(rax, 4);
            shl(rdx, 4);
            movupd(xmm0, ptr[rdi + rax]);
            movupd(xmm1, ptr[rdi + rdx]);
            movupd(ptr[rdi + rax], xmm1);
            movupd(ptr[rdi + rdx], xmm0);
            add(ecx, 1);
            cmp(ecx, nPairs);
            jb(brLoop);
        }

        /* ---- stage 1 (m=2, h=1): one complex per xmm ---- */
        mov(rsi, rdi);
        lea(rdx, ptr[rdi + N * 16]);
        Label s1Loop;
        L(s1Loop);
        movupd(xmm0, ptr[rsi]);
        movupd(xmm1, ptr[rsi + 16]);
        movapd(xmm2, xmm0);
        addpd(xmm2, xmm1);
        subpd(xmm0, xmm1);
        movupd(ptr[rsi], xmm2);
        movupd(ptr[rsi + 16], xmm0);
        add(rsi, 32);
        cmp(rsi, rdx);
        jb(s1Loop);

        /* ---- stage 2 (m=4, h=2): W=(1, -i) per block of 4 complexes ---- */
        if (n >= 2) {
            movapd(xmm7, ptr[rip + sign2pdLbl_]); /* (0.0, -0.0) */
            mov(rsi, rdi);
            lea(rdx, ptr[rdi + N * 16]);
            Label s2Loop;
            L(s2Loop);
            movupd(xmm0, ptr[rsi]);
            movupd(xmm1, ptr[rsi + 16]);
            movupd(xmm2, ptr[rsi + 32]);
            movupd(xmm3, ptr[rsi + 48]);
            /* (-i)*d = (d.im, -d.re) */
            movapd(xmm4, xmm3);
            shufpd(xmm4, xmm4, 0x1);
            xorpd(xmm4, xmm7);
            movapd(xmm5, xmm0);
            addpd(xmm5, xmm2); /* X0 = a + c */
            subpd(xmm0, xmm2); /* X2 = a - c */
            movapd(xmm6, xmm1);
            addpd(xmm6, xmm4); /* X1 = b + (-i)d */
            subpd(xmm1, xmm4); /* X3 = b - (-i)d */
            movupd(ptr[rsi], xmm5);
            movupd(ptr[rsi + 16], xmm6);
            movupd(ptr[rsi + 32], xmm0);
            movupd(ptr[rsi + 48], xmm1);
            add(rsi, 64);
            cmp(rsi, rdx);
            jb(s2Loop);
        }

        /* ---- stages 3..n: AVX 2-butterfly (1 ymm = 2 double complexes) ---- */
        for (int s = 3; s <= n; ++s) {
            const int h = 1 << (s - 1);
            lea(r8, ptr[rip + stageWlbl_[s]]);
            lea(r9, ptr[rip + stageWnlbl_[s]]);
            mov(rsi, rdi);
            lea(rdx, ptr[rdi + N * 16]);
            Label blockLoop;
            L(blockLoop);
            lea(r10, ptr[rsi + h * 16]);
            xor_(ecx, ecx);
            Label kLoop;
            L(kLoop);
            vmovupd(ymm0, ptr[rsi + rcx * 8]);
            vmovupd(ymm1, ptr[r10 + rcx * 8]);
            vmovupd(ymm2, ptr[r8 + rcx * 8]);
            vmovupd(ymm3, ptr[r9 + rcx * 8]);
            vmulpd(ymm4, ymm1, ymm2);
            vmulpd(ymm5, ymm1, ymm3);
            vhsubpd(ymm4, ymm4, ymm5);
            vaddpd(ymm6, ymm0, ymm4);
            vsubpd(ymm7, ymm0, ymm4);
            vmovupd(ptr[rsi + rcx * 8], ymm6);
            vmovupd(ptr[r10 + rcx * 8], ymm7);
            add(ecx, 4); /* 2 double complexes = 4 doubles */
            cmp(ecx, 2 * h);
            jb(kLoop);
            add(rsi, (1 << s) * 16);
            cmp(rsi, rdx);
            jb(blockLoop);
        }
        vzeroupper();
        ret();
    }

    static uint32_t sign2_bits(int idx) {
        const float s2[4] = {0.0f, 0.0f, 0.0f, -0.0f};
        uint32_t u;
        __builtin_memcpy(&u, &s2[idx], sizeof(u));
        return u;
    }

    Xbyak::Label bitrevLbl_;
    Xbyak::Label sign2Lbl_;
    Xbyak::Label sign2pdLbl_;
    Xbyak::Label stageWlbl_[16];
    Xbyak::Label stageWnlbl_[16];
    const float *winDup_;
};

/*
 * Specialized immediate kernels for N=2,4,8,16 (binder amend
 * n2-16-specialize, DELTA F3).  Precision-parameterized:
 *
 *   - fp32: fully-unrolled register-resident straight-line bodies on xmm
 *     pair registers (one xmm per complex PAIR); all SSE, no vzeroupper.
 *   - fp64: one ymm per complex PAIR (2 double complexes per ymm, low
 *     128 = X[2p], high 128 = X[2p+1]); AVX pd, vzeroupper before ret.
 *
 * The buffer is read exactly once and written exactly once for the flat
 * sizes; bit-reversal is folded into the stage-2 load offsets; the
 * butterfly arithmetic is the same add/sub/mul operand set as the
 * corresponding PoolEmitter stage, so the specialized kernels produce the
 * same values as the loop path at the same N.
 */
template <typename R>
class SmallEmitter final : public Xbyak::CodeGenerator {
public:
    SmallEmitter(int n, const std::vector<std::vector<R>> &stageW,
                 const std::vector<std::vector<R>> &stageWn,
                 const float *G)
        : Xbyak::CodeGenerator(8192), G_(G) {
        emitBody(n);
        emitPool(n, stageW, stageWn);
    }

private:
    void emitBody(int n) {
        if (n == 1) {
            emitFlat2();
            return;
        }
        if (n == 2) {
            emitFlat4();
            return;
        }
        if constexpr (std::is_same_v<R, float>) {
            emitPairBodyFp32(n); /* n=3,4 (N=8,16) */
        } else {
            emitPairBodyFp64(n);
        }
    }

    /* ---------------- fp32 flat / pair bodies (unchanged) ---------------- */

    void emitFlat2() {
        using namespace Xbyak;
        if constexpr (std::is_same_v<R, float>) {
            movq(xmm0, ptr[rdi]);       /* x[0] */
            movq(xmm1, ptr[rdi + 8]);   /* x[1] */
            if (G_ != nullptr) {
                mulps(xmm0, ptr[rip + gDupLbl_[0]]);
                mulps(xmm1, ptr[rip + gDupLbl_[1]]);
            }
            movaps(xmm2, xmm0);
            addps(xmm2, xmm1);          /* X0 = a+b */
            subps(xmm0, xmm1);          /* X1 = a-b */
            movlps(ptr[rdi], xmm2);
            movlps(ptr[rdi + 8], xmm0);
        } else {
            movupd(xmm0, ptr[rdi]);       /* x[0] */
            movupd(xmm1, ptr[rdi + 16]);  /* x[1] */
            movapd(xmm2, xmm0);
            addpd(xmm2, xmm1);            /* X0 = a+b */
            subpd(xmm0, xmm1);            /* X1 = a-b */
            movupd(ptr[rdi], xmm2);
            movupd(ptr[rdi + 16], xmm0);
        }
        ret();
    }

    void emitFlat4() {
        using namespace Xbyak;
        if constexpr (std::is_same_v<R, float>) {
            movq(xmm0, ptr[rdi]);       /* pos0 <- x[0] */
            movq(xmm1, ptr[rdi + 16]);  /* pos1 <- x[2] */
            movq(xmm2, ptr[rdi + 8]);   /* pos2 <- x[1] */
            movq(xmm3, ptr[rdi + 24]);  /* pos3 <- x[3] */
            if (G_ != nullptr) {
                mulps(xmm0, ptr[rip + gDupLbl_[0]]);
                mulps(xmm1, ptr[rip + gDupLbl_[2]]);
                mulps(xmm2, ptr[rip + gDupLbl_[1]]);
                mulps(xmm3, ptr[rip + gDupLbl_[3]]);
            }
            movaps(xmm4, xmm0);
            addps(xmm4, xmm1);          /* p = x0+x2 */
            subps(xmm0, xmm1);          /* q = x0-x2 */
            movaps(xmm5, xmm2);
            addps(xmm5, xmm3);          /* r = x1+x3 */
            subps(xmm2, xmm3);          /* s = x1-x3 */
            movaps(xmm6, ptr[rip + signFlatLbl_]); /* (0, -0, 0, 0) */
            movaps(xmm7, xmm2);
            shufps(xmm7, xmm7, 0xB1);
            xorps(xmm7, xmm6);          /* (si, -sr) = (-i)*s */
            movaps(xmm1, xmm4);
            addps(xmm4, xmm5);          /* X0 = p+r */
            subps(xmm1, xmm5);          /* X2 = p-r */
            movaps(xmm3, xmm0);
            addps(xmm3, xmm7);          /* X1 = q+t */
            subps(xmm0, xmm7);          /* X3 = q-t */
            movlps(ptr[rdi], xmm4);
            movlps(ptr[rdi + 8], xmm3);
            movlps(ptr[rdi + 16], xmm1);
            movlps(ptr[rdi + 24], xmm0);
        } else {
            movupd(xmm0, ptr[rdi]);       /* pos0 <- x[0] */
            movupd(xmm1, ptr[rdi + 32]);  /* pos1 <- x[2] */
            movupd(xmm2, ptr[rdi + 16]);  /* pos2 <- x[1] */
            movupd(xmm3, ptr[rdi + 48]);  /* pos3 <- x[3] */
            movapd(xmm4, xmm0);
            addpd(xmm4, xmm1);            /* p = x0+x2 */
            subpd(xmm0, xmm1);            /* q = x0-x2 */
            movapd(xmm5, xmm2);
            addpd(xmm5, xmm3);            /* r = x1+x3 */
            subpd(xmm2, xmm3);            /* s = x1-x3 */
            movapd(xmm6, ptr[rip + signFlatLbl_]); /* (0.0, -0.0) */
            movapd(xmm7, xmm2);
            shufpd(xmm7, xmm7, 0x1);
            xorpd(xmm7, xmm6);            /* (si, -sr) = (-i)*s */
            movapd(xmm1, xmm4);
            addpd(xmm4, xmm5);            /* X0 = p+r */
            subpd(xmm1, xmm5);            /* X2 = p-r */
            movapd(xmm3, xmm0);
            addpd(xmm3, xmm7);            /* X1 = q+t */
            subpd(xmm0, xmm7);            /* X3 = q-t */
            movupd(ptr[rdi], xmm4);
            movupd(ptr[rdi + 16], xmm3);
            movupd(ptr[rdi + 32], xmm1);
            movupd(ptr[rdi + 48], xmm0);
        }
        ret();
    }

    void emitPairBodyFp32(int n) {
        using namespace Xbyak;
        const int N = 1 << n;
        const int nPairs = N / 2;

        const Xmm P[8] = {xmm0, xmm1, xmm2, xmm3,
                          xmm4, xmm5, xmm6, xmm7};
        const Xmm t0 = xmm8;
        const Xmm t1 = xmm9;
        const Xmm t2 = xmm10;
        const Xmm t3 = xmm11;
        const Xmm t4 = xmm12;
        const Xmm sgn = xmm14;

        for (int p = 0; p < nPairs; ++p) {
            const int ia = (int)bitrev((uint32_t)(2 * p), n);
            const int ib = (int)bitrev((uint32_t)(2 * p + 1), n);
            movq(P[p], ptr[rdi + (size_t)ia * 8]);
            movq(t0, ptr[rdi + (size_t)ib * 8]);
            if (G_ != nullptr) {
                mulps(P[p], ptr[rip + gDupLbl_[ia]]);
                mulps(t0, ptr[rip + gDupLbl_[ib]]);
            }
            movaps(t1, P[p]);
            addps(t1, t0);
            subps(P[p], t0);
            punpcklqdq(t1, P[p]);
            movaps(P[p], t1);
        }

        movaps(sgn, ptr[rip + sign2Lbl_]);

        for (int s = 2; s <= n; ++s) {
            const int m = 1 << s;
            const int h = m / 2;
            const int pairsPerBlock = m / 2;
            const int halfPairs = h / 2;

            if (s == 2) {
                for (int blk = 0; blk < N / m; ++blk) {
                    const int bp = blk * pairsPerBlock;
                    const Xmm &bot = P[bp];
                    const Xmm &top = P[bp + 1];
                    movaps(t0, top);
                    shufps(t0, top, 0xB1);
                    xorps(t0, sgn);
                    blendps(top, t0, 0x0C);
                    movaps(t1, bot);
                    addps(bot, top);
                    subps(t1, top);
                    movaps(top, t1);
                }
            } else {
                lea(r8, ptr[rip + stageWlbl_[s]]);
                for (int blk = 0; blk < N / m; ++blk) {
                    const int bp = blk * pairsPerBlock;
                    for (int q = 0; q < halfPairs; ++q) {
                        const Xmm &bot = P[bp + q];
                        const Xmm &top = P[bp + halfPairs + q];
                        movaps(t0, top);
                        movsldup(t0, t0);
                        movaps(t1, top);
                        movshdup(t1, t1);
                        movaps(t2, ptr[r8 + q * 16]);
                        movaps(t3, t2);
                        shufps(t3, t3, 0xB1);
                        mulps(t0, t2);
                        mulps(t1, t3);
                        addsubps(t0, t1);
                        movaps(t4, bot);
                        addps(bot, t0);
                        subps(t4, t0);
                        movaps(top, t4);
                    }
                }
            }
        }

        for (int p = 0; p < nPairs; ++p) {
            movups(ptr[rdi + (size_t)p * 16], P[p]);
        }
        ret();
    }

    /* ---- fp64 pair body: one ymm per complex PAIR ----
     * P[p] = (X[2p], X[2p+1]); low 128 = X[2p], high 128 = X[2p+1].
     * Load scratch xmm8/xmm9/xmm10 are overwritten before the stage loop
     * reuses ymm8..ymm14 as temps. */
    void emitPairBodyFp64(int n) {
        using namespace Xbyak;
        const int N = 1 << n;
        const int nPairs = N / 2;

        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};
        const Ymm t0 = ymm8;
        const Ymm t1 = ymm9;
        const Ymm t2 = ymm10;
        const Ymm t3 = ymm11;
        const Ymm t4 = ymm12;
        const Ymm sgn = ymm14;
        const Xmm a = xmm8;
        const Xmm b = xmm9;
        const Xmm s = xmm10;

        /* ---- load + stage m=2 fused ---- */
        for (int p = 0; p < nPairs; ++p) {
            const int ia = (int)bitrev((uint32_t)(2 * p), n);
            const int ib = (int)bitrev((uint32_t)(2 * p + 1), n);
            movupd(a, ptr[rdi + (size_t)ia * 16]);
            movupd(b, ptr[rdi + (size_t)ib * 16]);
            movapd(s, a);
            addpd(s, b);              /* s = a+b */
            subpd(a, b);              /* a = a-b */
            vinsertf128(P[p], P[p], s, 0);
            vinsertf128(P[p], P[p], a, 1);
        }

        vmovapd(sgn, ptr[rip + sign2Lbl_]); /* (0,-0,0,-0) */

        for (int s = 2; s <= n; ++s) {
            const int m = 1 << s;
            const int h = m / 2;
            const int pairsPerBlock = m / 2;
            const int halfPairs = h / 2;

            if (s == 2) {
                /* m=4: W = (1, -i) per 128-bit lane */
                for (int blk = 0; blk < N / m; ++blk) {
                    const int bp = blk * pairsPerBlock;
                    const Ymm &bot = P[bp];
                    const Ymm &top = P[bp + 1];
                    vshufpd(t0, top, top, 0x5); /* 0x5 = swap BOTH 128-bit lanes */
                    vxorpd(t0, t0, sgn);
                    vblendpd(top, top, t0, 0x0C);
                    vmovapd(t1, bot);
                    vaddpd(bot, top);
                    vsubpd(t1, top);
                    vmovapd(top, t1);
                }
            } else {
                /* m>=8: W pair chunk from the pool; complex product via
                 * vmulpd x2 + vhsubpd. */
                lea(r8, ptr[rip + stageWlbl_[s]]);
                lea(r9, ptr[rip + stageWnlbl_[s]]);
                for (int blk = 0; blk < N / m; ++blk) {
                    const int bp = blk * pairsPerBlock;
                    for (int q = 0; q < halfPairs; ++q) {
                        const Ymm &bot = P[bp + q];
                        const Ymm &top = P[bp + halfPairs + q];
                        vmovupd(t2, ptr[r8 + q * 32]);
                        vmovupd(t3, ptr[r9 + q * 32]);
                        vmulpd(t0, top, t2);
                        vmulpd(t1, top, t3);
                        vhsubpd(t0, t0, t1);
                        vmovapd(t4, bot);
                        vaddpd(bot, t0);
                        vsubpd(t4, t0);
                        vmovapd(top, t4);
                    }
                }
            }
        }

        /* ---- single store pass (natural order, in-place) ---- */
        for (int p = 0; p < nPairs; ++p) {
            vmovupd(ptr[rdi + (size_t)p * 32], P[p]);
        }
        vzeroupper();
        ret();
    }

    void emitPool(int n, const std::vector<std::vector<R>> &stageW,
                  const std::vector<std::vector<R>> &stageWn) {
        using namespace Xbyak;
        if constexpr (std::is_same_v<R, float>) {
            if (n == 2) {
                align(16);
                L(signFlatLbl_);
                dd(0u);
                dd(0x80000000u);
                dd(0u);
                dd(0u);
            }
            if (n >= 3) {
                align(16);
                L(sign2Lbl_);
                dd(sign2_bits(0));
                dd(sign2_bits(1));
                dd(sign2_bits(2));
                dd(sign2_bits(3));
            }
            for (int s = 3; s <= n; ++s) {
                align(16);
                L(stageWlbl_[s]);
                emit_reals_into(*this, stageW[s - 3]);
            }
            if (G_ != nullptr) {
                const int N = 1 << n;
                for (int i = 0; i < N; ++i) {
                    align(16);
                    L(gDupLbl_[i]);
                    dd(float_bits(G_[i]));
                    dd(float_bits(G_[i]));
                    dd(0u);
                    dd(0u);
                }
            }
        } else {
            if (n == 2) {
                align(16);
                L(signFlatLbl_);
                dq(0);
                dq(0x8000000000000000ull);
            }
            if (n >= 3) {
                align(32);
                L(sign2Lbl_);
                dq(0);
                dq(0x8000000000000000ull);
                dq(0);
                dq(0x8000000000000000ull);
            }
            for (int s = 3; s <= n; ++s) {
                align(32);
                L(stageWlbl_[s]);
                emit_reals_into(*this, stageW[s - 3]);
                align(32);
                L(stageWnlbl_[s]);
                emit_reals_into(*this, stageWn[s - 3]);
            }
        }
    }

    static uint32_t float_bits(float f) {
        uint32_t u;
        __builtin_memcpy(&u, &f, sizeof(u));
        return u;
    }

    static uint32_t sign2_bits(int idx) {
        const float s2[4] = {0.0f, 0.0f, 0.0f, -0.0f};
        return float_bits(s2[idx]);
    }

    Xbyak::Label signFlatLbl_;
    Xbyak::Label sign2Lbl_;
    Xbyak::Label stageWlbl_[16];
    Xbyak::Label stageWnlbl_[16];
    Xbyak::Label gDupLbl_[16];
    const float *G_;
};

/*
 * LUT-address path (n=9..15): reuse the promoted kernel structure.  Twiddles
 * + bitrev + sign2 live in an aligned_alloc block; addresses baked as movabs.
 * n=11..15 is a longer table on the same path; bitrev indices stay uint16_t
 * (N=32768 → max index 32767).  Precision-parameterized: fp64 uses a double
 * LUT (lut_d) and a 2-double sign2pd constant.
 */
template <typename R>
class LutEmitter final : public Xbyak::CodeGenerator {
public:
    LutEmitter(const uint16_t *bitrevTab, int nPairs, int n,
               const R *sign2, const R *const *stageW,
               const R *const *stageWn, const float *winDup)
        : Xbyak::CodeGenerator(131072) {
        using namespace Xbyak;
        const int N = 1 << n;

        if constexpr (std::is_same_v<R, float>) {
            emitBodyFp32(bitrevTab, nPairs, N, n, sign2, stageW, stageWn,
                         winDup);
        } else {
            emitBodyFp64(bitrevTab, nPairs, N, n, sign2, stageW, stageWn);
        }
    }

private:
    void emitBodyFp32(const uint16_t *bitrevTab, int nPairs, int N, int n,
                      const float *sign2, const float *const *stageW,
                      const float *const *stageWn, const float *winDup) {
        using namespace Xbyak;
        /* ---- bit-reversal ---- */
        mov(r8, reinterpret_cast<uint64_t>(bitrevTab));
        if (nPairs > 0) {
            xor_(ecx, ecx);
            Label brLoop;
            L(brLoop);
            movzx(eax, word[r8 + rcx * 4]);
            movzx(edx, word[r8 + rcx * 4 + 2]);
            movq(xmm0, ptr[rdi + rax * 8]);
            movq(xmm1, ptr[rdi + rdx * 8]);
            movq(ptr[rdi + rax * 8], xmm1);
            movq(ptr[rdi + rdx * 8], xmm0);
            add(ecx, 1);
            cmp(ecx, nPairs);
            jb(brLoop);
        }

        /* ---- stage 1 (window option b) ---- */
        if (winDup != nullptr) {
            mov(r11, reinterpret_cast<uint64_t>(winDup));
        }
        mov(rsi, rdi);
        lea(rdx, ptr[rdi + N * 8]);
        Label s1Loop;
        L(s1Loop);
        movups(xmm0, ptr[rsi]);
        if (winDup != nullptr) {
            movups(xmm1, ptr[r11]);
            mulps(xmm0, xmm1);
            add(r11, 16);
        }
        movaps(xmm2, xmm0);
        shufps(xmm2, xmm2, 0x4E);
        movaps(xmm3, xmm0);
        addps(xmm3, xmm2);
        subps(xmm0, xmm2);
        movlps(ptr[rsi], xmm3);
        movlps(ptr[rsi + 8], xmm0);
        add(rsi, 16);
        cmp(rsi, rdx);
        jb(s1Loop);

        /* ---- stage 2 ---- */
        if (n >= 2) {
            mov(r8, reinterpret_cast<uint64_t>(sign2));
            movaps(xmm7, ptr[r8]);
            mov(rsi, rdi);
            lea(rdx, ptr[rdi + N * 8]);
            Label s2Loop;
            L(s2Loop);
            movups(xmm0, ptr[rsi]);
            movups(xmm1, ptr[rsi + 16]);
            movaps(xmm2, xmm1);
            shufps(xmm2, xmm2, 0xB1);
            xorps(xmm2, xmm7);
            blendps(xmm1, xmm2, 0x0C);
            movaps(xmm3, xmm0);
            addps(xmm3, xmm1);
            subps(xmm0, xmm1);
            movups(ptr[rsi], xmm3);
            movups(ptr[rsi + 16], xmm0);
            add(rsi, 32);
            cmp(rsi, rdx);
            jb(s2Loop);
        }

        /* ---- stages 3..n ---- */
        for (int s = 3; s <= n; ++s) {
            const int h = 1 << (s - 1);
            mov(r8, reinterpret_cast<uint64_t>(stageW[s]));
            mov(r9, reinterpret_cast<uint64_t>(stageWn[s]));
            mov(rsi, rdi);
            lea(rdx, ptr[rdi + N * 8]);
            Label blockLoop;
            L(blockLoop);
            lea(r10, ptr[rsi + h * 8]);
            xor_(ecx, ecx);
            Label kLoop;
            L(kLoop);
            vmovups(ymm0, ptr[rsi + rcx * 8]);
            vmovups(ymm1, ptr[r10 + rcx * 8]);
            vmovups(ymm2, ptr[r8 + rcx * 8]);
            vmovups(ymm3, ptr[r9 + rcx * 8]);
            vmulps(ymm4, ymm1, ymm2);
            vmulps(ymm5, ymm1, ymm3);
            vhsubps(ymm4, ymm4, ymm5);
            vshufps(ymm4, ymm4, ymm4, 0xD8);
            vaddps(ymm6, ymm0, ymm4);
            vsubps(ymm7, ymm0, ymm4);
            vmovups(ptr[rsi + rcx * 8], ymm6);
            vmovups(ptr[r10 + rcx * 8], ymm7);
            add(ecx, 4);
            cmp(ecx, h);
            jb(kLoop);
            add(rsi, (1 << s) * 8);
            cmp(rsi, rdx);
            jb(blockLoop);
        }
        vzeroupper();
        ret();
    }

    void emitBodyFp64(const uint16_t *bitrevTab, int nPairs, int N, int n,
                      const double *sign2, const double *const *stageW,
                      const double *const *stageWn) {
        using namespace Xbyak;
        /* ---- bit-reversal (complex index ×16 bytes) ---- */
        mov(r8, reinterpret_cast<uint64_t>(bitrevTab));
        if (nPairs > 0) {
            xor_(ecx, ecx);
            Label brLoop;
            L(brLoop);
            movzx(eax, word[r8 + rcx * 4]);
            movzx(edx, word[r8 + rcx * 4 + 2]);
            shl(rax, 4);
            shl(rdx, 4);
            movupd(xmm0, ptr[rdi + rax]);
            movupd(xmm1, ptr[rdi + rdx]);
            movupd(ptr[rdi + rax], xmm1);
            movupd(ptr[rdi + rdx], xmm0);
            add(ecx, 1);
            cmp(ecx, nPairs);
            jb(brLoop);
        }

        /* ---- stage 1 ---- */
        mov(rsi, rdi);
        lea(rdx, ptr[rdi + N * 16]);
        Label s1Loop;
        L(s1Loop);
        movupd(xmm0, ptr[rsi]);
        movupd(xmm1, ptr[rsi + 16]);
        movapd(xmm2, xmm0);
        addpd(xmm2, xmm1);
        subpd(xmm0, xmm1);
        movupd(ptr[rsi], xmm2);
        movupd(ptr[rsi + 16], xmm0);
        add(rsi, 32);
        cmp(rsi, rdx);
        jb(s1Loop);

        /* ---- stage 2 ---- */
        if (n >= 2) {
            mov(r8, reinterpret_cast<uint64_t>(sign2));
            movapd(xmm7, ptr[r8]); /* (0.0, -0.0) */
            mov(rsi, rdi);
            lea(rdx, ptr[rdi + N * 16]);
            Label s2Loop;
            L(s2Loop);
            movupd(xmm0, ptr[rsi]);
            movupd(xmm1, ptr[rsi + 16]);
            movupd(xmm2, ptr[rsi + 32]);
            movupd(xmm3, ptr[rsi + 48]);
            movapd(xmm4, xmm3);
            shufpd(xmm4, xmm4, 0x1);
            xorpd(xmm4, xmm7);
            movapd(xmm5, xmm0);
            addpd(xmm5, xmm2);
            subpd(xmm0, xmm2);
            movapd(xmm6, xmm1);
            addpd(xmm6, xmm4);
            subpd(xmm1, xmm4);
            movupd(ptr[rsi], xmm5);
            movupd(ptr[rsi + 16], xmm6);
            movupd(ptr[rsi + 32], xmm0);
            movupd(ptr[rsi + 48], xmm1);
            add(rsi, 64);
            cmp(rsi, rdx);
            jb(s2Loop);
        }

        /* ---- stages 3..n ---- */
        for (int s = 3; s <= n; ++s) {
            const int h = 1 << (s - 1);
            mov(r8, reinterpret_cast<uint64_t>(stageW[s]));
            mov(r9, reinterpret_cast<uint64_t>(stageWn[s]));
            mov(rsi, rdi);
            lea(rdx, ptr[rdi + N * 16]);
            Label blockLoop;
            L(blockLoop);
            lea(r10, ptr[rsi + h * 16]);
            xor_(ecx, ecx);
            Label kLoop;
            L(kLoop);
            vmovupd(ymm0, ptr[rsi + rcx * 8]);
            vmovupd(ymm1, ptr[r10 + rcx * 8]);
            vmovupd(ymm2, ptr[r8 + rcx * 8]);
            vmovupd(ymm3, ptr[r9 + rcx * 8]);
            vmulpd(ymm4, ymm1, ymm2);
            vmulpd(ymm5, ymm1, ymm3);
            vhsubpd(ymm4, ymm4, ymm5);
            vaddpd(ymm6, ymm0, ymm4);
            vsubpd(ymm7, ymm0, ymm4);
            vmovupd(ptr[rsi + rcx * 8], ymm6);
            vmovupd(ptr[r10 + rcx * 8], ymm7);
            add(ecx, 4);
            cmp(ecx, 2 * h);
            jb(kLoop);
            add(rsi, (1 << s) * 16);
            cmp(rsi, rdx);
            jb(blockLoop);
        }
        vzeroupper();
        ret();
    }
};

/*
 * SchedEmitter — scheduling-refactored fp64 (pd) kernel for N >= 32
 * (topic jit-mixed-radix-codelet R0 schedule + R1 codelet instruction
 * quality + R2 memory-side levers; pd path ONLY — the fp32 ps path keeps
 * Small/Pool/Lut untouched).
 *
 * Schedule (R0/R1): N = r1*...*rk; stage 0 fuses the digit-reversal
 * permutation into the first codelet sweep (out-of-place into the plan
 * scratch via a sequential inverse-digitrev table); stages 1..k-2 run
 * in-place on the scratch; the final stage reads the scratch and writes
 * the natural array directly (ping-pong closed in place of a copy pass).
 * Stage t processes contiguous blocks of B_t = r1*..*r_t; within a
 * block, leaf (block, k) gathers r_t values at stride B_{t-1} and runs
 * the whole DFT-r_t in registers.
 *
 * R2 F6 adjacent-level cache blocking: consecutive stages t..t+d-1 can
 * be merged into one L1-resident TILE group (tile = B_{t+d-1} elements,
 * 16 B/element, budget-derived at plan time).  The tile loop sweeps the
 * group's stages confined to each tile before advancing, so the
 * intermediate result between adjacent levels never round-trips
 * main-array-wide through L1 (pure loop interchange over disjoint
 * tiles — bitwise-identical results).  A group may include stage 0: the
 * digitrev gather pipeline writes a tile then immediately runs the
 * following stages inside the same tile.
 *
 * R2 F7 write-back / prefetch re-evaluation:
 *   - vmovntpd non-temporal stores ONLY for sweep outputs whose next
 *     read is not tile-local (unblocked full sweeps + the last stage of
 *     a tile group); the final natural-order store pass NEVER uses NT
 *     (the caller reads the result right after execute).  sfence is
 *     emitted at every boundary where NT stores precede ordinary loads
 *     of the same lines (x86 WC ordering rule).  Auto-gated to N >= 4096
 *     (below that the whole array is L1-resident and NT would only hurt).
 *   - optional prefetcht0 of the NEXT tile head (8 cache lines) at tile
 *     loop boundaries (measured knob; the old prefetch falsification
 *     targeted a radix-2 streaming structure, not this tile structure).
 *
 * R1 leaf shapes (DELTA F4 — codelet instruction quality):
 *
 *   - radix-{2,4,8,16} stages run VECTOR-2 leaves (FFTW n1fv style): one
 *     ymm holds the SAME DIT position of TWO adjacent instances of the
 *     parallel dimension — (k, k+1) for twiddled stages, blocks (2b, 2b+1)
 *     for the stage-0 gather — so every load, twiddle multiply, butterfly
 *     and store instruction covers two instances.  (k, k+1) pairs are
 *     adjacent and 32B-aligned (k even, Bprev even): twiddled stages load
 *     AND store whole 32B yms at stride Bprev.
 *   - radix-16 vector-2 = DFT-8(even slots) + DFT-8(odd slots) + W_16
 *     cross with a 256B L1 spill frame (16 positions x 2 instances need
 *     more than the 16 ymm available; the spill stays L1-resident and the
 *     main array is never touched mid-transform).
 *   - radix-32 stays single-k (pair-ymm lanes = adjacent DIT positions of
 *     ONE k; 32 instances x 2 positions would need 32 ymm) with the same
 *     R1 micro-optimizations.
 *
 * R1 instruction moves (vs R0):
 *   - butterflies: vsubpd-first + vaddpd = 2 instructions (R0 paid two
 *     extra vmovapd per butterfly to save/restore the bottom value);
 *   - twiddle multiplies read BOTH the table halves / rip-pool constants
 *     as memory operands (vmulpd + vfmaddsub231pd with [mem]); the rip
 *     pool carries pre-swapped (wi, wr) twins so the FMA path never
 *     shuffles W at runtime;
 *   - constant folding: W = -i slots (exponent = m/4) become
 *     vshufpd+vxorpd sign flips (lane-uniform in the vector-2 layout);
 *   - the stage-0 gather table stores PRE-SHIFTED 32-bit byte offsets:
 *     mov(dword) + vmovupd = 2 instructions per gathered value (R0 paid
 *     movzx + shl + vmovupd);
 *   - k-loops walk pointers (add) instead of recomputing (mov+shl+add),
 *     and vector-2 stages halve the k-loop trip count.
 *
 * Twiddle discipline (R0 F3, unchanged in spirit): per-stage twiddles
 * TW_t live in one LUT block — 64B slots per (k-pair, j) for vector-2
 * rows ([W^{jk}, W^{j(k+1)} | swap-twin]) and 32B slots per (k, j) for
 * radix-32 rows ([W^{jk} | twin]); the FMA path loads only W (the slot
 * carries the pre-swapped twin half), the non-FMA path derives the
 * conjugate in registers (vshufpd+vxorpd).  Leaf-internal stage twiddles
 * are rip-pool constants shared by every leaf of the stage.  FMA3 is
 * emitted only when the plan-time CPU feature check passed
 * ([[CON-FFT-002]] — MUST NOT emit FMA3 without FMA).
 *
 * Register conventions (caller-saved only, SysV AMD64; rbp pushed and a
 * <=272B frame opened when any radix-16/32 leaf (spill) or kStages > 1
 * (final-stage cursors) needs it; rbx/r12/r13 pushed when kStages > 1):
 *   rdi = io (stage 0) / scratch (stages >= 1) base; rsi = block cursor;
 *   r8 = gather-tab cursor (stage 0 plain) / sweep end (stages >= 1);
 *   r9 = TW_t base; r10 = TW row for the current k; r11 = k-loop end;
 *   rax = leaf load base; rdx = parked natural base; r12 = final-stage
 *   natural block cursor / tile-group tile cursor; r13 = gather-tab
 *   cursor inside a stage-0 tile group (survives the in-tile sweeps);
 *   rcx = tile-group array end; rbx = final-stage natural k(-pair) base.
 *   ymm0..7 = leaf data, ymm8..15 = scratch, ymm14 = sign mask.
 */
class SchedEmitter final : public Xbyak::CodeGenerator {
public:
    struct Stage {
        int radix;          /* r_t */
        int bprev;          /* B_{t-1} in elements (t=0 -> 1) */
        int b;              /* B_t in elements */
        const double *tw;   /* TW_t base, nullptr for t=0 */
    };

    /* groupLen[t] = d >= 2: stages t..t+d-1 run as one L1-resident tile
     * group with tile = stages[t+d-1].b elements (R2 F6).  groupLen
     * values of 0/1 = unblocked full-array sweep (R0/R1 shape); the
     * final stage is never part of a tile group.  nt / prefetch = R2 F7
     * emission flags (topic-local plan knobs, see fft_jit_kernel_create).
     * The nt flag is auto-gated to N >= 4096 (whole array must exceed
     * L1d before bypassing it can pay). */
    SchedEmitter(int n, const uint32_t *gatherTab, const double *scratch,
                 const Stage *stages, int kStages, bool fma,
                 const int *groupLen, bool nt, int prefetch)
        : Xbyak::CodeGenerator(262144), fma_(fma),
          nt_(nt && ((1 << n) >= 4096)), prefetch_(prefetch) {
        int gl[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (groupLen != nullptr) {
            for (int i = 0; i < 8; ++i) {
                gl[i] = groupLen[i];
            }
        }
        emitBody(n, gatherTab, scratch, stages, kStages, gl);
        emitPool();
    }

private:
    static int lf_log2(int v) {
        int s = 0;
        while ((1 << s) < v) {
            ++s;
        }
        return s;
    }

    /* ---- body ---- */
    void emitBody(int n, const uint32_t *gatherTab, const double *scratch,
                  const Stage *stages, int kStages, const int *groupLen) {
        using namespace Xbyak;
        const int N = 1 << n;

        bool needSpill = false; /* radix-16 v2 cross / radix-32 subleaf */
        for (int t = 0; t < kStages; ++t) {
            if (stages[t].radix >= 16) {
                needSpill = true;
            }
        }
        const bool needCursors = (kStages > 1);
        /* spill base keeps clear of the pushed rbx/r12/r13 slots: with
         * r13 pushed at [rbp-24], the 8x32B spill frame must start at
         * rbp-288 or below (R1's -272 predates the r13 push and would
         * overlap the saved r13 with its j=7 slot). */
        spillOff_ = needCursors ? -288 : -256;
        if (needSpill || needCursors) {
            push(rbp);
            mov(rbp, rsp);
            if (needCursors) {
                push(rbx);
                push(r12);
                push(r13); /* r13 = stage-0-group gather-tab cursor */
            }
            if (needSpill) {
                sub(rsp, 272);
            }
        }

        if (kStages == 1) {
            /* whole transform = one codelet (N == r1 == 32) */
            emitLeaf32(rdi, 16, false, rdi);
        } else {
            int t = 0;
            if (groupLen[0] >= 2) {
                /* R2 F6: stage 0 leads a tile group — per-tile gather
                 * pipeline (gather tile -> in-tile sweeps), the tile
                 * stays L1-resident from the gather stores through the
                 * last in-tile sweep. */
                emitStage0Group(N, gatherTab, scratch, stages,
                                groupLen[0]);
                t = groupLen[0];
            } else {
                emitStage0Plain(N, gatherTab, scratch, stages[0]);
                t = 1;
            }
            while (t < kStages - 1) {
                const int d = groupLen[t] >= 2 ? groupLen[t] : 1;
                if (d >= 2) {
                    emitTileGroup(stages, t, d, N);
                } else {
                    emitSweepFull(stages[t], N, /*allowNt=*/true);
                }
                t += d;
            }
            /* final stage: reads the scratch, writes the natural array
             * directly (ping-pong closed in place of a copy pass).
             * Stores are NEVER non-temporal (the caller reads the
             * result immediately after execute). */
            const Stage &st = stages[kStages - 1];
            mov(r9, reinterpret_cast<uint64_t>(st.tw));
            mov(rsi, rdi);
            lea(r8, ptr[rdi + N * 16]); /* end (r8 table dead) */
            mov(r12, rdx);             /* natural block cursor */
            Label blk;
            L(blk);
            lea(r11, ptr[rsi + st.bprev * 16]); /* k-loop end */
            mov(rax, rsi);
            mov(r10, r9);
            Label kLoop;
            L(kLoop);
            if (st.radix <= 16) {
                /* vector-2 over k pairs (k, k+1) */
                mov(rbx, r12);
                emitLeafV2(st.radix, rax, st.bprev * 16, rbx, true);
                add(rax, 32);
                add(r10, st.radix * 64);
                add(r12, 32);
            } else {
                mov(rbx, r12);
                emitLeaf32(rax, st.bprev * 16, true, rbx);
                add(rax, 16);
                add(r10, 32 * 32);
                add(r12, 16);
            }
            cmp(rax, r11);
            jb(kLoop);
            add(rsi, st.b * 16);
            add(r12, st.b * 16);
            cmp(rsi, r8);
            jb(blk);
        }

        vzeroupper();
        if (needCursors) {
            lea(rsp, ptr[rbp - 24]); /* at the pushed r13 slot */
            pop(r13);
            pop(r12);
            pop(rbx);
            pop(rbp);
        } else if (needSpill) {
            lea(rsp, ptr[rbp]);
            pop(rbp);
        }
        ret();
    }

    /* ---- stage 0, unblocked (R0/R1 shape): digitrev gather fused into
     * the first codelet sweep, out-of-place into the scratch ---- */
    void emitStage0Plain(int N, const uint32_t *gatherTab,
                         const double *scratch, const Stage &st) {
        using namespace Xbyak;
        mov(r8, reinterpret_cast<uint64_t>(gatherTab));
        mov(rsi, reinterpret_cast<uint64_t>(scratch));
        lea(rdx, ptr[rsi + N * 16]); /* scratch end */
        ntSweep_ = nt_; /* scratch stores: re-read only by a later full
                           sweep -> NT candidate (R2 F7) */
        Label blk;
        L(blk);
        if (st.radix <= 16) {
            /* vector-2 over block pairs (2b, 2b+1) */
            gatherTab_ = &r8;
            emitLeafV2(st.radix, rsi, 16, rsi, false);
            gatherTab_ = nullptr;
            add(r8, st.radix * 8);   /* 2 table rows (uint32) */
            add(rsi, st.radix * 32); /* 2 blocks */
        } else {
            gatherTab_ = &r8;
            emitLeaf32(r8, 16, false, rsi);
            gatherTab_ = nullptr;
            add(r8, st.radix * 4);
            add(rsi, st.radix * 16);
        }
        cmp(rsi, rdx);
        jb(blk);
        ntSweep_ = false;
        if (nt_) {
            sfence(); /* NT scratch writes precede later loads (F7) */
        }
        /* stages 1..k-2 run in-place on the scratch; park the natural
         * base in rdx (stage-0 loop is done) */
        mov(rdx, rdi);
        mov(rdi, reinterpret_cast<uint64_t>(scratch));
    }

    /* ---- R2 F6: stage-0 tile group.  Per tile: gather exactly the
     * tile's digitrev rows (tab cursor r13 stays sequential across
     * tiles), write the tile in the scratch, then run stages 1..d-1
     * confined to the tile before advancing (intermediate results never
     * leave L1). ---- */
    void emitStage0Group(int N, const uint32_t *gatherTab,
                         const double *scratch, const Stage *stages,
                         int d) {
        using namespace Xbyak;
        const Stage &st = stages[0];
        const int T = stages[d - 1].b; /* tile elements */
        bool ntUsed = false;
        mov(r13, reinterpret_cast<uint64_t>(gatherTab)); /* tab cursor */
        mov(r12, reinterpret_cast<uint64_t>(scratch));   /* tile cursor */
        lea(rcx, ptr[r12 + N * 16]); /* array end (tile-loop bound) */
        Label tile;
        L(tile);
        emitPrefetchHead(r12, T * 16);
        mov(rsi, r12);
        lea(rdx, ptr[r12 + T * 16]); /* this tile's scratch end */
        Label blk;
        L(blk);
        if (st.radix <= 16) {
            gatherTab_ = &r13;
            emitLeafV2(st.radix, rsi, 16, rsi, false);
            gatherTab_ = nullptr;
            add(r13, st.radix * 8);  /* 2 table rows (uint32) */
            add(rsi, st.radix * 32); /* 2 blocks */
        } else {
            gatherTab_ = &r13;
            emitLeaf32(r13, 16, false, rsi);
            gatherTab_ = nullptr;
            add(r13, st.radix * 4);
            add(rsi, st.radix * 16);
        }
        cmp(rsi, rdx);
        jb(blk);
        for (int s = 1; s < d; ++s) {
            const bool last = (s == d - 1);
            emitSweepInTile(stages[s], T * 16, last);
            ntUsed |= (nt_ && last);
        }
        add(r12, T * 16);
        cmp(r12, rcx);
        jb(tile);
        if (ntUsed) {
            sfence(); /* last-in-group NT writes precede later loads */
        }
        /* park the natural base; the scratch becomes the working array */
        mov(rdx, rdi);
        mov(rdi, reinterpret_cast<uint64_t>(scratch));
    }

    /* ---- one stage's sweep confined to the current tile ([r12,
     * r12+tileBytes)); in-tile stores stay WB (they are re-read from L1
     * inside the tile) EXCEPT the last stage of the group (its output is
     * consumed only by a later full sweep -> NT candidate). ---- */
    void emitSweepInTile(const Stage &st, int tileBytes, bool lastOfGroup) {
        using namespace Xbyak;
        ntSweep_ = nt_ && lastOfGroup;
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, r12);
        lea(r8, ptr[r12 + tileBytes]);
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 16]); /* k-loop end */
        mov(rax, rsi);
        mov(r10, r9);
        Label kLoop;
        L(kLoop);
        if (st.radix <= 16) {
            /* vector-2 over k pairs (k, k+1) */
            emitLeafV2(st.radix, rax, st.bprev * 16, rax, true);
            add(rax, 32);
            add(r10, st.radix * 64);
        } else {
            emitLeaf32(rax, st.bprev * 16, true, rax);
            add(rax, 16);
            add(r10, 32 * 32);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 16);
        cmp(rsi, r8);
        jb(blk);
        ntSweep_ = false;
    }

    /* ---- one stage's unblocked full-array sweep (R0/R1 shape); NT
     * stores + trailing sfence when enabled (output re-read only by a
     * later full sweep). ---- */
    void emitSweepFull(const Stage &st, int N, bool allowNt) {
        using namespace Xbyak;
        ntSweep_ = nt_ && allowNt;
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, rdi);
        lea(r8, ptr[rdi + N * 16]); /* end (r8 table dead) */
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 16]); /* k-loop end */
        mov(rax, rsi);
        mov(r10, r9);
        Label kLoop;
        L(kLoop);
        if (st.radix <= 16) {
            /* vector-2 over k pairs (k, k+1) */
            emitLeafV2(st.radix, rax, st.bprev * 16, rax, true);
            add(rax, 32);
            add(r10, st.radix * 64);
        } else {
            emitLeaf32(rax, st.bprev * 16, true, rax);
            add(rax, 16);
            add(r10, 32 * 32);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 16);
        cmp(rsi, r8);
        jb(blk);
        ntSweep_ = false;
        if (nt_ && allowNt) {
            sfence();
        }
    }

    /* ---- R2 F6: middle tile group (stages t..t+d-1, t >= 1) ---- */
    void emitTileGroup(const Stage *stages, int t, int d, int N) {
        using namespace Xbyak;
        const int T = stages[t + d - 1].b; /* tile elements */
        bool ntUsed = false;
        mov(r12, rdi); /* tile cursor = scratch base */
        lea(rcx, ptr[rdi + N * 16]);
        Label tile;
        L(tile);
        emitPrefetchHead(r12, T * 16);
        for (int s = t; s < t + d; ++s) {
            const bool last = (s == t + d - 1);
            emitSweepInTile(stages[s], T * 16, last);
            ntUsed |= (nt_ && last);
        }
        add(r12, T * 16);
        cmp(r12, rcx);
        jb(tile);
        if (ntUsed) {
            sfence(); /* last-in-group NT writes precede later loads */
        }
    }

    /* ---- R2 F7: prefetcht0 the head (8 cache lines) of the NEXT tile
     * at every tile-loop boundary (knob-gated; prefetch of an address
     * past the array end is architecturally harmless). ---- */
    void emitPrefetchHead(const Xbyak::Reg64 &base, int tileBytes) {
        if (prefetch_ <= 0) {
            return;
        }
        for (int i = 0; i < 8; ++i) {
            prefetcht0(ptr[base + tileBytes + i * 64]);
        }
    }

    /* ---- vector-2 leaf dispatch (radix 2/4/8/16) ----
     * Loads come from loadBase + slot*strideBytes (twiddled stages) or
     * through the gatherTab_ cursor (stage 0); stores go to storeBase +
     * pos*strideBytes (twiddled) or storeBase + pos*16 / (r+pos)*16 for
     * the two gathered blocks (stage 0, contiguous scratch).  Twiddle
     * rows (hasTw) come from r10: 64B slots per (k-pair, j). */
    void emitLeafV2(int r, const Xbyak::Reg64 &loadBase, int strideBytes,
                    const Xbyak::Reg64 &storeBase, bool hasTw) {
        using namespace Xbyak;
        if (r >= 4 || (!fma_ && hasTw)) {
            vmovapd(ymm14, ptr[rip + sign2Lbl_]);
        }
        if (r == 16) {
            emitV2Sub(8, loadBase, strideBytes, hasTw, 0);
            for (int j = 0; j < 8; ++j) {
                vmovupd(ptr[rbp + spillOff_ + j * 32], Ymm(j));
            }
            emitV2Sub(8, loadBase, strideBytes, hasTw, 1);
            cross16V2(storeBase, strideBytes);
        } else {
            emitV2Sub(r, loadBase, strideBytes, hasTw, -1);
            storeV2(r, storeBase, strideBytes);
        }
    }

    /* r-point DFT in per-position vector-2 registers P[0..r-1] (r <= 8).
     * par < 0: whole leaf (slot(pos) = bitrev(pos, log2 r));
     * par >= 0: sub-leaf of the radix-16 leaf (slot(pos) =
     * 2*bitrev(pos, 3) + par; gather rows stride 16). */
    void emitV2Sub(int r, const Xbyak::Reg64 &loadBase, int strideBytes,
                   bool hasTw, int par) {
        using namespace Xbyak;
        const int nlf = lf_log2(r);
        const int rr = (par >= 0) ? 16 : r; /* gather row stride (r1) */
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};

        /* ---- load (+twiddle) + m=2 fuse ---- */
        for (int u = 0; u < r / 2; ++u) {
            int sa, sb;
            if (par >= 0) {
                sa = 2 * (int)bitrev((uint32_t)(2 * u), nlf) + par;
                sb = 2 * (int)bitrev((uint32_t)(2 * u + 1), nlf) + par;
            } else {
                sa = (int)bitrev((uint32_t)(2 * u), nlf);
                sb = (int)bitrev((uint32_t)(2 * u + 1), nlf);
            }
            if (gatherTab_ != nullptr) {
                const Reg64 &tab = *gatherTab_;
                mov(eax, dword[tab + sa * 4]);
                vmovupd(xmm8, ptr[rdi + rax]);
                mov(eax, dword[tab + (rr + sa) * 4]);
                vmovupd(xmm9, ptr[rdi + rax]);
                mov(eax, dword[tab + sb * 4]);
                vmovupd(xmm10, ptr[rdi + rax]);
                mov(eax, dword[tab + (rr + sb) * 4]);
                vmovupd(xmm11, ptr[rdi + rax]);
                /* xmm14 holds the sign mask (ymm14) — the second
                 * block's fuse reuses the now-dead load regs */
                vaddpd(xmm12, xmm8, xmm10);
                vsubpd(xmm13, xmm8, xmm10);
                vaddpd(xmm8, xmm9, xmm11);
                vsubpd(xmm10, xmm9, xmm11);
                vinsertf128(P[2 * u], P[2 * u], xmm12, 0);
                vinsertf128(P[2 * u], P[2 * u], xmm8, 1);
                vinsertf128(P[2 * u + 1], P[2 * u + 1], xmm13, 0);
                vinsertf128(P[2 * u + 1], P[2 * u + 1], xmm10, 1);
            } else {
                loadTwV2(ymm10, ptr[loadBase + sa * strideBytes], sa, hasTw);
                loadTwV2(ymm11, ptr[loadBase + sb * strideBytes], sb, hasTw);
                vaddpd(P[2 * u], ymm10, ymm11);
                vsubpd(P[2 * u + 1], ymm10, ymm11);
            }
        }

        /* ---- stages 2..nlf (butterflies pair positions (i, i+h)) ---- */
        for (int st = 2; st <= nlf; ++st) {
            const int m = 1 << st;
            const int h = m >> 1;
            for (int blk = 0; blk < r / m; ++blk) {
                for (int i = 0; i < h; ++i) {
                    const Ymm &bot = P[blk * m + i];
                    const Ymm &top = P[blk * m + h + i];
                    if (i == 0) {
                        /* T = top in place: diff via scratch */
                        vsubpd(ymm12, bot, top);
                        vaddpd(bot, bot, top);
                        vmovapd(top, ymm12);
                    } else if (i == (m >> 2)) {
                        /* W_m^{m/4} = -i: swap + sign flip */
                        vshufpd(ymm12, top, top, 0x5);
                        vxorpd(ymm12, ymm12, ymm14);
                        vsubpd(top, bot, ymm12);
                        vaddpd(bot, bot, ymm12);
                    } else {
                        mulTwPoolV2(top, st, i);
                        vsubpd(top, bot, ymm12);
                        vaddpd(bot, bot, ymm12);
                    }
                }
            }
        }
    }

    /* load the (k, k+1) pair at addr; twiddle by TW row slot (64B slots
     * at r10 + slot*64) when slot != 0; result in dst (ymm10/ymm11). */
    void loadTwV2(const Xbyak::Ymm &dst, const Xbyak::Operand &addr,
                  int slot, bool hasTw) {
        using namespace Xbyak;
        if (slot == 0 || !hasTw) {
            vmovupd(dst, addr);
            return;
        }
        vmovupd(ymm8, addr);
        if (fma_) {
            vmovddup(ymm9, ymm8);            /* (ar, ar) per lane */
            vshufpd(dst, ymm8, ymm8, 0xF);   /* (ai, ai) per lane */
            vmulpd(dst, dst, ptr[r10 + slot * 64 + 32]);
            vfmaddsub231pd(dst, ymm9, ptr[r10 + slot * 64]);
        } else {
            vmovupd(ymm12, ptr[r10 + slot * 64]); /* W pair */
            vshufpd(ymm13, ymm12, ymm12, 0x5);
            vxorpd(ymm13, ymm13, ymm14);          /* (wi, -wr) */
            vmulpd(dst, ymm8, ymm12);
            vmulpd(ymm13, ymm8, ymm13);
            vhsubpd(dst, dst, ymm13);
        }
    }

    /* T (ymm12) = top * W_{2^st}^e from the rip pool (broadcast const +
     * pre-swapped twin as memory operands; non-FMA derives the conjugate
     * in registers). */
    void mulTwPoolV2(const Xbyak::Ymm &top, int st, int e) {
        using namespace Xbyak;
        if (fma_) {
            vmovddup(ymm13, top);
            vshufpd(ymm12, top, top, 0xF);
            vmulpd(ymm12, ymm12, ptr[rip + twSwapLbl_[st][e]]);
            vfmaddsub231pd(ymm12, ymm13, ptr[rip + twLbl_[st][e]]);
        } else {
            vmovapd(ymm13, ptr[rip + twLbl_[st][e]]);
            vshufpd(ymm15, ymm13, ymm13, 0x5);
            vxorpd(ymm15, ymm15, ymm14);
            vmulpd(ymm12, top, ymm13);
            vmulpd(ymm15, top, ymm15);
            vhsubpd(ymm12, ymm12, ymm15);
        }
    }

    /* radix-16 cross: y[j] = e[j] + W_16^j o[j], y[j+8] = e[j] - W_16^j o[j]
     * (j = 0..7).  e[] reloaded from the spill frame, o[] in P. */
    void cross16V2(const Xbyak::Reg64 &storeBase, int strideBytes) {
        using namespace Xbyak;
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};
        for (int j = 0; j < 8; ++j) {
            vmovupd(ymm8, ptr[rbp + spillOff_ + j * 32]); /* e[j] */
            if (j == 0) {
                vaddpd(ymm10, ymm8, P[0]);
                vsubpd(ymm11, ymm8, P[0]);
            } else if (j == 4) {
                /* W_16^4 = -i */
                vshufpd(ymm12, P[4], P[4], 0x5);
                vxorpd(ymm12, ymm12, ymm14);
                vaddpd(ymm10, ymm8, ymm12);
                vsubpd(ymm11, ymm8, ymm12);
            } else {
                mulTwPoolV2(P[j], 4, j); /* W_16^j = W_{2^4}^j */
                vaddpd(ymm10, ymm8, ymm12);
                vsubpd(ymm11, ymm8, ymm12);
            }
            storeV2One(ymm10, j, storeBase, strideBytes, 16);
            storeV2One(ymm11, j + 8, storeBase, strideBytes, 16);
        }
    }

    void storeV2(int r, const Xbyak::Reg64 &storeBase, int strideBytes) {
        using namespace Xbyak;
        for (int p = 0; p < r; ++p) {
            storeV2One(Ymm(p), p, storeBase, strideBytes, r);
        }
    }

    /* store position pos: whole 32B ym at storeBase + pos*stride (normal
     * / final stages), or per-block 16B extracts into the two gathered
     * blocks (stage 0: storeBase + pos*16 and storeBase + (r+pos)*16).
     * R2 F7: when the sweep is NT-eligible the stores are vmovntpd
     * (non-temporal; the sfence lives at the sweep/tile boundary). */
    void storeV2One(const Xbyak::Ymm &v, int pos,
                    const Xbyak::Reg64 &storeBase, int strideBytes, int r) {
        using namespace Xbyak;
        if (gatherTab_ != nullptr) {
            vextractf128(xmm12, v, 0);
            if (ntSweep_) {
                vmovntpd(ptr[storeBase + pos * 16], xmm12);
            } else {
                vmovupd(ptr[storeBase + pos * 16], xmm12);
            }
            vextractf128(xmm12, v, 1);
            if (ntSweep_) {
                vmovntpd(ptr[storeBase + (r + pos) * 16], xmm12);
            } else {
                vmovupd(ptr[storeBase + (r + pos) * 16], xmm12);
            }
        } else if (ntSweep_) {
            vmovntpd(ptr[storeBase + pos * strideBytes], v);
        } else {
            vmovupd(ptr[storeBase + pos * strideBytes], v);
        }
    }

    /* ---- single-k radix-32 leaf (pair-ymm lanes = adjacent DIT
     * positions of one k; vector-2 would need 32 live yms) ---- */
    void emitLeaf32(const Xbyak::Reg64 &loadBase, int strideBytes,
                    bool hasTw, const Xbyak::Reg64 &storeBase) {
        using namespace Xbyak;
        vmovapd(ymm14, ptr[rip + sign2Lbl_]);
        emitSubLeaf16Sk(loadBase, strideBytes, hasTw, 0);
        for (int p = 0; p < 8; ++p) {
            vmovupd(ptr[rbp + spillOff_ + p * 32], Ymm(p));
        }
        emitSubLeaf16Sk(loadBase, strideBytes, hasTw, 1);
        /* cross: y[k] = e[k] + W_32^k o[k], pair chunks (W^{2p}, W^{2p+1})
         * from the rip pool (mixing foldable and general lanes — the
         * chunked multiply stays; e-pair reloaded from the spill frame) */
        for (int p = 0; p < 8; ++p) {
            vmovupd(ymm9, ptr[rbp + spillOff_ + p * 32]); /* A pair */
            if (fma_) {
                vmovddup(ymm11, Ymm(p));
                vshufpd(ymm8, Ymm(p), Ymm(p), 0xF);
                vmulpd(ymm8, ymm8, ptr[rip + w32SwapLbl_[p]]);
                vfmaddsub231pd(ymm8, ymm11, ptr[rip + w32Lbl_[p]]);
            } else {
                vmovupd(ymm10, ptr[rip + w32Lbl_[p]]);
                vshufpd(ymm12, ymm10, ymm10, 0x5);
                vxorpd(ymm12, ymm12, ymm14);
                vmulpd(ymm8, Ymm(p), ymm10);
                vmulpd(ymm12, Ymm(p), ymm12);
                vhsubpd(ymm8, ymm8, ymm12);
            }
            vaddpd(ymm13, ymm9, ymm8);  /* y[k]    -> slots (2p, 2p+1) */
            vsubpd(ymm15, ymm9, ymm8);  /* y[k+16] -> (2p+16, 2p+17);
                                           ymm14 = sgn must stay live */
            if (strideBytes == 16) {
                /* stage-0 gather (or the N==32 single-codelet transform,
                 * where ntSweep_ is never set) */
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p) * 16], ymm13);
                    vmovntpd(ptr[storeBase + (2 * p + 16) * 16], ymm15);
                } else {
                    vmovupd(ptr[storeBase + (2 * p) * 16], ymm13);
                    vmovupd(ptr[storeBase + (2 * p + 16) * 16], ymm15);
                }
            } else {
                vextractf128(xmm8, ymm13, 0);
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p) * strideBytes], xmm8);
                } else {
                    vmovupd(ptr[storeBase + (2 * p) * strideBytes], xmm8);
                }
                vextractf128(xmm8, ymm13, 1);
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p + 1) * strideBytes],
                             xmm8);
                } else {
                    vmovupd(ptr[storeBase + (2 * p + 1) * strideBytes],
                            xmm8);
                }
                vextractf128(xmm8, ymm15, 0);
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p + 16) * strideBytes],
                             xmm8);
                } else {
                    vmovupd(ptr[storeBase + (2 * p + 16) * strideBytes],
                            xmm8);
                }
                vextractf128(xmm8, ymm15, 1);
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p + 17) * strideBytes],
                             xmm8);
                } else {
                    vmovupd(ptr[storeBase + (2 * p + 17) * strideBytes],
                            xmm8);
                }
            }
        }
    }

    /* DFT-16 sub-leaf of the radix-32 leaf: input slot =
     * 2*bitrev4(pos) + par, result left in ymm0..7 (pair layout). */
    void emitSubLeaf16Sk(const Xbyak::Reg64 &loadBase, int strideBytes,
                         bool hasTw, int par) {
        using namespace Xbyak;
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};

        for (int p = 0; p < 8; ++p) {
            const int sa = 2 * (int)bitrev((uint32_t)(2 * p), 4) + par;
            const int sb = 2 * (int)bitrev((uint32_t)(2 * p + 1), 4) + par;
            if (gatherTab_ != nullptr) {
                const Reg64 &tab = *gatherTab_;
                mov(eax, dword[tab + sa * 4]);
                vmovupd(xmm8, ptr[rdi + rax]);
                mov(eax, dword[tab + sb * 4]);
                vmovupd(xmm9, ptr[rdi + rax]);
            } else {
                vmovupd(xmm8, ptr[loadBase + sa * strideBytes]);
                vmovupd(xmm9, ptr[loadBase + sb * strideBytes]);
            }
            if (hasTw && sa != 0) {
                mulTwXmm(xmm10, xmm8, sa);
            }
            if (hasTw && sb != 0) {
                mulTwXmm(xmm12, xmm9, sb);
            }
            const Xmm &a2 = (hasTw && sa != 0) ? xmm10 : xmm8;
            const Xmm &b2 = (hasTw && sb != 0) ? xmm12 : xmm9;
            /* xmm14 reserved for the sign mask (ymm14) */
            vaddpd(xmm13, a2, b2);
            vsubpd(xmm15, a2, b2);
            vinsertf128(P[p], P[p], xmm13, 0);
            vinsertf128(P[p], P[p], xmm15, 1);
        }

        /* stage 2 (m=4): top HIGH lane x (-i) blended into the scratch */
        for (int blk = 0; blk < 4; ++blk) {
            const Ymm &bot = P[2 * blk];
            const Ymm &top = P[2 * blk + 1];
            vshufpd(ymm8, top, top, 0x5);
            vxorpd(ymm8, ymm8, ymm14);
            vblendpd(ymm8, top, ymm8, 0x0C);
            vsubpd(top, bot, ymm8);
            vaddpd(bot, bot, ymm8);
        }

        /* stages 3..4: pair chunks (W^{2q}, W^{2q+1}) from the rip pool */
        for (int st = 3; st <= 4; ++st) {
            const int halfPairs = 1 << (st - 2);
            const int pairsPerBlock = 1 << (st - 1);
            for (int blk = 0; blk < 16 / (1 << st); ++blk) {
                const int bp = blk * pairsPerBlock;
                for (int q = 0; q < halfPairs; ++q) {
                    const Ymm &bot = P[bp + q];
                    const Ymm &top = P[bp + halfPairs + q];
                    if (fma_) {
                        vmovddup(ymm11, top);
                        vshufpd(ymm8, top, top, 0xF);
                        vmulpd(ymm8, ymm8, ptr[rip + lwSwapLbl_[st][q]]);
                        vfmaddsub231pd(ymm8, ymm11, ptr[rip + lwLbl_[st][q]]);
                    } else {
                        vmovapd(ymm10, ptr[rip + lwLbl_[st][q]]);
                        vshufpd(ymm12, ymm10, ymm10, 0x5);
                        vxorpd(ymm12, ymm12, ymm14);
                        vmulpd(ymm8, top, ymm10);
                        vmulpd(ymm12, top, ymm12);
                        vhsubpd(ymm8, ymm8, ymm12);
                    }
                    vsubpd(top, bot, ymm8);
                    vaddpd(bot, bot, ymm8);
                }
            }
        }
    }

    /* dst (xmm) = d * TW[slot]; 32B slots at r10 + slot*32 =
     * (wr, wi, wi, +wr|FMA / -wr|no-FMA) — both halves are memory
     * operands of the complex multiply. */
    void mulTwXmm(const Xbyak::Xmm &dst, const Xbyak::Xmm &d, int slot) {
        using namespace Xbyak;
        if (fma_) {
            vmovddup(xmm11, d);             /* (dr, dr) */
            vshufpd(dst, d, d, 3);          /* (di, di) */
            vmulpd(dst, dst, ptr[r10 + slot * 32 + 16]);
            vfmaddsub231pd(dst, xmm11, ptr[r10 + slot * 32]);
        } else {
            vmulpd(dst, d, ptr[r10 + slot * 32]);
            vmulpd(xmm11, d, ptr[r10 + slot * 32 + 16]);
            vhsubpd(dst, dst, xmm11);
        }
    }

    /* rip-relative constant pool: sign mask + vector-2 broadcast
     * twiddles (+ swap twins) + single-k pair chunks (+ twins) + W_32
     * cross pairs (+ twins).  Shared by every leaf of the plan. */
    void emitPool() {
        using namespace Xbyak;
        align(32);
        L(sign2Lbl_);
        dq(0);
        dq(0x8000000000000000ull);
        dq(0);
        dq(0x8000000000000000ull);

        /* W_{2^s}^e for s = 3..4, e = 1..2^{s-1}-1 (e = 0 skipped = 1,
         * e = 2^{s-2} skipped = -i fold); s=4 entries double as the
         * radix-16 cross W_16^j. */
        for (int s = 3; s <= 4; ++s) {
            for (int e = 1; e < (1 << (s - 1)); ++e) {
                if (e == (1 << (s - 2))) {
                    continue;
                }
                const double ang =
                    -2.0 * kPi * (double)e / (double)(1 << s);
                const double wr = std::cos(ang);
                const double wi = std::sin(ang);
                align(32);
                L(twLbl_[s][e]);
                dq(bits_from_double(wr));
                dq(bits_from_double(wi));
                dq(bits_from_double(wr));
                dq(bits_from_double(wi));
                align(32);
                L(twSwapLbl_[s][e]);
                dq(bits_from_double(wi));
                dq(bits_from_double(wr));
                dq(bits_from_double(wi));
                dq(bits_from_double(wr));
            }
        }

        /* single-k pair chunks: chunk q = (W_{2^s}^{2q}, W_{2^s}^{2q+1}) */
        for (int s = 3; s <= 4; ++s) {
            const int halfPairs = 1 << (s - 2);
            for (int q = 0; q < halfPairs; ++q) {
                double c[2], sn[2];
                for (int l = 0; l < 2; ++l) {
                    const int e = 2 * q + l;
                    const double ang =
                        -2.0 * kPi * (double)e / (double)(1 << s);
                    c[l] = std::cos(ang);
                    sn[l] = std::sin(ang);
                }
                align(32);
                L(lwLbl_[s][q]);
                dq(bits_from_double(c[0]));
                dq(bits_from_double(sn[0]));
                dq(bits_from_double(c[1]));
                dq(bits_from_double(sn[1]));
                align(32);
                L(lwSwapLbl_[s][q]);
                dq(bits_from_double(sn[0]));
                dq(bits_from_double(c[0]));
                dq(bits_from_double(sn[1]));
                dq(bits_from_double(c[1]));
            }
        }

        /* W_32 cross pairs: chunk p = (W_32^{2p}, W_32^{2p+1}) */
        for (int p = 0; p < 8; ++p) {
            double c[2], sn[2];
            for (int l = 0; l < 2; ++l) {
                const int e = 2 * p + l;
                const double ang = -2.0 * kPi * (double)e / 32.0;
                c[l] = std::cos(ang);
                sn[l] = std::sin(ang);
            }
            align(32);
            L(w32Lbl_[p]);
            dq(bits_from_double(c[0]));
            dq(bits_from_double(sn[0]));
            dq(bits_from_double(c[1]));
            dq(bits_from_double(sn[1]));
            align(32);
            L(w32SwapLbl_[p]);
            dq(bits_from_double(sn[0]));
            dq(bits_from_double(c[0]));
            dq(bits_from_double(sn[1]));
            dq(bits_from_double(c[1]));
        }
    }

    static uint64_t bits_from_double(double d) {
        uint64_t u;
        __builtin_memcpy(&u, &d, sizeof(u));
        return u;
    }

    bool fma_;
    bool nt_ = false;         /* R2 F7: NT stores enabled (N >= 4096) */
    int prefetch_ = 0;        /* R2 F7: tile-head prefetcht0 (0 = off) */
    bool ntSweep_ = false;    /* current sweep's stores are NT-eligible */
    int spillOff_ = -256; /* r16/r32 leaf spill base offset from rbp */
    const Xbyak::Reg64 *gatherTab_ = nullptr; /* stage-0 table mode */
    Xbyak::Label sign2Lbl_;
    Xbyak::Label twLbl_[5][8];     /* [s][e]: W_{2^s}^e broadcast (v2) */
    Xbyak::Label twSwapLbl_[5][8];
    Xbyak::Label lwLbl_[5][4];     /* [s][q]: pair chunk (single-k) */
    Xbyak::Label lwSwapLbl_[5][4];
    Xbyak::Label w32Lbl_[8];
    Xbyak::Label w32SwapLbl_[8];
};

/* ========================================================================
 * topic jit-fp64-conv-fuse — SchedEmitterC: FUSED cyclic-convolution weld
 * (L-I, DELTA F1; draft {#API-FFT-011} / {#BEH-FFT-015}).
 *
 * Copy-then-edit of the promoted SchedEmitter above (which stays untouched
 * — the fp64 forward path is regression-only).  ONE welded kernel runs the
 * forward half and the conjugate-trick inverse half of
 * y = IFFT(FFT(x) .* H):
 *
 *   - half A (forward, finalMode 1): stage 0 gathers io -> scratch, middle
 *     stages sweep the scratch, and the FINAL stage multiplies X[k]*H[k]
 *     and conjugates AT THE STORE BOUNDARY — per store vector the natural
 *     H pair (32B, [r14 + rax + off]) plus its pre-swapped twin
 *     ([r15 + rax + off]) feeds the SAME vmulpd/vhsubpd or
 *     vmovddup/vfmaddsub231pd complex-multiply idiom the twiddle loads use
 *     (FMA3 gate per [[CON-FFT-002]]), then one vxorpd against the sign
 *     mask flips the imaginary lane: io = conj(X*H), natural order.
 *     r14/r15 park hMul/hSwap relative to the SCRATCH base; the final
 *     stage's rax walks the scratch k-cursor, so rax + pos*stride lands
 *     exactly on the natural-order H pair of the stored (k, k+1) chunk.
 *   - half B (inverse, finalMode 2): rdi is restored from the parked rdx
 *     (io), stage 0 gathers the natural conj(X*H) directly — no wrapper
 *     sweep between halves; the FINAL stage stores conj(.) * (1/N) via one
 *     vmulpd against an (invN, -invN) rip constant.
 *
 * The three-segment wrapper's pointwise / conj / conj*invN sweeps are
 * eliminated; the intermediate spectrum transits io exactly once (written
 * by the half-A final store, gather-read by the half-B stage-0) instead of
 * write + 2x read-modify-write + read.  Methodology = the ps topic
 * (poc/jit-ps-sched-fuse, same-kernel fused conv 3.4-5.5x evidence)
 * ported to the pd vector-2 leaf layout.
 *
 * Requires kStages >= 2 (the N==32 single-codelet decomposition is not
 * weldable; the conv build forces k>=2 splits at n <= 5).
 * ======================================================================== */
class SchedEmitterC final : public Xbyak::CodeGenerator {
public:
    struct Stage {
        int radix;          /* r_t */
        int bprev;          /* B_{t-1} in elements (t=0 -> 1) */
        int b;              /* B_t in elements */
        const double *tw;   /* TW_t base, nullptr for t=0 */
    };

    SchedEmitterC(int n, const uint32_t *gatherTab, const double *scratch,
                  const Stage *stages, int kStages, bool fma,
                  const int *groupLen, const double *hMul,
                  const double *hSwap, bool nt, int prefetch)
        : Xbyak::CodeGenerator(524288), fma_(fma),
          nt_(nt && ((1 << n) >= 4096)), prefetch_(prefetch), n_(n) {
        if (kStages < 2) {
            /* unreachable by construction: the conv build excludes k < 2
             * decompositions.  Fail the emission loudly. */
            throw std::bad_alloc();
        }
        int gl[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (groupLen != nullptr) {
            for (int i = 0; i < 8; ++i) {
                gl[i] = groupLen[i];
            }
        }
        bool needSpill = false;
        for (int t = 0; t < kStages; ++t) {
            if (stages[t].radix >= 16) {
                needSpill = true;
            }
        }
        /* spill frame keeps clear of the pushed rbx/r12/r13/r14/r15 slots:
         * r15 sits at [rbp-40]; the 8x32B spill frame starts at rbp-304
         * (frame = 272B from rbp-312 when needSpill). */
        spillOff_ = -304;
        push(rbp);
        mov(rbp, rsp);
        push(rbx);
        push(r12);
        push(r13); /* r13 = stage-0-group gather-tab cursor */
        push(r14); /* r14 = hMul - scratch (conv half A anchor) */
        push(r15); /* r15 = hSwap - scratch (conv half A anchor) */
        if (needSpill) {
            sub(rsp, 272);
        }

        const uint64_t hm = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(hMul) -
            reinterpret_cast<uintptr_t>(scratch));
        const uint64_t hs = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(hSwap) -
            reinterpret_cast<uintptr_t>(scratch));

        const int N = 1 << n;
        /* ---- half A: forward; final stores = conj(X*H) ---- */
        mov(r14, hm);
        mov(r15, hs);
        emitHalf(N, gatherTab, scratch, stages, kStages, gl, /*mode=*/1);

        /* ---- half B: inverse (conjugate trick).  rdi currently parks the
         * scratch base; the ORIGINAL io base lives in rdx — restore it so
         * the stage-0 gather reads the natural conj(X*H) written by half
         * A's final stage. ---- */
        mov(rdi, rdx);
        emitHalf(N, gatherTab, scratch, stages, kStages, gl, /*mode=*/2);

        vzeroupper();
        lea(rsp, ptr[rbp - 40]); /* at the pushed r15 slot */
        pop(r15);
        pop(r14);
        pop(r13);
        pop(r12);
        pop(rbx);
        pop(rbp);
        ret();
        emitPool();
    }

private:
    static int lf_log2(int v) {
        int s = 0;
        while ((1 << s) < v) {
            ++s;
        }
        return s;
    }

    /* ---- one transform half (stage 0, middle sweeps, fused final).
     * finalMode_ stays 0 through the gather / middle sweeps; only the
     * final stage runs with the half's fused mode. ---- */
    void emitHalf(int N, const uint32_t *gatherTab, const double *scratch,
                  const Stage *stages, int kStages, const int *groupLen,
                  int mode) {
        finalMode_ = 0;
        int t = 0;
        if (groupLen[0] >= 2) {
            emitStage0Group(N, gatherTab, scratch, stages, groupLen[0]);
            t = groupLen[0];
        } else {
            emitStage0Plain(N, gatherTab, scratch, stages[0]);
            t = 1;
        }
        while (t < kStages - 1) {
            const int d = groupLen[t] >= 2 ? groupLen[t] : 1;
            if (d >= 2) {
                emitTileGroup(stages, t, d, N);
            } else {
                emitSweepFull(stages[t], N, /*allowNt=*/true);
            }
            t += d;
        }
        /* final stage: reads the scratch, writes the natural array with
         * the fused boundary op (mode).  Stores are NEVER non-temporal
         * (the caller / the next half reads the result immediately). */
        emitFinalStage(stages[kStages - 1], N, mode);
    }

    /* ---- stage 0, unblocked (R0/R1 shape): digitrev gather fused into
     * the first codelet sweep, out-of-place into the scratch ---- */
    void emitStage0Plain(int N, const uint32_t *gatherTab,
                         const double *scratch, const Stage &st) {
        using namespace Xbyak;
        mov(r8, reinterpret_cast<uint64_t>(gatherTab));
        mov(rsi, reinterpret_cast<uint64_t>(scratch));
        lea(rdx, ptr[rsi + N * 16]); /* scratch end */
        ntSweep_ = nt_; /* scratch stores: re-read only by a later full
                           sweep -> NT candidate (R2 F7) */
        Label blk;
        L(blk);
        if (st.radix <= 16) {
            /* vector-2 over block pairs (2b, 2b+1) */
            gatherTab_ = &r8;
            emitLeafV2(st.radix, rsi, 16, rsi, false);
            gatherTab_ = nullptr;
            add(r8, st.radix * 8);   /* 2 table rows (uint32) */
            add(rsi, st.radix * 32); /* 2 blocks */
        } else {
            gatherTab_ = &r8;
            emitLeaf32(r8, 16, false, rsi);
            gatherTab_ = nullptr;
            add(r8, st.radix * 4);
            add(rsi, st.radix * 16);
        }
        cmp(rsi, rdx);
        jb(blk);
        ntSweep_ = false;
        if (nt_) {
            sfence(); /* NT scratch writes precede later loads (F7) */
        }
        /* stages 1..k-2 run in-place on the scratch; park the natural
         * base in rdx (stage-0 loop is done) */
        mov(rdx, rdi);
        mov(rdi, reinterpret_cast<uint64_t>(scratch));
    }

    /* ---- R2 F6: stage-0 tile group.  Per tile: gather exactly the
     * tile's digitrev rows (tab cursor r13 stays sequential across
     * tiles), write the tile in the scratch, then run stages 1..d-1
     * confined to the tile before advancing (intermediate results never
     * leave L1). ---- */
    void emitStage0Group(int N, const uint32_t *gatherTab,
                         const double *scratch, const Stage *stages,
                         int d) {
        using namespace Xbyak;
        const Stage &st = stages[0];
        const int T = stages[d - 1].b; /* tile elements */
        bool ntUsed = false;
        mov(r13, reinterpret_cast<uint64_t>(gatherTab)); /* tab cursor */
        mov(r12, reinterpret_cast<uint64_t>(scratch));   /* tile cursor */
        lea(rcx, ptr[r12 + N * 16]); /* array end (tile-loop bound) */
        Label tile;
        L(tile);
        emitPrefetchHead(r12, T * 16);
        mov(rsi, r12);
        lea(rdx, ptr[r12 + T * 16]); /* this tile's scratch end */
        Label blk;
        L(blk);
        if (st.radix <= 16) {
            gatherTab_ = &r13;
            emitLeafV2(st.radix, rsi, 16, rsi, false);
            gatherTab_ = nullptr;
            add(r13, st.radix * 8);  /* 2 table rows (uint32) */
            add(rsi, st.radix * 32); /* 2 blocks */
        } else {
            gatherTab_ = &r13;
            emitLeaf32(r13, 16, false, rsi);
            gatherTab_ = nullptr;
            add(r13, st.radix * 4);
            add(rsi, st.radix * 16);
        }
        cmp(rsi, rdx);
        jb(blk);
        for (int s = 1; s < d; ++s) {
            const bool last = (s == d - 1);
            emitSweepInTile(stages[s], T * 16, last);
            ntUsed |= (nt_ && last);
        }
        add(r12, T * 16);
        cmp(r12, rcx);
        jb(tile);
        if (ntUsed) {
            sfence(); /* last-in-group NT writes precede later loads */
        }
        /* park the natural base; the scratch becomes the working array */
        mov(rdx, rdi);
        mov(rdi, reinterpret_cast<uint64_t>(scratch));
    }

    /* ---- one stage's sweep confined to the current tile ([r12,
     * r12+tileBytes)); in-tile stores stay WB (they are re-read from L1
     * inside the tile) EXCEPT the last stage of the group (its output is
     * consumed only by a later full sweep -> NT candidate). ---- */
    void emitSweepInTile(const Stage &st, int tileBytes, bool lastOfGroup) {
        using namespace Xbyak;
        ntSweep_ = nt_ && lastOfGroup;
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, r12);
        lea(r8, ptr[r12 + tileBytes]);
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 16]); /* k-loop end */
        mov(rax, rsi);
        mov(r10, r9);
        Label kLoop;
        L(kLoop);
        if (st.radix <= 16) {
            /* vector-2 over k pairs (k, k+1) */
            emitLeafV2(st.radix, rax, st.bprev * 16, rax, true);
            add(rax, 32);
            add(r10, st.radix * 64);
        } else {
            emitLeaf32(rax, st.bprev * 16, true, rax);
            add(rax, 16);
            add(r10, 32 * 32);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 16);
        cmp(rsi, r8);
        jb(blk);
        ntSweep_ = false;
    }

    /* ---- one stage's unblocked full-array sweep (R0/R1 shape); NT
     * stores + trailing sfence when enabled (output re-read only by a
     * later full sweep). ---- */
    void emitSweepFull(const Stage &st, int N, bool allowNt) {
        using namespace Xbyak;
        ntSweep_ = nt_ && allowNt;
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, rdi);
        lea(r8, ptr[rdi + N * 16]); /* end (r8 table dead) */
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 16]); /* k-loop end */
        mov(rax, rsi);
        mov(r10, r9);
        Label kLoop;
        L(kLoop);
        if (st.radix <= 16) {
            /* vector-2 over k pairs (k, k+1) */
            emitLeafV2(st.radix, rax, st.bprev * 16, rax, true);
            add(rax, 32);
            add(r10, st.radix * 64);
        } else {
            emitLeaf32(rax, st.bprev * 16, true, rax);
            add(rax, 16);
            add(r10, 32 * 32);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 16);
        cmp(rsi, r8);
        jb(blk);
        ntSweep_ = false;
        if (nt_ && allowNt) {
            sfence();
        }
    }

    /* ---- R2 F6: middle tile group (stages t..t+d-1, t >= 1) ---- */
    void emitTileGroup(const Stage *stages, int t, int d, int N) {
        using namespace Xbyak;
        const int T = stages[t + d - 1].b; /* tile elements */
        bool ntUsed = false;
        mov(r12, rdi); /* tile cursor = scratch base */
        lea(rcx, ptr[rdi + N * 16]);
        Label tile;
        L(tile);
        emitPrefetchHead(r12, T * 16);
        for (int s = t; s < t + d; ++s) {
            const bool last = (s == t + d - 1);
            emitSweepInTile(stages[s], T * 16, last);
            ntUsed |= (nt_ && last);
        }
        add(r12, T * 16);
        cmp(r12, rcx);
        jb(tile);
        if (ntUsed) {
            sfence(); /* last-in-group NT writes precede later loads */
        }
    }

    /* ---- R2 F7: prefetcht0 the head (8 cache lines) of the NEXT tile
     * at every tile-loop boundary (knob-gated; prefetch of an address
     * past the array end is architecturally harmless). ---- */
    void emitPrefetchHead(const Xbyak::Reg64 &base, int tileBytes) {
        if (prefetch_ <= 0) {
            return;
        }
        for (int i = 0; i < 8; ++i) {
            prefetcht0(ptr[base + tileBytes + i * 64]);
        }
    }

    /* ---- final stage: reads the scratch, writes the natural array with
     * the fused boundary op (mode 1 = conj(X*H), 2 = conj*invN).
     * rax = scratch k-cursor anchors the H tables via r14/r15. ---- */
    void emitFinalStage(const Stage &st, int N, int mode) {
        using namespace Xbyak;
        finalMode_ = mode;
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, rdi);
        lea(r8, ptr[rdi + N * 16]); /* end (r8 table dead) */
        mov(r12, rdx);             /* natural block cursor */
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 16]); /* k-loop end */
        mov(rax, rsi);
        mov(r10, r9);
        Label kLoop;
        L(kLoop);
        if (st.radix <= 16) {
            /* vector-2 over k pairs (k, k+1) */
            mov(rbx, r12);
            emitLeafV2(st.radix, rax, st.bprev * 16, rbx, true);
            add(rax, 32);
            add(r10, st.radix * 64);
            add(r12, 32);
        } else {
            mov(rbx, r12);
            emitLeaf32(rax, st.bprev * 16, true, rbx);
            add(rax, 16);
            add(r10, 32 * 32);
            add(r12, 16);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 16);
        add(r12, st.b * 16);
        cmp(rsi, r8);
        jb(blk);
        finalMode_ = 0;
    }

    /* ---- topic jit-fp64-conv-fuse: fused final-stage boundary op, in
     * place on a (k, k+1) natural pair chunk.  finalMode_ 1: v <-
     * conj(v * H_pair) with the natural H pair at [r14 + rax + off] and
     * its pre-swapped twin at [r15 + rax + off] (FMA3 twin = (hi, +hr),
     * non-FMA = (hi, -hr) — same idiom/sign discipline as the twiddle
     * twins).  finalMode_ 2: v <- conj(v) * (1/N) via one vmulpd. ---- */
    void fuseFinalPairInPlace(const Xbyak::Ymm &v, int off) {
        using namespace Xbyak;
        if (finalMode_ == 1) {
            if (fma_) {
                vmovddup(ymm12, v);            /* (ar, ar) per lane */
                vshufpd(v, v, v, 0xF);         /* (ai, ai) per lane */
                vmulpd(v, v, ptr[r15 + rax + off]);
                vfmaddsub231pd(v, ymm12, ptr[r14 + rax + off]);
            } else {
                vmulpd(ymm12, v, ptr[r14 + rax + off]);
                vmulpd(ymm13, v, ptr[r15 + rax + off]);
                vhsubpd(v, ymm12, ymm13);
            }
            vxorpd(v, v, ptr[rip + sign2Lbl_]);
        } else if (finalMode_ == 2) {
            vmulpd(v, v, ptr[rip + invNLbl_]);
        }
    }

    /* ---- vector-2 leaf dispatch (radix 2/4/8/16) ----
     * Loads come from loadBase + slot*strideBytes (twiddled stages) or
     * through the gatherTab_ cursor (stage 0); stores go to storeBase +
     * pos*strideBytes (twiddled) or storeBase + pos*16 / (r+pos)*16 for
     * the two gathered blocks (stage 0, contiguous scratch).  Twiddle
     * rows (hasTw) come from r10: 64B slots per (k-pair, j). */
    void emitLeafV2(int r, const Xbyak::Reg64 &loadBase, int strideBytes,
                    const Xbyak::Reg64 &storeBase, bool hasTw) {
        using namespace Xbyak;
        if (r >= 4 || (!fma_ && hasTw)) {
            vmovapd(ymm14, ptr[rip + sign2Lbl_]);
        }
        if (r == 16) {
            emitV2Sub(8, loadBase, strideBytes, hasTw, 0);
            for (int j = 0; j < 8; ++j) {
                vmovupd(ptr[rbp + spillOff_ + j * 32], Ymm(j));
            }
            emitV2Sub(8, loadBase, strideBytes, hasTw, 1);
            cross16V2(storeBase, strideBytes);
        } else {
            emitV2Sub(r, loadBase, strideBytes, hasTw, -1);
            storeV2(r, storeBase, strideBytes);
        }
    }

    /* r-point DFT in per-position vector-2 registers P[0..r-1] (r <= 8).
     * par < 0: whole leaf (slot(pos) = bitrev(pos, log2 r));
     * par >= 0: sub-leaf of the radix-16 leaf (slot(pos) =
     * 2*bitrev(pos, 3) + par; gather rows stride 16). */
    void emitV2Sub(int r, const Xbyak::Reg64 &loadBase, int strideBytes,
                   bool hasTw, int par) {
        using namespace Xbyak;
        const int nlf = lf_log2(r);
        const int rr = (par >= 0) ? 16 : r; /* gather row stride (r1) */
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};

        /* ---- load (+twiddle) + m=2 fuse ---- */
        for (int u = 0; u < r / 2; ++u) {
            int sa, sb;
            if (par >= 0) {
                sa = 2 * (int)bitrev((uint32_t)(2 * u), nlf) + par;
                sb = 2 * (int)bitrev((uint32_t)(2 * u + 1), nlf) + par;
            } else {
                sa = (int)bitrev((uint32_t)(2 * u), nlf);
                sb = (int)bitrev((uint32_t)(2 * u + 1), nlf);
            }
            if (gatherTab_ != nullptr) {
                const Reg64 &tab = *gatherTab_;
                mov(eax, dword[tab + sa * 4]);
                vmovupd(xmm8, ptr[rdi + rax]);
                mov(eax, dword[tab + (rr + sa) * 4]);
                vmovupd(xmm9, ptr[rdi + rax]);
                mov(eax, dword[tab + sb * 4]);
                vmovupd(xmm10, ptr[rdi + rax]);
                mov(eax, dword[tab + (rr + sb) * 4]);
                vmovupd(xmm11, ptr[rdi + rax]);
                /* xmm14 holds the sign mask (ymm14) — the second
                 * block's fuse reuses the now-dead load regs */
                vaddpd(xmm12, xmm8, xmm10);
                vsubpd(xmm13, xmm8, xmm10);
                vaddpd(xmm8, xmm9, xmm11);
                vsubpd(xmm10, xmm9, xmm11);
                vinsertf128(P[2 * u], P[2 * u], xmm12, 0);
                vinsertf128(P[2 * u], P[2 * u], xmm8, 1);
                vinsertf128(P[2 * u + 1], P[2 * u + 1], xmm13, 0);
                vinsertf128(P[2 * u + 1], P[2 * u + 1], xmm10, 1);
            } else {
                loadTwV2(ymm10, ptr[loadBase + sa * strideBytes], sa, hasTw);
                loadTwV2(ymm11, ptr[loadBase + sb * strideBytes], sb, hasTw);
                vaddpd(P[2 * u], ymm10, ymm11);
                vsubpd(P[2 * u + 1], ymm10, ymm11);
            }
        }

        /* ---- stages 2..nlf (butterflies pair positions (i, i+h)) ---- */
        for (int st = 2; st <= nlf; ++st) {
            const int m = 1 << st;
            const int h = m >> 1;
            for (int blk = 0; blk < r / m; ++blk) {
                for (int i = 0; i < h; ++i) {
                    const Ymm &bot = P[blk * m + i];
                    const Ymm &top = P[blk * m + h + i];
                    if (i == 0) {
                        /* T = top in place: diff via scratch */
                        vsubpd(ymm12, bot, top);
                        vaddpd(bot, bot, top);
                        vmovapd(top, ymm12);
                    } else if (i == (m >> 2)) {
                        /* W_m^{m/4} = -i: swap + sign flip */
                        vshufpd(ymm12, top, top, 0x5);
                        vxorpd(ymm12, ymm12, ymm14);
                        vsubpd(top, bot, ymm12);
                        vaddpd(bot, bot, ymm12);
                    } else {
                        mulTwPoolV2(top, st, i);
                        vsubpd(top, bot, ymm12);
                        vaddpd(bot, bot, ymm12);
                    }
                }
            }
        }
    }

    /* load the (k, k+1) pair at addr; twiddle by TW row slot (64B slots
     * at r10 + slot*64) when slot != 0; result in dst (ymm10/ymm11). */
    void loadTwV2(const Xbyak::Ymm &dst, const Xbyak::Operand &addr,
                  int slot, bool hasTw) {
        using namespace Xbyak;
        if (slot == 0 || !hasTw) {
            vmovupd(dst, addr);
            return;
        }
        vmovupd(ymm8, addr);
        if (fma_) {
            vmovddup(ymm9, ymm8);            /* (ar, ar) per lane */
            vshufpd(dst, ymm8, ymm8, 0xF);   /* (ai, ai) per lane */
            vmulpd(dst, dst, ptr[r10 + slot * 64 + 32]);
            vfmaddsub231pd(dst, ymm9, ptr[r10 + slot * 64]);
        } else {
            vmovupd(ymm12, ptr[r10 + slot * 64]); /* W pair */
            vshufpd(ymm13, ymm12, ymm12, 0x5);
            vxorpd(ymm13, ymm13, ymm14);          /* (wi, -wr) */
            vmulpd(dst, ymm8, ymm12);
            vmulpd(ymm13, ymm8, ymm13);
            vhsubpd(dst, dst, ymm13);
        }
    }

    /* T (ymm12) = top * W_{2^st}^e from the rip pool (broadcast const +
     * pre-swapped twin as memory operands; non-FMA derives the conjugate
     * in registers). */
    void mulTwPoolV2(const Xbyak::Ymm &top, int st, int e) {
        using namespace Xbyak;
        if (fma_) {
            vmovddup(ymm13, top);
            vshufpd(ymm12, top, top, 0xF);
            vmulpd(ymm12, ymm12, ptr[rip + twSwapLbl_[st][e]]);
            vfmaddsub231pd(ymm12, ymm13, ptr[rip + twLbl_[st][e]]);
        } else {
            vmovapd(ymm13, ptr[rip + twLbl_[st][e]]);
            vshufpd(ymm15, ymm13, ymm13, 0x5);
            vxorpd(ymm15, ymm15, ymm14);
            vmulpd(ymm12, top, ymm13);
            vmulpd(ymm15, top, ymm15);
            vhsubpd(ymm12, ymm12, ymm15);
        }
    }

    /* radix-16 cross: y[j] = e[j] + W_16^j o[j], y[j+8] = e[j] - W_16^j o[j]
     * (j = 0..7).  e[] reloaded from the spill frame, o[] in P. */
    void cross16V2(const Xbyak::Reg64 &storeBase, int strideBytes) {
        using namespace Xbyak;
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};
        for (int j = 0; j < 8; ++j) {
            vmovupd(ymm8, ptr[rbp + spillOff_ + j * 32]); /* e[j] */
            if (j == 0) {
                vaddpd(ymm10, ymm8, P[0]);
                vsubpd(ymm11, ymm8, P[0]);
            } else if (j == 4) {
                /* W_16^4 = -i */
                vshufpd(ymm12, P[4], P[4], 0x5);
                vxorpd(ymm12, ymm12, ymm14);
                vaddpd(ymm10, ymm8, ymm12);
                vsubpd(ymm11, ymm8, ymm12);
            } else {
                mulTwPoolV2(P[j], 4, j); /* W_16^j = W_{2^4}^j */
                vaddpd(ymm10, ymm8, ymm12);
                vsubpd(ymm11, ymm8, ymm12);
            }
            storeV2One(ymm10, j, storeBase, strideBytes, 16);
            storeV2One(ymm11, j + 8, storeBase, strideBytes, 16);
        }
    }

    void storeV2(int r, const Xbyak::Reg64 &storeBase, int strideBytes) {
        using namespace Xbyak;
        for (int p = 0; p < r; ++p) {
            storeV2One(Ymm(p), p, storeBase, strideBytes, r);
        }
    }

    /* store position pos: whole 32B ym at storeBase + pos*stride (normal
     * / final stages), or per-block 16B extracts into the two gathered
     * blocks (stage 0: storeBase + pos*16 and storeBase + (r+pos)*16).
     * FUSED final stages (finalMode_ != 0): the boundary op runs on the
     * ym first (X*H + conj, or conj*invN — see fuseFinalPairInPlace).
     * R2 F7: when the sweep is NT-eligible the stores are vmovntpd
     * (non-temporal; the sfence lives at the sweep/tile boundary). */
    void storeV2One(const Xbyak::Ymm &v, int pos,
                    const Xbyak::Reg64 &storeBase, int strideBytes, int r) {
        using namespace Xbyak;
        if (gatherTab_ != nullptr) {
            vextractf128(xmm12, v, 0);
            if (ntSweep_) {
                vmovntpd(ptr[storeBase + pos * 16], xmm12);
            } else {
                vmovupd(ptr[storeBase + pos * 16], xmm12);
            }
            vextractf128(xmm12, v, 1);
            if (ntSweep_) {
                vmovntpd(ptr[storeBase + (r + pos) * 16], xmm12);
            } else {
                vmovupd(ptr[storeBase + (r + pos) * 16], xmm12);
            }
            return;
        }
        const int off = pos * strideBytes;
        if (finalMode_ != 0) {
            fuseFinalPairInPlace(v, off);
            vmovupd(ptr[storeBase + off], v);
            return;
        }
        if (ntSweep_) {
            vmovntpd(ptr[storeBase + off], v);
        } else {
            vmovupd(ptr[storeBase + off], v);
        }
    }

    /* ---- single-k radix-32 leaf (pair-ymm lanes = adjacent DIT
     * positions of one k; vector-2 would need 32 live yms) ---- */
    void emitLeaf32(const Xbyak::Reg64 &loadBase, int strideBytes,
                    bool hasTw, const Xbyak::Reg64 &storeBase) {
        using namespace Xbyak;
        vmovapd(ymm14, ptr[rip + sign2Lbl_]);
        emitSubLeaf16Sk(loadBase, strideBytes, hasTw, 0);
        for (int p = 0; p < 8; ++p) {
            vmovupd(ptr[rbp + spillOff_ + p * 32], Ymm(p));
        }
        emitSubLeaf16Sk(loadBase, strideBytes, hasTw, 1);
        /* cross: y[k] = e[k] + W_32^k o[k], pair chunks (W^{2p}, W^{2p+1})
         * from the rip pool (mixing foldable and general lanes — the
         * chunked multiply stays; e-pair reloaded from the spill frame) */
        for (int p = 0; p < 8; ++p) {
            vmovupd(ymm9, ptr[rbp + spillOff_ + p * 32]); /* A pair */
            if (fma_) {
                vmovddup(ymm11, Ymm(p));
                vshufpd(ymm8, Ymm(p), Ymm(p), 0xF);
                vmulpd(ymm8, ymm8, ptr[rip + w32SwapLbl_[p]]);
                vfmaddsub231pd(ymm8, ymm11, ptr[rip + w32Lbl_[p]]);
            } else {
                vmovupd(ymm10, ptr[rip + w32Lbl_[p]]);
                vshufpd(ymm12, ymm10, ymm10, 0x5);
                vxorpd(ymm12, ymm12, ymm14);
                vmulpd(ymm8, Ymm(p), ymm10);
                vmulpd(ymm12, Ymm(p), ymm12);
                vhsubpd(ymm8, ymm8, ymm12);
            }
            vaddpd(ymm13, ymm9, ymm8);  /* y[k]    -> slots (2p, 2p+1) */
            vsubpd(ymm15, ymm9, ymm8);  /* y[k+16] -> (2p+16, 2p+17);
                                           ymm14 = sgn must stay live */
            if (strideBytes == 16) {
                /* stage-0 gather (the N==32 single-codelet transform is
                 * excluded from the weld by construction) */
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p) * 16], ymm13);
                    vmovntpd(ptr[storeBase + (2 * p + 16) * 16], ymm15);
                } else {
                    vmovupd(ptr[storeBase + (2 * p) * 16], ymm13);
                    vmovupd(ptr[storeBase + (2 * p + 16) * 16], ymm15);
                }
            } else {
                if (finalMode_ != 0) {
                    /* fused final: boundary op on the pair chunks BEFORE
                     * the extracts (H pairs are natural-contiguous at
                     * rax + (2p / 2p+16) * stride) */
                    fuseFinalPairInPlace(ymm13, (2 * p) * strideBytes);
                    fuseFinalPairInPlace(ymm15, (2 * p + 16) * strideBytes);
                }
                vextractf128(xmm8, ymm13, 0);
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p) * strideBytes], xmm8);
                } else {
                    vmovupd(ptr[storeBase + (2 * p) * strideBytes], xmm8);
                }
                vextractf128(xmm8, ymm13, 1);
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p + 1) * strideBytes],
                             xmm8);
                } else {
                    vmovupd(ptr[storeBase + (2 * p + 1) * strideBytes],
                            xmm8);
                }
                vextractf128(xmm8, ymm15, 0);
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p + 16) * strideBytes],
                             xmm8);
                } else {
                    vmovupd(ptr[storeBase + (2 * p + 16) * strideBytes],
                            xmm8);
                }
                vextractf128(xmm8, ymm15, 1);
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p + 17) * strideBytes],
                             xmm8);
                } else {
                    vmovupd(ptr[storeBase + (2 * p + 17) * strideBytes],
                            xmm8);
                }
            }
        }
    }

    /* DFT-16 sub-leaf of the radix-32 leaf: input slot =
     * 2*bitrev4(pos) + par, result left in ymm0..7 (pair layout). */
    void emitSubLeaf16Sk(const Xbyak::Reg64 &loadBase, int strideBytes,
                         bool hasTw, int par) {
        using namespace Xbyak;
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};

        for (int p = 0; p < 8; ++p) {
            const int sa = 2 * (int)bitrev((uint32_t)(2 * p), 4) + par;
            const int sb = 2 * (int)bitrev((uint32_t)(2 * p + 1), 4) + par;
            if (gatherTab_ != nullptr) {
                const Reg64 &tab = *gatherTab_;
                mov(eax, dword[tab + sa * 4]);
                vmovupd(xmm8, ptr[rdi + rax]);
                mov(eax, dword[tab + sb * 4]);
                vmovupd(xmm9, ptr[rdi + rax]);
            } else {
                vmovupd(xmm8, ptr[loadBase + sa * strideBytes]);
                vmovupd(xmm9, ptr[loadBase + sb * strideBytes]);
            }
            if (hasTw && sa != 0) {
                mulTwXmm(xmm10, xmm8, sa);
            }
            if (hasTw && sb != 0) {
                mulTwXmm(xmm12, xmm9, sb);
            }
            const Xmm &a2 = (hasTw && sa != 0) ? xmm10 : xmm8;
            const Xmm &b2 = (hasTw && sb != 0) ? xmm12 : xmm9;
            /* xmm14 reserved for the sign mask (ymm14) */
            vaddpd(xmm13, a2, b2);
            vsubpd(xmm15, a2, b2);
            vinsertf128(P[p], P[p], xmm13, 0);
            vinsertf128(P[p], P[p], xmm15, 1);
        }

        /* stage 2 (m=4): top HIGH lane x (-i) blended into the scratch */
        for (int blk = 0; blk < 4; ++blk) {
            const Ymm &bot = P[2 * blk];
            const Ymm &top = P[2 * blk + 1];
            vshufpd(ymm8, top, top, 0x5);
            vxorpd(ymm8, ymm8, ymm14);
            vblendpd(ymm8, top, ymm8, 0x0C);
            vsubpd(top, bot, ymm8);
            vaddpd(bot, bot, ymm8);
        }

        /* stages 3..4: pair chunks (W^{2q}, W^{2q+1}) from the rip pool */
        for (int st = 3; st <= 4; ++st) {
            const int halfPairs = 1 << (st - 2);
            const int pairsPerBlock = 1 << (st - 1);
            for (int blk = 0; blk < 16 / (1 << st); ++blk) {
                const int bp = blk * pairsPerBlock;
                for (int q = 0; q < halfPairs; ++q) {
                    const Ymm &bot = P[bp + q];
                    const Ymm &top = P[bp + halfPairs + q];
                    if (fma_) {
                        vmovddup(ymm11, top);
                        vshufpd(ymm8, top, top, 0xF);
                        vmulpd(ymm8, ymm8, ptr[rip + lwSwapLbl_[st][q]]);
                        vfmaddsub231pd(ymm8, ymm11, ptr[rip + lwLbl_[st][q]]);
                    } else {
                        vmovapd(ymm10, ptr[rip + lwLbl_[st][q]]);
                        vshufpd(ymm12, ymm10, ymm10, 0x5);
                        vxorpd(ymm12, ymm12, ymm14);
                        vmulpd(ymm8, top, ymm10);
                        vmulpd(ymm12, top, ymm12);
                        vhsubpd(ymm8, ymm8, ymm12);
                    }
                    vsubpd(top, bot, ymm8);
                    vaddpd(bot, bot, ymm8);
                }
            }
        }
    }

    /* dst (xmm) = d * TW[slot]; 32B slots at r10 + slot*32 =
     * (wr, wi, wi, +wr|FMA / -wr|no-FMA) — both halves are memory
     * operands of the complex multiply. */
    void mulTwXmm(const Xbyak::Xmm &dst, const Xbyak::Xmm &d, int slot) {
        using namespace Xbyak;
        if (fma_) {
            vmovddup(xmm11, d);             /* (dr, dr) */
            vshufpd(dst, d, d, 3);          /* (di, di) */
            vmulpd(dst, dst, ptr[r10 + slot * 32 + 16]);
            vfmaddsub231pd(dst, xmm11, ptr[r10 + slot * 32]);
        } else {
            vmulpd(dst, d, ptr[r10 + slot * 32]);
            vmulpd(xmm11, d, ptr[r10 + slot * 32 + 16]);
            vhsubpd(dst, dst, xmm11);
        }
    }

    /* rip-relative constant pool: sign mask + vector-2 broadcast
     * twiddles (+ swap twins) + single-k pair chunks (+ twins) + W_32
     * cross pairs (+ twins) + the weld's (invN, -invN) constant.
     * Shared by every leaf of the plan. */
    void emitPool() {
        using namespace Xbyak;
        align(32);
        L(sign2Lbl_);
        dq(0);
        dq(0x8000000000000000ull);
        dq(0);
        dq(0x8000000000000000ull);

        /* topic jit-fp64-conv-fuse: conj(.) * (1/N) in one vmulpd */
        const double inv = 1.0 / (double)(1 << n_);
        align(32);
        L(invNLbl_);
        dq(bits_from_double(inv));
        dq(bits_from_double(-inv));
        dq(bits_from_double(inv));
        dq(bits_from_double(-inv));

        /* W_{2^s}^e for s = 3..4, e = 1..2^{s-1}-1 (e = 0 skipped = 1,
         * e = 2^{s-2} skipped = -i fold); s=4 entries double as the
         * radix-16 cross W_16^j. */
        for (int s = 3; s <= 4; ++s) {
            for (int e = 1; e < (1 << (s - 1)); ++e) {
                if (e == (1 << (s - 2))) {
                    continue;
                }
                const double ang =
                    -2.0 * kPi * (double)e / (double)(1 << s);
                const double wr = std::cos(ang);
                const double wi = std::sin(ang);
                align(32);
                L(twLbl_[s][e]);
                dq(bits_from_double(wr));
                dq(bits_from_double(wi));
                dq(bits_from_double(wr));
                dq(bits_from_double(wi));
                align(32);
                L(twSwapLbl_[s][e]);
                dq(bits_from_double(wi));
                dq(bits_from_double(wr));
                dq(bits_from_double(wi));
                dq(bits_from_double(wr));
            }
        }

        /* single-k pair chunks: chunk q = (W_{2^s}^{2q}, W_{2^s}^{2q+1}) */
        for (int s = 3; s <= 4; ++s) {
            const int halfPairs = 1 << (s - 2);
            for (int q = 0; q < halfPairs; ++q) {
                double c[2], sn[2];
                for (int l = 0; l < 2; ++l) {
                    const int e = 2 * q + l;
                    const double ang =
                        -2.0 * kPi * (double)e / (double)(1 << s);
                    c[l] = std::cos(ang);
                    sn[l] = std::sin(ang);
                }
                align(32);
                L(lwLbl_[s][q]);
                dq(bits_from_double(c[0]));
                dq(bits_from_double(sn[0]));
                dq(bits_from_double(c[1]));
                dq(bits_from_double(sn[1]));
                align(32);
                L(lwSwapLbl_[s][q]);
                dq(bits_from_double(sn[0]));
                dq(bits_from_double(c[0]));
                dq(bits_from_double(sn[1]));
                dq(bits_from_double(c[1]));
            }
        }

        /* W_32 cross pairs: chunk p = (W_32^{2p}, W_32^{2p+1}) */
        for (int p = 0; p < 8; ++p) {
            double c[2], sn[2];
            for (int l = 0; l < 2; ++l) {
                const int e = 2 * p + l;
                const double ang = -2.0 * kPi * (double)e / 32.0;
                c[l] = std::cos(ang);
                sn[l] = std::sin(ang);
            }
            align(32);
            L(w32Lbl_[p]);
            dq(bits_from_double(c[0]));
            dq(bits_from_double(sn[0]));
            dq(bits_from_double(c[1]));
            dq(bits_from_double(sn[1]));
            align(32);
            L(w32SwapLbl_[p]);
            dq(bits_from_double(sn[0]));
            dq(bits_from_double(c[0]));
            dq(bits_from_double(sn[1]));
            dq(bits_from_double(c[1]));
        }
    }

    static uint64_t bits_from_double(double d) {
        uint64_t u;
        __builtin_memcpy(&u, &d, sizeof(u));
        return u;
    }

    bool fma_;
    bool nt_ = false;         /* R2 F7: NT stores enabled (N >= 4096) */
    int prefetch_ = 0;        /* R2 F7: tile-head prefetcht0 (0 = off) */
    bool ntSweep_ = false;    /* current sweep's stores are NT-eligible */
    int spillOff_ = -304; /* r16/r32 leaf spill base offset from rbp */
    int n_ = 0;               /* log2(N) — invN pool constant */
    int finalMode_ = 0;       /* 0 plain; 1 = store conj(X*H) (half A);
                                 2 = store conj(.)*invN (half B) */
    const Xbyak::Reg64 *gatherTab_ = nullptr; /* stage-0 table mode */
    Xbyak::Label sign2Lbl_;
    Xbyak::Label invNLbl_;         /* (invN, -invN, invN, -invN) */
    Xbyak::Label twLbl_[5][8];     /* [s][e]: W_{2^s}^e broadcast (v2) */
    Xbyak::Label twSwapLbl_[5][8];
    Xbyak::Label lwLbl_[5][4];     /* [s][q]: pair chunk (single-k) */
    Xbyak::Label lwSwapLbl_[5][4];
    Xbyak::Label w32Lbl_[8];
    Xbyak::Label w32SwapLbl_[8];
};

/* ========================================================================
 * topic jit-fp64-linear-conv — SchedEmitterO: FUSED overlap-save LINEAR
 * convolution weld (DELTA F1; draft {#API-FFT-012} / {#BEH-FFT-017}).
 *
 * Copy-then-edit of the promoted SchedEmitterC above (which stays
 * untouched — the fp64 conv weld path is regression-only).  Each OLS
 * block = the conv weld body with the OLS boundary edits:
 *
 *   - ENTRY (half A stage-0): the window is gathered through a ROTATED
 *     table (slot v <- natural position (v + nh-1) mod N).  The OLS
 *     window shift — fp32 precedent pays it as the wrapper's post-shift
 *     sweep — sinks into the LOAD boundary as a pure plan-time table
 *     remap (zero runtime cost).  Cyclic convolution commutes with
 *     cyclic rotation, so the wrap-around-polluted head [0, nh-1) of
 *     the plain result lands at the END and the N-nh+1 valid samples
 *     become the natural PREFIX [0, N-nh] of the rotated result.
 *   - MIDDLE: unchanged conv weld sweeps / twiddles / H tables.
 *   - EXIT (half B final): the stores write ONLY the valid prefix —
 *     the discard zone [N-nh+1, N-1] is not written (skip, default) or
 *     written-then-abandoned (FFT_OLS_DISCARD=write adjudication knob);
 *     semantics unpromised either way ({#API-FFT-012}).
 *
 * The intermediate spectrum transits the plan-owned natural buffer spec_
 * exactly once (half A final store -> half B stage-0 gather — never as
 * separate wrapper sweeps), which decouples the window base from the out
 * base and lets ONE block body serve both faces:
 *
 *   fn(io)          the {#API-FFT-012} single-block contract execute
 *                   (win = out = io; rotated gather reads io once at
 *                   stage-0, valid prefix written back to io[0..N-nh]);
 *   fn_stream(P, y, B)  the O-ii form (a) EMITTED BLOCK LOOP — one call
 *                   sweeps the whole padded stream P = [0^{nh-1} | x |
 *                   0-tail]: windows form via the rotated gather at
 *                   rdi = P + b*hop, valid segments store straight to
 *                   y + b*hop at the store boundary (the wrapper pays
 *                   NO per-block window copy, NO output-side shift).
 *                   nblocks == 1 is the wrapper-fed form (b) block.
 *
 * Frame: pushes (rbx, r12..r15) + 336B so [rbp-312] = OUT slot and
 * [rbp-320 .. rbp-344] = stream-loop slots (P, y, nblocks-left, byte
 * cursor) sit below the 8x32B leaf spill frame at rbp-304.
 * ======================================================================== */
class SchedEmitterO final : public Xbyak::CodeGenerator {
public:
    struct Stage {
        int radix;          /* r_t */
        int bprev;          /* B_{t-1} in elements (t=0 -> 1) */
        int b;              /* B_t in elements */
        const double *tw;   /* TW_t base, nullptr for t=0 */
    };

    SchedEmitterO(int n, const uint32_t *gtabA, const uint32_t *gtabB,
                  const double *scratch, const double *spec,
                  const Stage *stages, int kStages, bool fma,
                  const int *groupLen, const double *hMul,
                  const double *hSwap, bool nt, int prefetch, int nh,
                  bool writeDiscard)
        : Xbyak::CodeGenerator(524288), fma_(fma),
          nt_(nt && ((1 << n) >= 4096)), prefetch_(prefetch), n_(n),
          spec_(spec), validEnd_((1 << n) - nh),
          writeDiscard_(writeDiscard),
          hopBytes_((size_t)((1 << n) - nh + 1) * 16) {
        if (kStages < 2) {
            /* unreachable by construction: the OLS build excludes k < 2
             * decompositions (same weldability rule as the conv build).
             * Fail the emission loudly. */
            throw std::bad_alloc();
        }
        int gl[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (groupLen != nullptr) {
            for (int i = 0; i < 8; ++i) {
                gl[i] = groupLen[i];
            }
        }
        (void)0; /* needSpill frame is unconditionally covered below */
        /* Frame (rbp-relative, IDENTICAL for both entry stubs): r15 sits
         * at [rbp-40]; the 8x32B leaf spill frame at rbp-304..-48; the
         * OLS slots BELOW it: [rbp-312] = out base (body), [rbp-320] P,
         * [rbp-328] y, [rbp-336] nblocks-left, [rbp-344] byte cursor
         * (stream stub).  Each entry stub emits its OWN prologue — the
         * stubs are independent ABI entries, so a shared prologue ahead
         * of them would be SKIPPED (rbp invalid in the body); the
         * epilogue is shared (rbp-relative, frame-shape-equal). */
        spillOff_ = -304;

        const uint64_t hm = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(hMul) -
            reinterpret_cast<uintptr_t>(scratch));
        const uint64_t hs = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(hSwap) -
            reinterpret_cast<uintptr_t>(scratch));

        /* ---- single-block contract entry: fn(io) — one window, win =
         * out = io (the intermediate spectrum transits spec_, NOT io).
         * {#API-FFT-012} execute face. ---- */
        L(entryFn_);
        emitPrologue();
        mov(rdx, rdi);
        call(bodyL);
        jmp(epiL);

        /* ---- stream entry: fn2(P, y, nblocks) — the EMITTED block loop
         * (O-ii form a): window formation via the rotated gather (the
         * overlap ring sunk into the stage-0 load boundary), hop advance,
         * tail-block semantics carried by the caller's zero-padded P.
         * nblocks MUST be >= 1. ---- */
        L(entryStream_);
        emitPrologue();
        mov(ptr[rbp - 320], rdi); /* P */
        mov(ptr[rbp - 328], rsi); /* y */
        mov(ptr[rbp - 336], rdx); /* nblocks */
        mov(qword[rbp - 344], 0); /* byte cursor */
        L(loopL_);
        mov(rdi, ptr[rbp - 320]);
        add(rdi, ptr[rbp - 344]);
        mov(rdx, ptr[rbp - 328]);
        add(rdx, ptr[rbp - 344]);
        call(bodyL);
        add(qword[rbp - 344], static_cast<uint32_t>(hopBytes_));
        sub(qword[rbp - 336], 1);
        jnz(loopL_);

        L(epiL);
        vzeroupper();
        lea(rsp, ptr[rbp - 40]); /* at the pushed r15 slot */
        pop(r15);
        pop(r14);
        pop(r13);
        pop(r12);
        pop(rbx);
        pop(rbp);
        ret();

        /* ---- ONE OLS block: rdi = window base, rdx = out base.  The
         * window is gathered through the ROTATED table (half A), the
         * spectrum transits the plan-owned natural buffer spec_ exactly
         * once, and the half-B final stores write ONLY the valid prefix
         * [0, N-nh] of the out block (skip mode). ---- */
        L(bodyL);
        mov(ptr[rbp - 312], rdx); /* out base survives the stage-0 sweeps */
        mov(r14, hm);
        mov(r15, hs);
        /* half A: forward of the rotated window; final = conj(X*H) */
        emitHalfOls(1 << n, gtabA, scratch, stages, kStages, gl, /*mode=*/1);
        /* half B: inverse (conjugate trick) of the natural spectrum. */
        mov(rdi, reinterpret_cast<uint64_t>(spec_));
        emitHalfOls(1 << n, gtabB, scratch, stages, kStages, gl, /*mode=*/2);
        ret();

        emitPool();
    }

    /* per-entry-stub prologue (see the constructor comment: each entry
     * is an independent ABI entry and must build its own frame). */
    void emitPrologue() {
        push(rbp);
        mov(rbp, rsp);
        push(rbx);
        push(r12);
        push(r13); /* r13 = stage-0-group gather-tab cursor */
        push(r14); /* r14 = hMul - scratch (half A anchor) */
        push(r15); /* r15 = hSwap - scratch (half A anchor) */
        sub(rsp, 336);
    }

    /* Entry addresses (stable for the kernel's lifetime; readyRE by the
     * caller).  fn = single-block contract face; stream = O-ii form (a)
     * emitted block loop (also serves form (b) with nblocks == 1). */
    fft_jit_fn_t fnEntry() const {
        return reinterpret_cast<fft_jit_fn_t>(
            const_cast<uint8_t *>(entryFn_.getAddress()));
    }
    fft_jit_ols_fn_t streamEntry() const {
        return reinterpret_cast<fft_jit_ols_fn_t>(
            const_cast<uint8_t *>(entryStream_.getAddress()));
    }

private:
    static int lf_log2(int v) {
        int s = 0;
        while ((1 << s) < v) {
            ++s;
        }
        return s;
    }

    /* ---- one transform half (stage 0, middle sweeps, fused final).
     * finalMode_ stays 0 through the gather / middle sweeps; only the
     * final stage runs with the half's fused mode. ---- */
    void emitHalfOls(int N, const uint32_t *gatherTab,
                     const double *scratch, const Stage *stages, int kStages,
                     const int *groupLen, int mode) {
        finalMode_ = 0;
        olsMask_ = false;
        int t = 0;
        if (groupLen[0] >= 2) {
            emitStage0Group(N, gatherTab, scratch, stages, groupLen[0]);
            t = groupLen[0];
        } else {
            emitStage0Plain(N, gatherTab, scratch, stages[0]);
            t = 1;
        }
        while (t < kStages - 1) {
            const int d = groupLen[t] >= 2 ? groupLen[t] : 1;
            if (d >= 2) {
                emitTileGroup(stages, t, d, N);
            } else {
                emitSweepFull(stages[t], N, /*allowNt=*/true);
            }
            t += d;
        }
        /* final stage: reads the scratch, writes the natural array with
         * the fused boundary op (mode).  Stores are NEVER non-temporal
         * (the caller / the next half reads the result immediately).
         * Half B (mode 2) stores ONLY the OLS valid prefix — the discard
         * zone [N-nh+1, N-1] is not written (skip mode, default). */
        if (mode == 2 && !writeDiscard_) {
            olsMask_ = true;
        }
        emitFinalStage(stages[kStages - 1], N, mode);
        olsMask_ = false;
    }

    /* ---- stage 0, unblocked (R0/R1 shape): digitrev gather fused into
     * the first codelet sweep, out-of-place into the scratch ---- */
    void emitStage0Plain(int N, const uint32_t *gatherTab,
                         const double *scratch, const Stage &st) {
        using namespace Xbyak;
        mov(r8, reinterpret_cast<uint64_t>(gatherTab));
        mov(rsi, reinterpret_cast<uint64_t>(scratch));
        lea(rdx, ptr[rsi + N * 16]); /* scratch end */
        ntSweep_ = nt_; /* scratch stores: re-read only by a later full
                           sweep -> NT candidate (R2 F7) */
        Label blk;
        L(blk);
        if (st.radix <= 16) {
            /* vector-2 over block pairs (2b, 2b+1) */
            gatherTab_ = &r8;
            emitLeafV2(st.radix, rsi, 16, rsi, false);
            gatherTab_ = nullptr;
            add(r8, st.radix * 8);   /* 2 table rows (uint32) */
            add(rsi, st.radix * 32); /* 2 blocks */
        } else {
            gatherTab_ = &r8;
            emitLeaf32(r8, 16, false, rsi);
            gatherTab_ = nullptr;
            add(r8, st.radix * 4);
            add(rsi, st.radix * 16);
        }
        cmp(rsi, rdx);
        jb(blk);
        ntSweep_ = false;
        if (nt_) {
            sfence(); /* NT scratch writes precede later loads (F7) */
        }
        /* stages 1..k-2 run in-place on the scratch.  The OLS body parks
         * NOTHING in rdx: half A's final goes to the plan-owned spectrum
         * buffer (baked immediate) and the OUT base survives in its
         * [rbp-312] slot. */
        mov(rdi, reinterpret_cast<uint64_t>(scratch));
    }

    /* ---- R2 F6: stage-0 tile group.  Per tile: gather exactly the
     * tile's digitrev rows (tab cursor r13 stays sequential across
     * tiles), write the tile in the scratch, then run stages 1..d-1
     * confined to the tile before advancing (intermediate results never
     * leave L1). ---- */
    void emitStage0Group(int N, const uint32_t *gatherTab,
                         const double *scratch, const Stage *stages,
                         int d) {
        using namespace Xbyak;
        const Stage &st = stages[0];
        const int T = stages[d - 1].b; /* tile elements */
        bool ntUsed = false;
        mov(r13, reinterpret_cast<uint64_t>(gatherTab)); /* tab cursor */
        mov(r12, reinterpret_cast<uint64_t>(scratch));   /* tile cursor */
        lea(rcx, ptr[r12 + N * 16]); /* array end (tile-loop bound) */
        Label tile;
        L(tile);
        emitPrefetchHead(r12, T * 16);
        mov(rsi, r12);
        lea(rdx, ptr[r12 + T * 16]); /* this tile's scratch end */
        Label blk;
        L(blk);
        if (st.radix <= 16) {
            gatherTab_ = &r13;
            emitLeafV2(st.radix, rsi, 16, rsi, false);
            gatherTab_ = nullptr;
            add(r13, st.radix * 8);  /* 2 table rows (uint32) */
            add(rsi, st.radix * 32); /* 2 blocks */
        } else {
            gatherTab_ = &r13;
            emitLeaf32(r13, 16, false, rsi);
            gatherTab_ = nullptr;
            add(r13, st.radix * 4);
            add(rsi, st.radix * 16);
        }
        cmp(rsi, rdx);
        jb(blk);
        for (int s = 1; s < d; ++s) {
            const bool last = (s == d - 1);
            emitSweepInTile(stages[s], T * 16, last);
            ntUsed |= (nt_ && last);
        }
        add(r12, T * 16);
        cmp(r12, rcx);
        jb(tile);
        if (ntUsed) {
            sfence(); /* last-in-group NT writes precede later loads */
        }
        /* the scratch becomes the working array (no rdx park — the OLS
         * OUT base lives in its [rbp-312] slot, the spectrum buffer is a
         * baked immediate) */
        mov(rdi, reinterpret_cast<uint64_t>(scratch));
    }

    /* ---- one stage's sweep confined to the current tile ([r12,
     * r12+tileBytes)); in-tile stores stay WB (they are re-read from L1
     * inside the tile) EXCEPT the last stage of the group (its output is
     * consumed only by a later full sweep -> NT candidate). ---- */
    void emitSweepInTile(const Stage &st, int tileBytes, bool lastOfGroup) {
        using namespace Xbyak;
        ntSweep_ = nt_ && lastOfGroup;
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, r12);
        lea(r8, ptr[r12 + tileBytes]);
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 16]); /* k-loop end */
        mov(rax, rsi);
        mov(r10, r9);
        Label kLoop;
        L(kLoop);
        if (st.radix <= 16) {
            /* vector-2 over k pairs (k, k+1) */
            emitLeafV2(st.radix, rax, st.bprev * 16, rax, true);
            add(rax, 32);
            add(r10, st.radix * 64);
        } else {
            emitLeaf32(rax, st.bprev * 16, true, rax);
            add(rax, 16);
            add(r10, 32 * 32);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 16);
        cmp(rsi, r8);
        jb(blk);
        ntSweep_ = false;
    }

    /* ---- one stage's unblocked full-array sweep (R0/R1 shape); NT
     * stores + trailing sfence when enabled (output re-read only by a
     * later full sweep). ---- */
    void emitSweepFull(const Stage &st, int N, bool allowNt) {
        using namespace Xbyak;
        ntSweep_ = nt_ && allowNt;
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, rdi);
        lea(r8, ptr[rdi + N * 16]); /* end (r8 table dead) */
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 16]); /* k-loop end */
        mov(rax, rsi);
        mov(r10, r9);
        Label kLoop;
        L(kLoop);
        if (st.radix <= 16) {
            /* vector-2 over k pairs (k, k+1) */
            emitLeafV2(st.radix, rax, st.bprev * 16, rax, true);
            add(rax, 32);
            add(r10, st.radix * 64);
        } else {
            emitLeaf32(rax, st.bprev * 16, true, rax);
            add(rax, 16);
            add(r10, 32 * 32);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 16);
        cmp(rsi, r8);
        jb(blk);
        ntSweep_ = false;
        if (nt_ && allowNt) {
            sfence();
        }
    }

    /* ---- R2 F6: middle tile group (stages t..t+d-1, t >= 1) ---- */
    void emitTileGroup(const Stage *stages, int t, int d, int N) {
        using namespace Xbyak;
        const int T = stages[t + d - 1].b; /* tile elements */
        bool ntUsed = false;
        mov(r12, rdi); /* tile cursor = scratch base */
        lea(rcx, ptr[rdi + N * 16]);
        Label tile;
        L(tile);
        emitPrefetchHead(r12, T * 16);
        for (int s = t; s < t + d; ++s) {
            const bool last = (s == t + d - 1);
            emitSweepInTile(stages[s], T * 16, last);
            ntUsed |= (nt_ && last);
        }
        add(r12, T * 16);
        cmp(r12, rcx);
        jb(tile);
        if (ntUsed) {
            sfence(); /* last-in-group NT writes precede later loads */
        }
    }

    /* ---- R2 F7: prefetcht0 the head (8 cache lines) of the NEXT tile
     * at every tile-loop boundary (knob-gated; prefetch of an address
     * past the array end is architecturally harmless). ---- */
    void emitPrefetchHead(const Xbyak::Reg64 &base, int tileBytes) {
        if (prefetch_ <= 0) {
            return;
        }
        for (int i = 0; i < 8; ++i) {
            prefetcht0(ptr[base + tileBytes + i * 64]);
        }
    }

    /* ---- final stage: reads the scratch, writes the natural array with
     * the fused boundary op (mode 1 = conj(X*H), 2 = conj*invN).
     * rax = scratch k-cursor anchors the H tables via r14/r15. ---- */
    void emitFinalStage(const Stage &st, int N, int mode) {
        using namespace Xbyak;
        finalMode_ = mode;
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, rdi);
        lea(r8, ptr[rdi + N * 16]); /* end (r8 table dead) */
        if (mode == 1) {
            /* half A's natural array = the plan-owned spectrum buffer */
            mov(r12, reinterpret_cast<uint64_t>(spec_));
        } else {
            /* half B's natural array = the OUT base saved at body entry */
            mov(r12, ptr[rbp - 312]);
        }
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 16]); /* k-loop end */
        mov(rax, rsi);
        mov(r10, r9);
        if (olsMask_) {
            /* topic OLS predication anchor: r9's TW role is dead from
             * here (the final stage runs exactly ONE blk iteration —
             * b == N); r9 := the OUT base so the store predicates can
             * compare (storeBase - r9) against the constant valid-end
             * offsets.  rdx is dead in the final stage (scratch). */
            mov(r9, r12);
        }
        Label kLoop;
        L(kLoop);
        if (st.radix <= 16) {
            /* vector-2 over k pairs (k, k+1) */
            mov(rbx, r12);
            emitLeafV2(st.radix, rax, st.bprev * 16, rbx, true);
            add(rax, 32);
            add(r10, st.radix * 64);
            add(r12, 32);
        } else {
            mov(rbx, r12);
            emitLeaf32(rax, st.bprev * 16, true, rbx);
            add(rax, 16);
            add(r10, 32 * 32);
            add(r12, 16);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 16);
        add(r12, st.b * 16);
        cmp(rsi, r8);
        jb(blk);
        finalMode_ = 0;
    }

    /* ---- topic jit-fp64-conv-fuse: fused final-stage boundary op, in
     * place on a (k, k+1) natural pair chunk.  finalMode_ 1: v <-
     * conj(v * H_pair) with the natural H pair at [r14 + rax + off] and
     * its pre-swapped twin at [r15 + rax + off] (FMA3 twin = (hi, +hr),
     * non-FMA = (hi, -hr) — same idiom/sign discipline as the twiddle
     * twins).  finalMode_ 2: v <- conj(v) * (1/N) via one vmulpd. ---- */
    void fuseFinalPairInPlace(const Xbyak::Ymm &v, int off) {
        using namespace Xbyak;
        if (finalMode_ == 1) {
            if (fma_) {
                vmovddup(ymm12, v);            /* (ar, ar) per lane */
                vshufpd(v, v, v, 0xF);         /* (ai, ai) per lane */
                vmulpd(v, v, ptr[r15 + rax + off]);
                vfmaddsub231pd(v, ymm12, ptr[r14 + rax + off]);
            } else {
                vmulpd(ymm12, v, ptr[r14 + rax + off]);
                vmulpd(ymm13, v, ptr[r15 + rax + off]);
                vhsubpd(v, ymm12, ymm13);
            }
            vxorpd(v, v, ptr[rip + sign2Lbl_]);
        } else if (finalMode_ == 2) {
            vmulpd(v, v, ptr[rip + invNLbl_]);
        }
    }

    /* ---- vector-2 leaf dispatch (radix 2/4/8/16) ----
     * Loads come from loadBase + slot*strideBytes (twiddled stages) or
     * through the gatherTab_ cursor (stage 0); stores go to storeBase +
     * pos*strideBytes (twiddled) or storeBase + pos*16 / (r+pos)*16 for
     * the two gathered blocks (stage 0, contiguous scratch).  Twiddle
     * rows (hasTw) come from r10: 64B slots per (k-pair, j). */
    void emitLeafV2(int r, const Xbyak::Reg64 &loadBase, int strideBytes,
                    const Xbyak::Reg64 &storeBase, bool hasTw) {
        using namespace Xbyak;
        if (r >= 4 || (!fma_ && hasTw)) {
            vmovapd(ymm14, ptr[rip + sign2Lbl_]);
        }
        if (r == 16) {
            emitV2Sub(8, loadBase, strideBytes, hasTw, 0);
            for (int j = 0; j < 8; ++j) {
                vmovupd(ptr[rbp + spillOff_ + j * 32], Ymm(j));
            }
            emitV2Sub(8, loadBase, strideBytes, hasTw, 1);
            cross16V2(storeBase, strideBytes);
        } else {
            emitV2Sub(r, loadBase, strideBytes, hasTw, -1);
            storeV2(r, storeBase, strideBytes);
        }
    }

    /* r-point DFT in per-position vector-2 registers P[0..r-1] (r <= 8).
     * par < 0: whole leaf (slot(pos) = bitrev(pos, log2 r));
     * par >= 0: sub-leaf of the radix-16 leaf (slot(pos) =
     * 2*bitrev(pos, 3) + par; gather rows stride 16). */
    void emitV2Sub(int r, const Xbyak::Reg64 &loadBase, int strideBytes,
                   bool hasTw, int par) {
        using namespace Xbyak;
        const int nlf = lf_log2(r);
        const int rr = (par >= 0) ? 16 : r; /* gather row stride (r1) */
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};

        /* ---- load (+twiddle) + m=2 fuse ---- */
        for (int u = 0; u < r / 2; ++u) {
            int sa, sb;
            if (par >= 0) {
                sa = 2 * (int)bitrev((uint32_t)(2 * u), nlf) + par;
                sb = 2 * (int)bitrev((uint32_t)(2 * u + 1), nlf) + par;
            } else {
                sa = (int)bitrev((uint32_t)(2 * u), nlf);
                sb = (int)bitrev((uint32_t)(2 * u + 1), nlf);
            }
            if (gatherTab_ != nullptr) {
                const Reg64 &tab = *gatherTab_;
                mov(eax, dword[tab + sa * 4]);
                vmovupd(xmm8, ptr[rdi + rax]);
                mov(eax, dword[tab + (rr + sa) * 4]);
                vmovupd(xmm9, ptr[rdi + rax]);
                mov(eax, dword[tab + sb * 4]);
                vmovupd(xmm10, ptr[rdi + rax]);
                mov(eax, dword[tab + (rr + sb) * 4]);
                vmovupd(xmm11, ptr[rdi + rax]);
                /* xmm14 holds the sign mask (ymm14) — the second
                 * block's fuse reuses the now-dead load regs */
                vaddpd(xmm12, xmm8, xmm10);
                vsubpd(xmm13, xmm8, xmm10);
                vaddpd(xmm8, xmm9, xmm11);
                vsubpd(xmm10, xmm9, xmm11);
                vinsertf128(P[2 * u], P[2 * u], xmm12, 0);
                vinsertf128(P[2 * u], P[2 * u], xmm8, 1);
                vinsertf128(P[2 * u + 1], P[2 * u + 1], xmm13, 0);
                vinsertf128(P[2 * u + 1], P[2 * u + 1], xmm10, 1);
            } else {
                loadTwV2(ymm10, ptr[loadBase + sa * strideBytes], sa, hasTw);
                loadTwV2(ymm11, ptr[loadBase + sb * strideBytes], sb, hasTw);
                vaddpd(P[2 * u], ymm10, ymm11);
                vsubpd(P[2 * u + 1], ymm10, ymm11);
            }
        }

        /* ---- stages 2..nlf (butterflies pair positions (i, i+h)) ---- */
        for (int st = 2; st <= nlf; ++st) {
            const int m = 1 << st;
            const int h = m >> 1;
            for (int blk = 0; blk < r / m; ++blk) {
                for (int i = 0; i < h; ++i) {
                    const Ymm &bot = P[blk * m + i];
                    const Ymm &top = P[blk * m + h + i];
                    if (i == 0) {
                        /* T = top in place: diff via scratch */
                        vsubpd(ymm12, bot, top);
                        vaddpd(bot, bot, top);
                        vmovapd(top, ymm12);
                    } else if (i == (m >> 2)) {
                        /* W_m^{m/4} = -i: swap + sign flip */
                        vshufpd(ymm12, top, top, 0x5);
                        vxorpd(ymm12, ymm12, ymm14);
                        vsubpd(top, bot, ymm12);
                        vaddpd(bot, bot, ymm12);
                    } else {
                        mulTwPoolV2(top, st, i);
                        vsubpd(top, bot, ymm12);
                        vaddpd(bot, bot, ymm12);
                    }
                }
            }
        }
    }

    /* load the (k, k+1) pair at addr; twiddle by TW row slot (64B slots
     * at r10 + slot*64) when slot != 0; result in dst (ymm10/ymm11). */
    void loadTwV2(const Xbyak::Ymm &dst, const Xbyak::Operand &addr,
                  int slot, bool hasTw) {
        using namespace Xbyak;
        if (slot == 0 || !hasTw) {
            vmovupd(dst, addr);
            return;
        }
        vmovupd(ymm8, addr);
        if (fma_) {
            vmovddup(ymm9, ymm8);            /* (ar, ar) per lane */
            vshufpd(dst, ymm8, ymm8, 0xF);   /* (ai, ai) per lane */
            vmulpd(dst, dst, ptr[r10 + slot * 64 + 32]);
            vfmaddsub231pd(dst, ymm9, ptr[r10 + slot * 64]);
        } else {
            vmovupd(ymm12, ptr[r10 + slot * 64]); /* W pair */
            vshufpd(ymm13, ymm12, ymm12, 0x5);
            vxorpd(ymm13, ymm13, ymm14);          /* (wi, -wr) */
            vmulpd(dst, ymm8, ymm12);
            vmulpd(ymm13, ymm8, ymm13);
            vhsubpd(dst, dst, ymm13);
        }
    }

    /* T (ymm12) = top * W_{2^st}^e from the rip pool (broadcast const +
     * pre-swapped twin as memory operands; non-FMA derives the conjugate
     * in registers). */
    void mulTwPoolV2(const Xbyak::Ymm &top, int st, int e) {
        using namespace Xbyak;
        if (fma_) {
            vmovddup(ymm13, top);
            vshufpd(ymm12, top, top, 0xF);
            vmulpd(ymm12, ymm12, ptr[rip + twSwapLbl_[st][e]]);
            vfmaddsub231pd(ymm12, ymm13, ptr[rip + twLbl_[st][e]]);
        } else {
            vmovapd(ymm13, ptr[rip + twLbl_[st][e]]);
            vshufpd(ymm15, ymm13, ymm13, 0x5);
            vxorpd(ymm15, ymm15, ymm14);
            vmulpd(ymm12, top, ymm13);
            vmulpd(ymm15, top, ymm15);
            vhsubpd(ymm12, ymm12, ymm15);
        }
    }

    /* radix-16 cross: y[j] = e[j] + W_16^j o[j], y[j+8] = e[j] - W_16^j o[j]
     * (j = 0..7).  e[] reloaded from the spill frame, o[] in P. */
    void cross16V2(const Xbyak::Reg64 &storeBase, int strideBytes) {
        using namespace Xbyak;
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};
        for (int j = 0; j < 8; ++j) {
            vmovupd(ymm8, ptr[rbp + spillOff_ + j * 32]); /* e[j] */
            if (j == 0) {
                vaddpd(ymm10, ymm8, P[0]);
                vsubpd(ymm11, ymm8, P[0]);
            } else if (j == 4) {
                /* W_16^4 = -i */
                vshufpd(ymm12, P[4], P[4], 0x5);
                vxorpd(ymm12, ymm12, ymm14);
                vaddpd(ymm10, ymm8, ymm12);
                vsubpd(ymm11, ymm8, ymm12);
            } else {
                mulTwPoolV2(P[j], 4, j); /* W_16^j = W_{2^4}^j */
                vaddpd(ymm10, ymm8, ymm12);
                vsubpd(ymm11, ymm8, ymm12);
            }
            storeV2One(ymm10, j, storeBase, strideBytes, 16);
            storeV2One(ymm11, j + 8, storeBase, strideBytes, 16);
        }
    }

    void storeV2(int r, const Xbyak::Reg64 &storeBase, int strideBytes) {
        using namespace Xbyak;
        for (int p = 0; p < r; ++p) {
            storeV2One(Ymm(p), p, storeBase, strideBytes, r);
        }
    }

    /* store position pos: whole 32B ym at storeBase + pos*stride (normal
     * / final stages), or per-block 16B extracts into the two gathered
     * blocks (stage 0: storeBase + pos*16 and storeBase + (r+pos)*16).
     * FUSED final stages (finalMode_ != 0): the boundary op runs on the
     * ym first (X*H + conj, or conj*invN — see fuseFinalPairInPlace).
     * R2 F7: when the sweep is NT-eligible the stores are vmovntpd
     * (non-temporal; the sfence lives at the sweep/tile boundary). */
    void storeV2One(const Xbyak::Ymm &v, int pos,
                    const Xbyak::Reg64 &storeBase, int strideBytes, int r) {
        using namespace Xbyak;
        if (gatherTab_ != nullptr) {
            vextractf128(xmm12, v, 0);
            if (ntSweep_) {
                vmovntpd(ptr[storeBase + pos * 16], xmm12);
            } else {
                vmovupd(ptr[storeBase + pos * 16], xmm12);
            }
            vextractf128(xmm12, v, 1);
            if (ntSweep_) {
                vmovntpd(ptr[storeBase + (r + pos) * 16], xmm12);
            } else {
                vmovupd(ptr[storeBase + (r + pos) * 16], xmm12);
            }
            return;
        }
        const int off = pos * strideBytes;
        if (finalMode_ != 0) {
            fuseFinalPairInPlace(v, off);
            if (olsMask_) {
                storePairOls(v, storeBase, off, pos, strideBytes);
                return;
            }
            vmovupd(ptr[storeBase + off], v);
            return;
        }
        if (ntSweep_) {
            vmovntpd(ptr[storeBase + off], v);
        } else {
            vmovupd(ptr[storeBase + off], v);
        }
    }

    /* ---- single-k radix-32 leaf (pair-ymm lanes = adjacent DIT
     * positions of one k; vector-2 would need 32 live yms) ---- */
    void emitLeaf32(const Xbyak::Reg64 &loadBase, int strideBytes,
                    bool hasTw, const Xbyak::Reg64 &storeBase) {
        using namespace Xbyak;
        vmovapd(ymm14, ptr[rip + sign2Lbl_]);
        emitSubLeaf16Sk(loadBase, strideBytes, hasTw, 0);
        for (int p = 0; p < 8; ++p) {
            vmovupd(ptr[rbp + spillOff_ + p * 32], Ymm(p));
        }
        emitSubLeaf16Sk(loadBase, strideBytes, hasTw, 1);
        /* cross: y[k] = e[k] + W_32^k o[k], pair chunks (W^{2p}, W^{2p+1})
         * from the rip pool (mixing foldable and general lanes — the
         * chunked multiply stays; e-pair reloaded from the spill frame) */
        for (int p = 0; p < 8; ++p) {
            vmovupd(ymm9, ptr[rbp + spillOff_ + p * 32]); /* A pair */
            if (fma_) {
                vmovddup(ymm11, Ymm(p));
                vshufpd(ymm8, Ymm(p), Ymm(p), 0xF);
                vmulpd(ymm8, ymm8, ptr[rip + w32SwapLbl_[p]]);
                vfmaddsub231pd(ymm8, ymm11, ptr[rip + w32Lbl_[p]]);
            } else {
                vmovupd(ymm10, ptr[rip + w32Lbl_[p]]);
                vshufpd(ymm12, ymm10, ymm10, 0x5);
                vxorpd(ymm12, ymm12, ymm14);
                vmulpd(ymm8, Ymm(p), ymm10);
                vmulpd(ymm12, Ymm(p), ymm12);
                vhsubpd(ymm8, ymm8, ymm12);
            }
            vaddpd(ymm13, ymm9, ymm8);  /* y[k]    -> slots (2p, 2p+1) */
            vsubpd(ymm15, ymm9, ymm8);  /* y[k+16] -> (2p+16, 2p+17);
                                           ymm14 = sgn must stay live */
            if (strideBytes == 16) {
                /* stage-0 gather (the N==32 single-codelet transform is
                 * excluded from the weld by construction) */
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p) * 16], ymm13);
                    vmovntpd(ptr[storeBase + (2 * p + 16) * 16], ymm15);
                } else {
                    vmovupd(ptr[storeBase + (2 * p) * 16], ymm13);
                    vmovupd(ptr[storeBase + (2 * p + 16) * 16], ymm15);
                }
            } else {
                if (finalMode_ != 0) {
                    /* fused final: boundary op on the pair chunks BEFORE
                     * the extracts (H pairs are natural-contiguous at
                     * rax + (2p / 2p+16) * stride) */
                    fuseFinalPairInPlace(ymm13, (2 * p) * strideBytes);
                    fuseFinalPairInPlace(ymm15, (2 * p + 16) * strideBytes);
                }
                /* topic OLS: final-stage half stores run through the
                 * valid-prefix predicate (each 16B extract = ONE natural
                 * position k + q*bprev); non-final sweeps keep the
                 * NT/plain store discipline of the conv weld. */
                vextractf128(xmm8, ymm13, 0);
                olsStoreHalf(xmm8, storeBase, (2 * p) * strideBytes, 2 * p,
                             strideBytes);
                vextractf128(xmm8, ymm13, 1);
                olsStoreHalf(xmm8, storeBase, (2 * p + 1) * strideBytes,
                             2 * p + 1, strideBytes);
                vextractf128(xmm8, ymm15, 0);
                olsStoreHalf(xmm8, storeBase, (2 * p + 16) * strideBytes,
                             2 * p + 16, strideBytes);
                vextractf128(xmm8, ymm15, 1);
                olsStoreHalf(xmm8, storeBase, (2 * p + 17) * strideBytes,
                             2 * p + 17, strideBytes);
            }
        }
    }

    /* DFT-16 sub-leaf of the radix-32 leaf: input slot =
     * 2*bitrev4(pos) + par, result left in ymm0..7 (pair layout). */
    void emitSubLeaf16Sk(const Xbyak::Reg64 &loadBase, int strideBytes,
                         bool hasTw, int par) {
        using namespace Xbyak;
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};

        for (int p = 0; p < 8; ++p) {
            const int sa = 2 * (int)bitrev((uint32_t)(2 * p), 4) + par;
            const int sb = 2 * (int)bitrev((uint32_t)(2 * p + 1), 4) + par;
            if (gatherTab_ != nullptr) {
                const Reg64 &tab = *gatherTab_;
                mov(eax, dword[tab + sa * 4]);
                vmovupd(xmm8, ptr[rdi + rax]);
                mov(eax, dword[tab + sb * 4]);
                vmovupd(xmm9, ptr[rdi + rax]);
            } else {
                vmovupd(xmm8, ptr[loadBase + sa * strideBytes]);
                vmovupd(xmm9, ptr[loadBase + sb * strideBytes]);
            }
            if (hasTw && sa != 0) {
                mulTwXmm(xmm10, xmm8, sa);
            }
            if (hasTw && sb != 0) {
                mulTwXmm(xmm12, xmm9, sb);
            }
            const Xmm &a2 = (hasTw && sa != 0) ? xmm10 : xmm8;
            const Xmm &b2 = (hasTw && sb != 0) ? xmm12 : xmm9;
            /* xmm14 reserved for the sign mask (ymm14) */
            vaddpd(xmm13, a2, b2);
            vsubpd(xmm15, a2, b2);
            vinsertf128(P[p], P[p], xmm13, 0);
            vinsertf128(P[p], P[p], xmm15, 1);
        }

        /* stage 2 (m=4): top HIGH lane x (-i) blended into the scratch */
        for (int blk = 0; blk < 4; ++blk) {
            const Ymm &bot = P[2 * blk];
            const Ymm &top = P[2 * blk + 1];
            vshufpd(ymm8, top, top, 0x5);
            vxorpd(ymm8, ymm8, ymm14);
            vblendpd(ymm8, top, ymm8, 0x0C);
            vsubpd(top, bot, ymm8);
            vaddpd(bot, bot, ymm8);
        }

        /* stages 3..4: pair chunks (W^{2q}, W^{2q+1}) from the rip pool */
        for (int st = 3; st <= 4; ++st) {
            const int halfPairs = 1 << (st - 2);
            const int pairsPerBlock = 1 << (st - 1);
            for (int blk = 0; blk < 16 / (1 << st); ++blk) {
                const int bp = blk * pairsPerBlock;
                for (int q = 0; q < halfPairs; ++q) {
                    const Ymm &bot = P[bp + q];
                    const Ymm &top = P[bp + halfPairs + q];
                    if (fma_) {
                        vmovddup(ymm11, top);
                        vshufpd(ymm8, top, top, 0xF);
                        vmulpd(ymm8, ymm8, ptr[rip + lwSwapLbl_[st][q]]);
                        vfmaddsub231pd(ymm8, ymm11, ptr[rip + lwLbl_[st][q]]);
                    } else {
                        vmovapd(ymm10, ptr[rip + lwLbl_[st][q]]);
                        vshufpd(ymm12, ymm10, ymm10, 0x5);
                        vxorpd(ymm12, ymm12, ymm14);
                        vmulpd(ymm8, top, ymm10);
                        vmulpd(ymm12, top, ymm12);
                        vhsubpd(ymm8, ymm8, ymm12);
                    }
                    vsubpd(top, bot, ymm8);
                    vaddpd(bot, bot, ymm8);
                }
            }
        }
    }

    /* dst (xmm) = d * TW[slot]; 32B slots at r10 + slot*32 =
     * (wr, wi, wi, +wr|FMA / -wr|no-FMA) — both halves are memory
     * operands of the complex multiply. */
    void mulTwXmm(const Xbyak::Xmm &dst, const Xbyak::Xmm &d, int slot) {
        using namespace Xbyak;
        if (fma_) {
            vmovddup(xmm11, d);             /* (dr, dr) */
            vshufpd(dst, d, d, 3);          /* (di, di) */
            vmulpd(dst, dst, ptr[r10 + slot * 32 + 16]);
            vfmaddsub231pd(dst, xmm11, ptr[r10 + slot * 32]);
        } else {
            vmulpd(dst, d, ptr[r10 + slot * 32]);
            vmulpd(xmm11, d, ptr[r10 + slot * 32 + 16]);
            vhsubpd(dst, dst, xmm11);
        }
    }

    /* rip-relative constant pool: sign mask + vector-2 broadcast
     * twiddles (+ swap twins) + single-k pair chunks (+ twins) + W_32
     * cross pairs (+ twins) + the weld's (invN, -invN) constant.
     * Shared by every leaf of the plan. */
    void emitPool() {
        using namespace Xbyak;
        align(32);
        L(sign2Lbl_);
        dq(0);
        dq(0x8000000000000000ull);
        dq(0);
        dq(0x8000000000000000ull);

        /* topic jit-fp64-conv-fuse: conj(.) * (1/N) in one vmulpd */
        const double inv = 1.0 / (double)(1 << n_);
        align(32);
        L(invNLbl_);
        dq(bits_from_double(inv));
        dq(bits_from_double(-inv));
        dq(bits_from_double(inv));
        dq(bits_from_double(-inv));

        /* W_{2^s}^e for s = 3..4, e = 1..2^{s-1}-1 (e = 0 skipped = 1,
         * e = 2^{s-2} skipped = -i fold); s=4 entries double as the
         * radix-16 cross W_16^j. */
        for (int s = 3; s <= 4; ++s) {
            for (int e = 1; e < (1 << (s - 1)); ++e) {
                if (e == (1 << (s - 2))) {
                    continue;
                }
                const double ang =
                    -2.0 * kPi * (double)e / (double)(1 << s);
                const double wr = std::cos(ang);
                const double wi = std::sin(ang);
                align(32);
                L(twLbl_[s][e]);
                dq(bits_from_double(wr));
                dq(bits_from_double(wi));
                dq(bits_from_double(wr));
                dq(bits_from_double(wi));
                align(32);
                L(twSwapLbl_[s][e]);
                dq(bits_from_double(wi));
                dq(bits_from_double(wr));
                dq(bits_from_double(wi));
                dq(bits_from_double(wr));
            }
        }

        /* single-k pair chunks: chunk q = (W_{2^s}^{2q}, W_{2^s}^{2q+1}) */
        for (int s = 3; s <= 4; ++s) {
            const int halfPairs = 1 << (s - 2);
            for (int q = 0; q < halfPairs; ++q) {
                double c[2], sn[2];
                for (int l = 0; l < 2; ++l) {
                    const int e = 2 * q + l;
                    const double ang =
                        -2.0 * kPi * (double)e / (double)(1 << s);
                    c[l] = std::cos(ang);
                    sn[l] = std::sin(ang);
                }
                align(32);
                L(lwLbl_[s][q]);
                dq(bits_from_double(c[0]));
                dq(bits_from_double(sn[0]));
                dq(bits_from_double(c[1]));
                dq(bits_from_double(sn[1]));
                align(32);
                L(lwSwapLbl_[s][q]);
                dq(bits_from_double(sn[0]));
                dq(bits_from_double(c[0]));
                dq(bits_from_double(sn[1]));
                dq(bits_from_double(c[1]));
            }
        }

        /* W_32 cross pairs: chunk p = (W_32^{2p}, W_32^{2p+1}) */
        for (int p = 0; p < 8; ++p) {
            double c[2], sn[2];
            for (int l = 0; l < 2; ++l) {
                const int e = 2 * p + l;
                const double ang = -2.0 * kPi * (double)e / 32.0;
                c[l] = std::cos(ang);
                sn[l] = std::sin(ang);
            }
            align(32);
            L(w32Lbl_[p]);
            dq(bits_from_double(c[0]));
            dq(bits_from_double(sn[0]));
            dq(bits_from_double(c[1]));
            dq(bits_from_double(sn[1]));
            align(32);
            L(w32SwapLbl_[p]);
            dq(bits_from_double(sn[0]));
            dq(bits_from_double(c[0]));
            dq(bits_from_double(sn[1]));
            dq(bits_from_double(c[1]));
        }
    }

    static uint64_t bits_from_double(double d) {
        uint64_t u;
        __builtin_memcpy(&u, &d, sizeof(u));
        return u;
    }

    /* topic OLS: valid-prefix store of one pair ymm — covers natural out
     * positions (k + col) and (k + col + 1), col = pos*bprev emit-time
     * constant, k = storeBase/16 the runtime pair base; validEnd_ = N-nh
     * (valid positions [0, validEnd_] inclusive).  storeBase walks in
     * 32B steps, so k is always even; the straddle (low valid, high in
     * the discard zone) happens exactly at k == validEnd_ - col. */
    void storePairOls(const Xbyak::Ymm &v, const Xbyak::Reg64 &storeBase,
                      int off, int pos, int strideBytes) {
        using namespace Xbyak;
        const int bprev = strideBytes / 16;
        const int col = pos * bprev;
        const int t = validEnd_ - col;
        if (t >= bprev - 1) {
            vmovupd(ptr[storeBase + off], v); /* fully valid column */
            return;
        }
        if (t < 0) {
            return; /* fully in the discard zone: not written (skip) */
        }
        if (t & 1) {
            /* odd boundary: pairs are both-valid or both-invalid */
            Label skipL;
            mov(rdx, storeBase);
            sub(rdx, r9);
            cmp(rdx, t * 16);
            jae(skipL);
            vmovupd(ptr[storeBase + off], v);
            L(skipL);
            return;
        }
        Label skipL, straddleL, doneL;
        mov(rdx, storeBase);
        sub(rdx, r9);
        cmp(rdx, t * 16);
        ja(skipL);
        je(straddleL);
        vmovupd(ptr[storeBase + off], v);
        jmp(doneL);
        L(straddleL); /* only the LOW 16B (position col + k) is valid */
        vextractf128(xmm12, v, 0);
        vmovupd(ptr[storeBase + off], xmm12);
        jmp(doneL);
        L(skipL);
        L(doneL);
    }

    /* topic OLS: valid-prefix store of one 16B half (single natural
     * position k + q*bprev; radix-32 final path). */
    void olsStoreHalf(const Xbyak::Xmm &v, const Xbyak::Reg64 &storeBase,
                      int off, int pos, int strideBytes) {
        using namespace Xbyak;
        if (!olsMask_) {
            if (ntSweep_) {
                vmovntpd(ptr[storeBase + off], v);
            } else {
                vmovupd(ptr[storeBase + off], v);
            }
            return;
        }
        const int bprev = strideBytes / 16;
        const int col = pos * bprev;
        const int t = validEnd_ - col;
        if (t >= bprev - 1) {
            vmovupd(ptr[storeBase + off], v);
            return;
        }
        if (t < 0) {
            return;
        }
        Label skipL;
        mov(rdx, storeBase); /* rdx dead in the final stage */
        sub(rdx, r9);        /* r9 = OUT base (predication anchor) */
        cmp(rdx, t * 16);    /* cursor offset (k*16) vs t*16 */
        ja(skipL);
        vmovupd(ptr[storeBase + off], v);
        L(skipL);
    }

    bool fma_;
    bool nt_ = false;         /* R2 F7: NT stores enabled (N >= 4096) */
    int prefetch_ = 0;        /* R2 F7: tile-head prefetcht0 (0 = off) */
    bool ntSweep_ = false;    /* current sweep's stores are NT-eligible */
    int spillOff_ = -304; /* r16/r32 leaf spill base offset from rbp */
    int n_ = 0;               /* log2(N) — invN pool constant */
    int finalMode_ = 0;       /* 0 plain; 1 = store conj(X*H) (half A);
                                 2 = store conj(.)*invN (half B) */
    /* topic jit-fp64-linear-conv (O-ii store-boundary sinking) */
    const double *spec_ = nullptr; /* plan-owned natural spectrum buffer */
    int validEnd_ = 0;             /* N - nh: valid prefix [0, validEnd_] */
    bool writeDiscard_ = false;    /* FFT_OLS_DISCARD=write: full-N stores */
    bool olsMask_ = false;         /* inside the half-B final stage */
    size_t hopBytes_ = 0;          /* (N-nh+1)*16 stream advance per block */
    Xbyak::Label entryFn_;         /* single-block contract entry */
    Xbyak::Label entryStream_;     /* emitted block-loop stream entry */
    Xbyak::Label loopL_;
    Xbyak::Label epiL;
    Xbyak::Label bodyL;
    const Xbyak::Reg64 *gatherTab_ = nullptr; /* stage-0 table mode */
    Xbyak::Label sign2Lbl_;
    Xbyak::Label invNLbl_;         /* (invN, -invN, invN, -invN) */
    Xbyak::Label twLbl_[5][8];     /* [s][e]: W_{2^s}^e broadcast (v2) */
    Xbyak::Label twSwapLbl_[5][8];
    Xbyak::Label lwLbl_[5][4];     /* [s][q]: pair chunk (single-k) */
    Xbyak::Label lwSwapLbl_[5][4];
    Xbyak::Label w32Lbl_[8];
    Xbyak::Label w32SwapLbl_[8];
};

/* ========================================================================
 * topic jit-fp64-inverse (amend resume 2026-09-17, DELTA F3) —
 * SchedEmitterI: FUSED fp64 inverse kernel, conjugate//N sunk to the
 * boundaries (draft {#API-FFT-009} / {#BEH-FFT-010}).
 *
 * Copy-then-edit of SchedEmitterC above (which stays untouched — the fp64
 * conv weld path is regression-only).  ONE kernel = the inverse half of
 * the conjugate trick, IFFT(x) = conj(F(conj(x))) / N, as a SINGLE
 * transform pass over the promoted mixed-radix DIT schedule (stage-0
 * digitrev gather out-of-place into the scratch, middle sweeps in-place,
 * final stage writes the natural array):
 *
 *   - ENTRY: the conjugate is sunk into the stage-0 gather LOAD boundary
 *     — every gathered element is conjugated right after the twiddle-free
 *     m=2 load fuse (one vxorpd per pair ymm against the rip sign mask;
 *     conj is linear over the +/- fuse, so conjugating the fuse outputs
 *     is bit-exact conjugating the loads — no separate conj sweep of the
 *     caller's buffer; the leaf then computes DFT_r(conj(x_block)),
 *     i.e. exactly the stage-0 of F(conj(x)));
 *   - MIDDLE: the promoted forward sweeps / twiddles run unchanged (the
 *     conjugate trick keeps the FORWARD twiddle semantics directly
 *     usable — no inverse twiddle table, no second emitter);
 *   - EXIT: the final stage stores conj(.) * (1/N) via ONE vmulpd per
 *     store vector against the (invN, -invN) rip constant (the conv
 *     weld half-B finalMode-2 idiom) — no separate conj/N sweep.
 *
 * vs the retired three-segment wrapper (independent conj sweep + forward
 * kernel + independent conj/N sweep — DELTA F1, kept as the N < 32 host
 * floor and as the fusion comparison baseline): the two extra array
 * sweeps disappear; the caller's buffer is gather-read once (conjugated
 * at the loads) and written once (conj*invN at the final stores).
 *
 * Requires kStages >= 2 (same weldability rule as the conv build; the
 * host layer routes N < 32 to the wrapper floor).  The frame layout is
 * identical to SchedEmitterC (r14/r15 stay pushed though the inverse
 * carries no H tables — shared epilogue math, no correctness impact).
 * ======================================================================== */
class SchedEmitterI final : public Xbyak::CodeGenerator {
public:
    struct Stage {
        int radix;          /* r_t */
        int bprev;          /* B_{t-1} in elements (t=0 -> 1) */
        int b;              /* B_t in elements */
        const double *tw;   /* TW_t base, nullptr for t=0 */
    };

    SchedEmitterI(int n, const uint32_t *gatherTab, const double *scratch,
                  const Stage *stages, int kStages, bool fma,
                  const int *groupLen, bool nt, int prefetch)
        : Xbyak::CodeGenerator(524288), fma_(fma),
          nt_(nt && ((1 << n) >= 4096)), prefetch_(prefetch), n_(n) {
        if (kStages < 2) {
            /* unreachable by construction: the inverse build excludes
             * k < 2 decompositions.  Fail the emission loudly. */
            throw std::bad_alloc();
        }
        int gl[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (groupLen != nullptr) {
            for (int i = 0; i < 8; ++i) {
                gl[i] = groupLen[i];
            }
        }
        bool needSpill = false;
        for (int t = 0; t < kStages; ++t) {
            if (stages[t].radix >= 16) {
                needSpill = true;
            }
        }
        /* spill frame keeps clear of the pushed rbx/r12/r13/r14/r15 slots
         * (r15 sits at [rbp-40]; the 8x32B spill frame starts at rbp-304;
         * frame = 272B from rbp-312 when needSpill) — same as SchedEmitterC. */
        spillOff_ = -304;
        push(rbp);
        mov(rbp, rsp);
        push(rbx);
        push(r12);
        push(r13); /* r13 = stage-0-group gather-tab cursor */
        push(r14); /* parked (no H tables on the inverse half) */
        push(r15); /* parked (no H tables on the inverse half) */
        if (needSpill) {
            sub(rsp, 272);
        }

        const int N = 1 << n;
        /* ---- the single fused inverse half.  rdi = the caller's io on
         * entry; stage 0 gathers io with the ENTRY conj at the load
         * boundary, parks io in rdx, middle stages sweep the scratch,
         * and the final stage writes the natural conj(.) * (1/N) result
         * back through rdx (mode 2). ---- */
        emitHalf(N, gatherTab, scratch, stages, kStages, gl, /*mode=*/2);

        vzeroupper();
        lea(rsp, ptr[rbp - 40]); /* at the pushed r15 slot */
        pop(r15);
        pop(r14);
        pop(r13);
        pop(r12);
        pop(rbx);
        pop(rbp);
        ret();
        emitPool();
    }

private:
    static int lf_log2(int v) {
        int s = 0;
        while ((1 << s) < v) {
            ++s;
        }
        return s;
    }

    /* ---- one transform half (stage 0, middle sweeps, fused final).
     * finalMode_ stays 0 through the gather / middle sweeps; only the
     * final stage runs with the fused mode (2 = conj*invN stores). ---- */
    void emitHalf(int N, const uint32_t *gatherTab, const double *scratch,
                  const Stage *stages, int kStages, const int *groupLen,
                  int mode) {
        finalMode_ = 0;
        int t = 0;
        if (groupLen[0] >= 2) {
            emitStage0Group(N, gatherTab, scratch, stages, groupLen[0]);
            t = groupLen[0];
        } else {
            emitStage0Plain(N, gatherTab, scratch, stages[0]);
            t = 1;
        }
        while (t < kStages - 1) {
            const int d = groupLen[t] >= 2 ? groupLen[t] : 1;
            if (d >= 2) {
                emitTileGroup(stages, t, d, N);
            } else {
                emitSweepFull(stages[t], N, /*allowNt=*/true);
            }
            t += d;
        }
        /* final stage: reads the scratch, writes the natural array with
         * the fused boundary op (conj*invN).  Stores are NEVER
         * non-temporal (the caller reads the result immediately). */
        emitFinalStage(stages[kStages - 1], N, mode);
    }

    /* ---- stage 0, unblocked (R0/R1 shape): digitrev gather fused into
     * the first codelet sweep, out-of-place into the scratch — with the
     * ENTRY conjugate sunk into the gather load boundary ---- */
    void emitStage0Plain(int N, const uint32_t *gatherTab,
                         const double *scratch, const Stage &st) {
        using namespace Xbyak;
        mov(r8, reinterpret_cast<uint64_t>(gatherTab));
        mov(rsi, reinterpret_cast<uint64_t>(scratch));
        lea(rdx, ptr[rsi + N * 16]); /* scratch end */
        ntSweep_ = nt_; /* scratch stores: re-read only by a later full
                           sweep -> NT candidate (R2 F7) */
        Label blk;
        L(blk);
        if (st.radix <= 16) {
            /* vector-2 over block pairs (2b, 2b+1) */
            gatherTab_ = &r8;
            emitLeafV2(st.radix, rsi, 16, rsi, false);
            gatherTab_ = nullptr;
            add(r8, st.radix * 8);   /* 2 table rows (uint32) */
            add(rsi, st.radix * 32); /* 2 blocks */
        } else {
            gatherTab_ = &r8;
            emitLeaf32(r8, 16, false, rsi);
            gatherTab_ = nullptr;
            add(r8, st.radix * 4);
            add(rsi, st.radix * 16);
        }
        cmp(rsi, rdx);
        jb(blk);
        ntSweep_ = false;
        if (nt_) {
            sfence(); /* NT scratch writes precede later loads (F7) */
        }
        /* stages 1..k-2 run in-place on the scratch; park the natural
         * base in rdx (stage-0 loop is done) */
        mov(rdx, rdi);
        mov(rdi, reinterpret_cast<uint64_t>(scratch));
    }

    /* ---- R2 F6: stage-0 tile group.  Per tile: gather exactly the
     * tile's digitrev rows (tab cursor r13 stays sequential across
     * tiles), write the tile in the scratch, then run stages 1..d-1
     * confined to the tile before advancing (intermediate results never
     * leave L1).  The ENTRY conj rides the gather loads. ---- */
    void emitStage0Group(int N, const uint32_t *gatherTab,
                         const double *scratch, const Stage *stages,
                         int d) {
        using namespace Xbyak;
        const Stage &st = stages[0];
        const int T = stages[d - 1].b; /* tile elements */
        bool ntUsed = false;
        mov(r13, reinterpret_cast<uint64_t>(gatherTab)); /* tab cursor */
        mov(r12, reinterpret_cast<uint64_t>(scratch));   /* tile cursor */
        lea(rcx, ptr[r12 + N * 16]); /* array end (tile-loop bound) */
        Label tile;
        L(tile);
        emitPrefetchHead(r12, T * 16);
        mov(rsi, r12);
        lea(rdx, ptr[r12 + T * 16]); /* this tile's scratch end */
        Label blk;
        L(blk);
        if (st.radix <= 16) {
            gatherTab_ = &r13;
            emitLeafV2(st.radix, rsi, 16, rsi, false);
            gatherTab_ = nullptr;
            add(r13, st.radix * 8);  /* 2 table rows (uint32) */
            add(rsi, st.radix * 32); /* 2 blocks */
        } else {
            gatherTab_ = &r13;
            emitLeaf32(r13, 16, false, rsi);
            gatherTab_ = nullptr;
            add(r13, st.radix * 4);
            add(rsi, st.radix * 16);
        }
        cmp(rsi, rdx);
        jb(blk);
        for (int s = 1; s < d; ++s) {
            const bool last = (s == d - 1);
            emitSweepInTile(stages[s], T * 16, last);
            ntUsed |= (nt_ && last);
        }
        add(r12, T * 16);
        cmp(r12, rcx);
        jb(tile);
        if (ntUsed) {
            sfence(); /* last-in-group NT writes precede later loads */
        }
        /* park the natural base; the scratch becomes the working array */
        mov(rdx, rdi);
        mov(rdi, reinterpret_cast<uint64_t>(scratch));
    }

    /* ---- one stage's sweep confined to the current tile ([r12,
     * r12+tileBytes)); in-tile stores stay WB (they are re-read from L1
     * inside the tile) EXCEPT the last stage of the group (its output is
     * consumed only by a later full sweep -> NT candidate). ---- */
    void emitSweepInTile(const Stage &st, int tileBytes, bool lastOfGroup) {
        using namespace Xbyak;
        ntSweep_ = nt_ && lastOfGroup;
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, r12);
        lea(r8, ptr[r12 + tileBytes]);
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 16]); /* k-loop end */
        mov(rax, rsi);
        mov(r10, r9);
        Label kLoop;
        L(kLoop);
        if (st.radix <= 16) {
            /* vector-2 over k pairs (k, k+1) */
            emitLeafV2(st.radix, rax, st.bprev * 16, rax, true);
            add(rax, 32);
            add(r10, st.radix * 64);
        } else {
            emitLeaf32(rax, st.bprev * 16, true, rax);
            add(rax, 16);
            add(r10, 32 * 32);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 16);
        cmp(rsi, r8);
        jb(blk);
        ntSweep_ = false;
    }

    /* ---- one stage's unblocked full-array sweep (R0/R1 shape); NT
     * stores + trailing sfence when enabled (output re-read only by a
     * later full sweep). ---- */
    void emitSweepFull(const Stage &st, int N, bool allowNt) {
        using namespace Xbyak;
        ntSweep_ = nt_ && allowNt;
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, rdi);
        lea(r8, ptr[rdi + N * 16]); /* end (r8 table dead) */
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 16]); /* k-loop end */
        mov(rax, rsi);
        mov(r10, r9);
        Label kLoop;
        L(kLoop);
        if (st.radix <= 16) {
            /* vector-2 over k pairs (k, k+1) */
            emitLeafV2(st.radix, rax, st.bprev * 16, rax, true);
            add(rax, 32);
            add(r10, st.radix * 64);
        } else {
            emitLeaf32(rax, st.bprev * 16, true, rax);
            add(rax, 16);
            add(r10, 32 * 32);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 16);
        cmp(rsi, r8);
        jb(blk);
        ntSweep_ = false;
        if (nt_ && allowNt) {
            sfence();
        }
    }

    /* ---- R2 F6: middle tile group (stages t..t+d-1, t >= 1) ---- */
    void emitTileGroup(const Stage *stages, int t, int d, int N) {
        using namespace Xbyak;
        const int T = stages[t + d - 1].b; /* tile elements */
        bool ntUsed = false;
        mov(r12, rdi); /* tile cursor = scratch base */
        lea(rcx, ptr[rdi + N * 16]);
        Label tile;
        L(tile);
        emitPrefetchHead(r12, T * 16);
        for (int s = t; s < t + d; ++s) {
            const bool last = (s == t + d - 1);
            emitSweepInTile(stages[s], T * 16, last);
            ntUsed |= (nt_ && last);
        }
        add(r12, T * 16);
        cmp(r12, rcx);
        jb(tile);
        if (ntUsed) {
            sfence(); /* last-in-group NT writes precede later loads */
        }
    }

    /* ---- R2 F7: prefetcht0 the head (8 cache lines) of the NEXT tile
     * at every tile-loop boundary (knob-gated; prefetch of an address
     * past the array end is architecturally harmless). ---- */
    void emitPrefetchHead(const Xbyak::Reg64 &base, int tileBytes) {
        if (prefetch_ <= 0) {
            return;
        }
        for (int i = 0; i < 8; ++i) {
            prefetcht0(ptr[base + tileBytes + i * 64]);
        }
    }

    /* ---- final stage: reads the scratch, writes the natural array with
     * the fused boundary op (mode 2 = conj*invN). ---- */
    void emitFinalStage(const Stage &st, int N, int mode) {
        using namespace Xbyak;
        finalMode_ = mode;
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, rdi);
        lea(r8, ptr[rdi + N * 16]); /* end (r8 table dead) */
        mov(r12, rdx);             /* natural block cursor */
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 16]); /* k-loop end */
        mov(rax, rsi);
        mov(r10, r9);
        Label kLoop;
        L(kLoop);
        if (st.radix <= 16) {
            /* vector-2 over k pairs (k, k+1) */
            mov(rbx, r12);
            emitLeafV2(st.radix, rax, st.bprev * 16, rbx, true);
            add(rax, 32);
            add(r10, st.radix * 64);
            add(r12, 32);
        } else {
            mov(rbx, r12);
            emitLeaf32(rax, st.bprev * 16, true, rbx);
            add(rax, 16);
            add(r10, 32 * 32);
            add(r12, 16);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 16);
        add(r12, st.b * 16);
        cmp(rsi, r8);
        jb(blk);
        finalMode_ = 0;
    }

    /* ---- topic jit-fp64-inverse: fused final-stage boundary op, in
     * place on a (k, k+1) natural pair chunk: v <- conj(v) * (1/N) via
     * ONE vmulpd against the (invN, -invN, invN, -invN) rip constant
     * (the conv weld half-B idiom: the sign-baked constant conjugates
     * and scales in a single multiply). ---- */
    void fuseFinalPairInPlace(const Xbyak::Ymm &v, int /*off*/) {
        using namespace Xbyak;
        if (finalMode_ == 2) {
            vmulpd(v, v, ptr[rip + invNLbl_]);
        }
    }

    /* ---- vector-2 leaf dispatch (radix 2/4/8/16) ----
     * Loads come from loadBase + slot*strideBytes (twiddled stages) or
     * through the gatherTab_ cursor (stage 0); stores go to storeBase +
     * pos*strideBytes (twiddled) or storeBase + pos*16 / (r+pos)*16 for
     * the two gathered blocks (stage 0, contiguous scratch).  Twiddle
     * rows (hasTw) come from r10: 64B slots per (k-pair, j). */
    void emitLeafV2(int r, const Xbyak::Reg64 &loadBase, int strideBytes,
                    const Xbyak::Reg64 &storeBase, bool hasTw) {
        using namespace Xbyak;
        if (r >= 4 || (!fma_ && hasTw)) {
            vmovapd(ymm14, ptr[rip + sign2Lbl_]);
        }
        if (r == 16) {
            emitV2Sub(8, loadBase, strideBytes, hasTw, 0);
            for (int j = 0; j < 8; ++j) {
                vmovupd(ptr[rbp + spillOff_ + j * 32], Ymm(j));
            }
            emitV2Sub(8, loadBase, strideBytes, hasTw, 1);
            cross16V2(storeBase, strideBytes);
        } else {
            emitV2Sub(r, loadBase, strideBytes, hasTw, -1);
            storeV2(r, storeBase, strideBytes);
        }
    }

    /* r-point DFT in per-position vector-2 registers P[0..r-1] (r <= 8).
     * par < 0: whole leaf (slot(pos) = bitrev(pos, log2 r));
     * par >= 0: sub-leaf of the radix-16 leaf (slot(pos) =
     * 2*bitrev(pos, 3) + par; gather rows stride 16). */
    void emitV2Sub(int r, const Xbyak::Reg64 &loadBase, int strideBytes,
                   bool hasTw, int par) {
        using namespace Xbyak;
        const int nlf = lf_log2(r);
        const int rr = (par >= 0) ? 16 : r; /* gather row stride (r1) */
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};

        /* ---- load (+twiddle) + m=2 fuse ---- */
        for (int u = 0; u < r / 2; ++u) {
            int sa, sb;
            if (par >= 0) {
                sa = 2 * (int)bitrev((uint32_t)(2 * u), nlf) + par;
                sb = 2 * (int)bitrev((uint32_t)(2 * u + 1), nlf) + par;
            } else {
                sa = (int)bitrev((uint32_t)(2 * u), nlf);
                sb = (int)bitrev((uint32_t)(2 * u + 1), nlf);
            }
            if (gatherTab_ != nullptr) {
                const Reg64 &tab = *gatherTab_;
                mov(eax, dword[tab + sa * 4]);
                vmovupd(xmm8, ptr[rdi + rax]);
                mov(eax, dword[tab + (rr + sa) * 4]);
                vmovupd(xmm9, ptr[rdi + rax]);
                mov(eax, dword[tab + sb * 4]);
                vmovupd(xmm10, ptr[rdi + rax]);
                mov(eax, dword[tab + (rr + sb) * 4]);
                vmovupd(xmm11, ptr[rdi + rax]);
                /* xmm14 holds the sign mask (ymm14) — the second
                 * block's fuse reuses the now-dead load regs */
                vaddpd(xmm12, xmm8, xmm10);
                vsubpd(xmm13, xmm8, xmm10);
                vaddpd(xmm8, xmm9, xmm11);
                vsubpd(xmm10, xmm9, xmm11);
                vinsertf128(P[2 * u], P[2 * u], xmm12, 0);
                vinsertf128(P[2 * u], P[2 * u], xmm8, 1);
                vinsertf128(P[2 * u + 1], P[2 * u + 1], xmm13, 0);
                vinsertf128(P[2 * u + 1], P[2 * u + 1], xmm10, 1);
                /* topic jit-fp64-inverse: ENTRY conj sunk into the
                 * gather load boundary.  The m=2 fuse above is
                 * twiddle-free (+/- only), so conjugating the pair
                 * register right after the fuse is bit-exact
                 * conjugating the four gathered elements (one vxorpd
                 * per pair ymm — the rip sign mask flips the imaginary
                 * lane of both complex lanes).  The butterflies below
                 * then run on DFT_r(conj(x_block)) inputs, exactly the
                 * stage-0 of F(conj(x)).  The gather branch is only
                 * ever entered un-twiddled (stage-0 tw = nullptr). */
                vxorpd(P[2 * u], P[2 * u], ptr[rip + sign2Lbl_]);
                vxorpd(P[2 * u + 1], P[2 * u + 1],
                       ptr[rip + sign2Lbl_]);
            } else {
                loadTwV2(ymm10, ptr[loadBase + sa * strideBytes], sa, hasTw);
                loadTwV2(ymm11, ptr[loadBase + sb * strideBytes], sb, hasTw);
                vaddpd(P[2 * u], ymm10, ymm11);
                vsubpd(P[2 * u + 1], ymm10, ymm11);
            }
        }

        /* ---- stages 2..nlf (butterflies pair positions (i, i+h)) ---- */
        for (int st = 2; st <= nlf; ++st) {
            const int m = 1 << st;
            const int h = m >> 1;
            for (int blk = 0; blk < r / m; ++blk) {
                for (int i = 0; i < h; ++i) {
                    const Ymm &bot = P[blk * m + i];
                    const Ymm &top = P[blk * m + h + i];
                    if (i == 0) {
                        /* T = top in place: diff via scratch */
                        vsubpd(ymm12, bot, top);
                        vaddpd(bot, bot, top);
                        vmovapd(top, ymm12);
                    } else if (i == (m >> 2)) {
                        /* W_m^{m/4} = -i: swap + sign flip */
                        vshufpd(ymm12, top, top, 0x5);
                        vxorpd(ymm12, ymm12, ymm14);
                        vsubpd(top, bot, ymm12);
                        vaddpd(bot, bot, ymm12);
                    } else {
                        mulTwPoolV2(top, st, i);
                        vsubpd(top, bot, ymm12);
                        vaddpd(bot, bot, ymm12);
                    }
                }
            }
        }
    }

    /* load the (k, k+1) pair at addr; twiddle by TW row slot (64B slots
     * at r10 + slot*64) when slot != 0; result in dst (ymm10/ymm11). */
    void loadTwV2(const Xbyak::Ymm &dst, const Xbyak::Operand &addr,
                  int slot, bool hasTw) {
        using namespace Xbyak;
        if (slot == 0 || !hasTw) {
            vmovupd(dst, addr);
            return;
        }
        vmovupd(ymm8, addr);
        if (fma_) {
            vmovddup(ymm9, ymm8);            /* (ar, ar) per lane */
            vshufpd(dst, ymm8, ymm8, 0xF);   /* (ai, ai) per lane */
            vmulpd(dst, dst, ptr[r10 + slot * 64 + 32]);
            vfmaddsub231pd(dst, ymm9, ptr[r10 + slot * 64]);
        } else {
            vmovupd(ymm12, ptr[r10 + slot * 64]); /* W pair */
            vshufpd(ymm13, ymm12, ymm12, 0x5);
            vxorpd(ymm13, ymm13, ymm14);          /* (wi, -wr) */
            vmulpd(dst, ymm8, ymm12);
            vmulpd(ymm13, ymm8, ymm13);
            vhsubpd(dst, dst, ymm13);
        }
    }

    /* T (ymm12) = top * W_{2^st}^e from the rip pool (broadcast const +
     * pre-swapped twin as memory operands; non-FMA derives the conjugate
     * in registers). */
    void mulTwPoolV2(const Xbyak::Ymm &top, int st, int e) {
        using namespace Xbyak;
        if (fma_) {
            vmovddup(ymm13, top);
            vshufpd(ymm12, top, top, 0xF);
            vmulpd(ymm12, ymm12, ptr[rip + twSwapLbl_[st][e]]);
            vfmaddsub231pd(ymm12, ymm13, ptr[rip + twLbl_[st][e]]);
        } else {
            vmovapd(ymm13, ptr[rip + twLbl_[st][e]]);
            vshufpd(ymm15, ymm13, ymm13, 0x5);
            vxorpd(ymm15, ymm15, ymm14);
            vmulpd(ymm12, top, ymm13);
            vmulpd(ymm15, top, ymm15);
            vhsubpd(ymm12, ymm12, ymm15);
        }
    }

    /* radix-16 cross: y[j] = e[j] + W_16^j o[j], y[j+8] = e[j] - W_16^j o[j]
     * (j = 0..7).  e[] reloaded from the spill frame, o[] in P. */
    void cross16V2(const Xbyak::Reg64 &storeBase, int strideBytes) {
        using namespace Xbyak;
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};
        for (int j = 0; j < 8; ++j) {
            vmovupd(ymm8, ptr[rbp + spillOff_ + j * 32]); /* e[j] */
            if (j == 0) {
                vaddpd(ymm10, ymm8, P[0]);
                vsubpd(ymm11, ymm8, P[0]);
            } else if (j == 4) {
                /* W_16^4 = -i */
                vshufpd(ymm12, P[4], P[4], 0x5);
                vxorpd(ymm12, ymm12, ymm14);
                vaddpd(ymm10, ymm8, ymm12);
                vsubpd(ymm11, ymm8, ymm12);
            } else {
                mulTwPoolV2(P[j], 4, j); /* W_16^j = W_{2^4}^j */
                vaddpd(ymm10, ymm8, ymm12);
                vsubpd(ymm11, ymm8, ymm12);
            }
            storeV2One(ymm10, j, storeBase, strideBytes, 16);
            storeV2One(ymm11, j + 8, storeBase, strideBytes, 16);
        }
    }

    void storeV2(int r, const Xbyak::Reg64 &storeBase, int strideBytes) {
        using namespace Xbyak;
        for (int p = 0; p < r; ++p) {
            storeV2One(Ymm(p), p, storeBase, strideBytes, r);
        }
    }

    /* store position pos: whole 32B ym at storeBase + pos*stride (normal
     * / final stages), or per-block 16B extracts into the two gathered
     * blocks (stage 0: storeBase + pos*16 and storeBase + (r+pos)*16).
     * FUSED final stage (finalMode_ != 0): the boundary op runs on the
     * ym first (conj*invN — see fuseFinalPairInPlace).  R2 F7: when the
     * sweep is NT-eligible the stores are vmovntpd (non-temporal; the
     * sfence lives at the sweep/tile boundary). */
    void storeV2One(const Xbyak::Ymm &v, int pos,
                    const Xbyak::Reg64 &storeBase, int strideBytes, int r) {
        using namespace Xbyak;
        if (gatherTab_ != nullptr) {
            vextractf128(xmm12, v, 0);
            if (ntSweep_) {
                vmovntpd(ptr[storeBase + pos * 16], xmm12);
            } else {
                vmovupd(ptr[storeBase + pos * 16], xmm12);
            }
            vextractf128(xmm12, v, 1);
            if (ntSweep_) {
                vmovntpd(ptr[storeBase + (r + pos) * 16], xmm12);
            } else {
                vmovupd(ptr[storeBase + (r + pos) * 16], xmm12);
            }
            return;
        }
        const int off = pos * strideBytes;
        if (finalMode_ != 0) {
            fuseFinalPairInPlace(v, off);
            vmovupd(ptr[storeBase + off], v);
            return;
        }
        if (ntSweep_) {
            vmovntpd(ptr[storeBase + off], v);
        } else {
            vmovupd(ptr[storeBase + off], v);
        }
    }

    /* ---- single-k radix-32 leaf (pair-ymm lanes = adjacent DIT
     * positions of one k; vector-2 would need 32 live yms) ---- */
    void emitLeaf32(const Xbyak::Reg64 &loadBase, int strideBytes,
                    bool hasTw, const Xbyak::Reg64 &storeBase) {
        using namespace Xbyak;
        vmovapd(ymm14, ptr[rip + sign2Lbl_]);
        emitSubLeaf16Sk(loadBase, strideBytes, hasTw, 0);
        for (int p = 0; p < 8; ++p) {
            vmovupd(ptr[rbp + spillOff_ + p * 32], Ymm(p));
        }
        emitSubLeaf16Sk(loadBase, strideBytes, hasTw, 1);
        /* cross: y[k] = e[k] + W_32^k o[k], pair chunks (W^{2p}, W^{2p+1})
         * from the rip pool (mixing foldable and general lanes — the
         * chunked multiply stays; e-pair reloaded from the spill frame) */
        for (int p = 0; p < 8; ++p) {
            vmovupd(ymm9, ptr[rbp + spillOff_ + p * 32]); /* A pair */
            if (fma_) {
                vmovddup(ymm11, Ymm(p));
                vshufpd(ymm8, Ymm(p), Ymm(p), 0xF);
                vmulpd(ymm8, ymm8, ptr[rip + w32SwapLbl_[p]]);
                vfmaddsub231pd(ymm8, ymm11, ptr[rip + w32Lbl_[p]]);
            } else {
                vmovupd(ymm10, ptr[rip + w32Lbl_[p]]);
                vshufpd(ymm12, ymm10, ymm10, 0x5);
                vxorpd(ymm12, ymm12, ymm14);
                vmulpd(ymm8, Ymm(p), ymm10);
                vmulpd(ymm12, Ymm(p), ymm12);
                vhsubpd(ymm8, ymm8, ymm12);
            }
            vaddpd(ymm13, ymm9, ymm8);  /* y[k]    -> slots (2p, 2p+1) */
            vsubpd(ymm15, ymm9, ymm8);  /* y[k+16] -> (2p+16, 2p+17);
                                           ymm14 = sgn must stay live */
            if (strideBytes == 16) {
                /* stage-0 gather (the N==32 single-codelet transform is
                 * excluded from the inverse build by construction) */
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p) * 16], ymm13);
                    vmovntpd(ptr[storeBase + (2 * p + 16) * 16], ymm15);
                } else {
                    vmovupd(ptr[storeBase + (2 * p) * 16], ymm13);
                    vmovupd(ptr[storeBase + (2 * p + 16) * 16], ymm15);
                }
            } else {
                if (finalMode_ != 0) {
                    /* fused final: boundary op on the pair chunks BEFORE
                     * the extracts */
                    fuseFinalPairInPlace(ymm13, (2 * p) * strideBytes);
                    fuseFinalPairInPlace(ymm15, (2 * p + 16) * strideBytes);
                }
                vextractf128(xmm8, ymm13, 0);
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p) * strideBytes], xmm8);
                } else {
                    vmovupd(ptr[storeBase + (2 * p) * strideBytes], xmm8);
                }
                vextractf128(xmm8, ymm13, 1);
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p + 1) * strideBytes],
                             xmm8);
                } else {
                    vmovupd(ptr[storeBase + (2 * p + 1) * strideBytes],
                            xmm8);
                }
                vextractf128(xmm8, ymm15, 0);
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p + 16) * strideBytes],
                             xmm8);
                } else {
                    vmovupd(ptr[storeBase + (2 * p + 16) * strideBytes],
                            xmm8);
                }
                vextractf128(xmm8, ymm15, 1);
                if (ntSweep_) {
                    vmovntpd(ptr[storeBase + (2 * p + 17) * strideBytes],
                             xmm8);
                } else {
                    vmovupd(ptr[storeBase + (2 * p + 17) * strideBytes],
                            xmm8);
                }
            }
        }
    }

    /* DFT-16 sub-leaf of the radix-32 leaf: input slot =
     * 2*bitrev4(pos) + par, result left in ymm0..7 (pair layout). */
    void emitSubLeaf16Sk(const Xbyak::Reg64 &loadBase, int strideBytes,
                         bool hasTw, int par) {
        using namespace Xbyak;
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};

        for (int p = 0; p < 8; ++p) {
            const int sa = 2 * (int)bitrev((uint32_t)(2 * p), 4) + par;
            const int sb = 2 * (int)bitrev((uint32_t)(2 * p + 1), 4) + par;
            if (gatherTab_ != nullptr) {
                const Reg64 &tab = *gatherTab_;
                mov(eax, dword[tab + sa * 4]);
                vmovupd(xmm8, ptr[rdi + rax]);
                mov(eax, dword[tab + sb * 4]);
                vmovupd(xmm9, ptr[rdi + rax]);
            } else {
                vmovupd(xmm8, ptr[loadBase + sa * strideBytes]);
                vmovupd(xmm9, ptr[loadBase + sb * strideBytes]);
            }
            if (hasTw && sa != 0) {
                mulTwXmm(xmm10, xmm8, sa);
            }
            if (hasTw && sb != 0) {
                mulTwXmm(xmm12, xmm9, sb);
            }
            const Xmm &a2 = (hasTw && sa != 0) ? xmm10 : xmm8;
            const Xmm &b2 = (hasTw && sb != 0) ? xmm12 : xmm9;
            /* xmm14 reserved for the sign mask (ymm14) */
            vaddpd(xmm13, a2, b2);
            vsubpd(xmm15, a2, b2);
            vinsertf128(P[p], P[p], xmm13, 0);
            vinsertf128(P[p], P[p], xmm15, 1);
            if (gatherTab_ != nullptr && !hasTw) {
                /* topic jit-fp64-inverse: ENTRY conj at the gather load
                 * boundary — same twiddle-free linearity argument as the
                 * vector-2 leaf (the stage-0 single-k gather is always
                 * un-twiddled: stages[0].tw = nullptr). */
                vxorpd(P[p], P[p], ptr[rip + sign2Lbl_]);
            }
        }

        /* stage 2 (m=4): top HIGH lane x (-i) blended into the scratch */
        for (int blk = 0; blk < 4; ++blk) {
            const Ymm &bot = P[2 * blk];
            const Ymm &top = P[2 * blk + 1];
            vshufpd(ymm8, top, top, 0x5);
            vxorpd(ymm8, ymm8, ymm14);
            vblendpd(ymm8, top, ymm8, 0x0C);
            vsubpd(top, bot, ymm8);
            vaddpd(bot, bot, ymm8);
        }

        /* stages 3..4: pair chunks (W^{2q}, W^{2q+1}) from the rip pool */
        for (int st = 3; st <= 4; ++st) {
            const int halfPairs = 1 << (st - 2);
            const int pairsPerBlock = 1 << (st - 1);
            for (int blk = 0; blk < 16 / (1 << st); ++blk) {
                const int bp = blk * pairsPerBlock;
                for (int q = 0; q < halfPairs; ++q) {
                    const Ymm &bot = P[bp + q];
                    const Ymm &top = P[bp + halfPairs + q];
                    if (fma_) {
                        vmovddup(ymm11, top);
                        vshufpd(ymm8, top, top, 0xF);
                        vmulpd(ymm8, ymm8, ptr[rip + lwSwapLbl_[st][q]]);
                        vfmaddsub231pd(ymm8, ymm11, ptr[rip + lwLbl_[st][q]]);
                    } else {
                        vmovapd(ymm10, ptr[rip + lwLbl_[st][q]]);
                        vshufpd(ymm12, ymm10, ymm10, 0x5);
                        vxorpd(ymm12, ymm12, ymm14);
                        vmulpd(ymm8, top, ymm10);
                        vmulpd(ymm12, top, ymm12);
                        vhsubpd(ymm8, ymm8, ymm12);
                    }
                    vsubpd(top, bot, ymm8);
                    vaddpd(bot, bot, ymm8);
                }
            }
        }
    }

    /* dst (xmm) = d * TW[slot]; 32B slots at r10 + slot*32 =
     * (wr, wi, wi, +wr|FMA / -wr|no-FMA) — both halves are memory
     * operands of the complex multiply. */
    void mulTwXmm(const Xbyak::Xmm &dst, const Xbyak::Xmm &d, int slot) {
        using namespace Xbyak;
        if (fma_) {
            vmovddup(xmm11, d);             /* (dr, dr) */
            vshufpd(dst, d, d, 3);          /* (di, di) */
            vmulpd(dst, dst, ptr[r10 + slot * 32 + 16]);
            vfmaddsub231pd(dst, xmm11, ptr[r10 + slot * 32]);
        } else {
            vmulpd(dst, d, ptr[r10 + slot * 32]);
            vmulpd(xmm11, d, ptr[r10 + slot * 32 + 16]);
            vhsubpd(dst, dst, xmm11);
        }
    }

    /* rip-relative constant pool: sign mask + vector-2 broadcast
     * twiddles (+ swap twins) + single-k pair chunks (+ twins) + W_32
     * cross pairs (+ twins) + the inverse's (invN, -invN) constant.
     * Shared by every leaf of the plan. */
    void emitPool() {
        using namespace Xbyak;
        align(32);
        L(sign2Lbl_);
        dq(0);
        dq(0x8000000000000000ull);
        dq(0);
        dq(0x8000000000000000ull);

        /* topic jit-fp64-inverse: conj(.) * (1/N) in one vmulpd */
        const double inv = 1.0 / (double)(1 << n_);
        align(32);
        L(invNLbl_);
        dq(bits_from_double(inv));
        dq(bits_from_double(-inv));
        dq(bits_from_double(inv));
        dq(bits_from_double(-inv));

        /* W_{2^s}^e for s = 3..4, e = 1..2^{s-1}-1 (e = 0 skipped = 1,
         * e = 2^{s-2} skipped = -i fold); s=4 entries double as the
         * radix-16 cross W_16^j. */
        for (int s = 3; s <= 4; ++s) {
            for (int e = 1; e < (1 << (s - 1)); ++e) {
                if (e == (1 << (s - 2))) {
                    continue;
                }
                const double ang =
                    -2.0 * kPi * (double)e / (double)(1 << s);
                const double wr = std::cos(ang);
                const double wi = std::sin(ang);
                align(32);
                L(twLbl_[s][e]);
                dq(bits_from_double(wr));
                dq(bits_from_double(wi));
                dq(bits_from_double(wr));
                dq(bits_from_double(wi));
                align(32);
                L(twSwapLbl_[s][e]);
                dq(bits_from_double(wi));
                dq(bits_from_double(wr));
                dq(bits_from_double(wi));
                dq(bits_from_double(wr));
            }
        }

        /* single-k pair chunks: chunk q = (W_{2^s}^{2q}, W_{2^s}^{2q+1}) */
        for (int s = 3; s <= 4; ++s) {
            const int halfPairs = 1 << (s - 2);
            for (int q = 0; q < halfPairs; ++q) {
                double c[2], sn[2];
                for (int l = 0; l < 2; ++l) {
                    const int e = 2 * q + l;
                    const double ang =
                        -2.0 * kPi * (double)e / (double)(1 << s);
                    c[l] = std::cos(ang);
                    sn[l] = std::sin(ang);
                }
                align(32);
                L(lwLbl_[s][q]);
                dq(bits_from_double(c[0]));
                dq(bits_from_double(sn[0]));
                dq(bits_from_double(c[1]));
                dq(bits_from_double(sn[1]));
                align(32);
                L(lwSwapLbl_[s][q]);
                dq(bits_from_double(sn[0]));
                dq(bits_from_double(c[0]));
                dq(bits_from_double(sn[1]));
                dq(bits_from_double(c[1]));
            }
        }

        /* W_32 cross pairs: chunk p = (W_32^{2p}, W_32^{2p+1}) */
        for (int p = 0; p < 8; ++p) {
            double c[2], sn[2];
            for (int l = 0; l < 2; ++l) {
                const int e = 2 * p + l;
                const double ang = -2.0 * kPi * (double)e / 32.0;
                c[l] = std::cos(ang);
                sn[l] = std::sin(ang);
            }
            align(32);
            L(w32Lbl_[p]);
            dq(bits_from_double(c[0]));
            dq(bits_from_double(sn[0]));
            dq(bits_from_double(c[1]));
            dq(bits_from_double(sn[1]));
            align(32);
            L(w32SwapLbl_[p]);
            dq(bits_from_double(sn[0]));
            dq(bits_from_double(c[0]));
            dq(bits_from_double(sn[1]));
            dq(bits_from_double(c[1]));
        }
    }

    static uint64_t bits_from_double(double d) {
        uint64_t u;
        __builtin_memcpy(&u, &d, sizeof(u));
        return u;
    }

    bool fma_;
    bool nt_ = false;         /* R2 F7: NT stores enabled (N >= 4096) */
    int prefetch_ = 0;        /* R2 F7: tile-head prefetcht0 (0 = off) */
    bool ntSweep_ = false;    /* current sweep's stores are NT-eligible */
    int spillOff_ = -304; /* r16/r32 leaf spill base offset from rbp */
    int n_ = 0;               /* log2(N) — invN pool constant */
    int finalMode_ = 0;       /* 0 plain; 2 = store conj(.)*invN (the
                                 fused inverse's only fused mode) */
    const Xbyak::Reg64 *gatherTab_ = nullptr; /* stage-0 table mode */
    Xbyak::Label sign2Lbl_;
    Xbyak::Label invNLbl_;         /* (invN, -invN, invN, -invN) */
    Xbyak::Label twLbl_[5][8];     /* [s][e]: W_{2^s}^e broadcast (v2) */
    Xbyak::Label twSwapLbl_[5][8];
    Xbyak::Label lwLbl_[5][4];     /* [s][q]: pair chunk (single-k) */
    Xbyak::Label lwSwapLbl_[5][4];
    Xbyak::Label w32Lbl_[8];
    Xbyak::Label w32SwapLbl_[8];
};

/* ---- R1 plan-time decomposition search (DELTA F5) + R2 block-strategy
 * dimension (DELTA F6/F7) + shared build ----
 *
 * Light enumerate-and-time (NOT a graph search): candidate decompositions
 * of N into radices {2,4,8,16,32} (ordered — same multiset in a different
 * stage order is a different candidate) are generated at plan time,
 * analytically prefiltered, and each is emitted to an executable buffer
 * and short-timed (min over rounds, same process, same N, same data).
 * R2: each decomposition candidate is additionally expanded over tile
 * BLOCK strategies (maxTile budget in elements; 0 = no blocking), with
 * variants that derive identical tile groupings deduplicated — the
 * candidate space is decompositions x block strategies and the fastest
 * (decomposition, tile) pair is re-emitted as the plan.  Plan-time cost
 * is budget-guarded (~4ms + ~11ms*N/32768; honest search_ns recorded).
 * Env knobs (topic-local debug, no public API):
 *   FFT_SCHED_ORDER=32first|smallfirst|<r1,r2,...>  force decomposition
 *   FFT_SCHED_SEARCH=off|0                          skip search (32-first)
 *   FFT_SCHED_BLOCK=off|auto|<tile elems>           force block strategy
 *   FFT_NT_STORE=on|off|auto                        R2 F7 NT stores
 *   FFT_PREFETCH=on|off|auto                        R2 F7 tile prefetch
 *   FFT_SCHED_LOG=1                                 stderr diagnostics
 */
struct SchedBuilt {
    SchedEmitter *em;
    double *lut;     /* gatherTab32 + TW tables (owned or borrowed) */
    double *scratch; /* stage-0 ping buffer (owned or borrowed) */
    bool ownsScratch;
    bool ownsLut;
};

/* R2 F6: derive the tile groups for one decomposition under a tile
 * budget (maxTile elements; 16B per element).  groupLen[t] = d >= 2
 * means stages t..t+d-1 form one tile group (tile = B[t+d-1] elements);
 * groupLen[t] <= 1 means stage t runs as an unblocked full sweep.  The
 * final stage is never part of a group (its stores are natural-order).
 * Greedy deepest merge: extend a group while the next stage is not the
 * final one and its block prefix still fits the budget.  Returns whether
 * any actual grouping (d >= 2) was produced. */
static bool sched_block_groups(const int *B, int kStages, int maxTile,
                               int *groupLen) {
    for (int i = 0; i < 8; ++i) {
        groupLen[i] = 0;
    }
    if (maxTile <= 0 || B[kStages - 1] <= maxTile) {
        return false; /* whole array fits one tile: blocking is a no-op */
    }
    bool any = false;
    int t = 0;
    while (t < kStages - 1) {
        int d = 1;
        while (t + d <= kStages - 2 && B[t + d] <= maxTile) {
            ++d;
        }
        groupLen[t] = d;
        if (d >= 2) {
            any = true;
        }
        t += d;
    }
    return any;
}

/* groupLen derivation from radices (B computed internally). */
static void sched_groups_of(const int *rad, int k, int maxTile,
                            int *groupLen) {
    int B[8];
    B[0] = rad[0];
    for (int t = 1; t < k; ++t) {
        B[t] = B[t - 1] * rad[t];
    }
    sched_block_groups(B, k, maxTile, groupLen);
}

static double sched_now_ns(void) {
    return std::chrono::duration<double, std::nano>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/* Build the tables + SchedEmitter for one decomposition + block
 * strategy.  borrowScratch != nullptr shares one ping buffer across
 * search candidates; nullptr allocates a fresh one for the final
 * kernel. */
static SchedBuilt schedBuildKernel(int N, int n, const int *radices,
                                   int kStages, bool hasFma,
                                   double *borrowScratch, int maxTile,
                                   bool nt, int prefetch,
                                   double *borrowLut = nullptr) {
    SchedBuilt b = {nullptr, nullptr, nullptr, false, false};

    int B[8];
    B[0] = radices[0];
    for (int t = 1; t < kStages; ++t) {
        B[t] = B[t - 1] * radices[t];
    }

    int groupLen[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    sched_block_groups(B, kStages, maxTile, groupLen);

    /* Inverse digit-reversal table gatherTab[p] = natural byte offset
     * (16*n; p = digitrev(n)); fused into the stage-0 gather. */
    std::vector<uint32_t> gatherTab;
    if (kStages > 1) {
        gatherTab.resize((size_t)N);
        for (int v = 0; v < N; ++v) {
            int e[8];
            int vv = v;
            for (int i = 0; i < kStages; ++i) {
                e[i] = vv % radices[kStages - 1 - i];
                vv /= radices[kStages - 1 - i];
            }
            int p = 0;
            for (int i = 0; i < kStages; ++i) {
                int m = 1;
                for (int l = 0; l < kStages - 1 - i; ++l) {
                    m *= radices[l];
                }
                p += e[i] * m;
            }
            gatherTab[(size_t)p] = (uint32_t)(v * 16);
        }
    }

    /* Per-stage TW tables (trig recurrence: 2 trig per row, then r
     * successive complex multiplies per row — drift <= ~32 ulp, well
     * inside the 1e-12 verification tolerance):
     *   vector-2 (radix<=16): rows = Bp/2 k-pairs, 64B slots
     *     [W^{jk}, W^{j(k+1)} | swap-twin (wi, wr)]
     *   single-k (radix 32): rows = Bp, 32B slots
     *     [W^{jk} | twin (wi, +wr|FMA / -wr|no-FMA)] */
    std::vector<std::vector<double>> twTab(kStages);
    for (int t = 1; t < kStages; ++t) {
        const int r = radices[t];
        const int Bp = B[t - 1];
        const int Bt = B[t];
        const bool v2 = (r <= 16);
        const int rowBytes = v2 ? r * 64 : r * 32;
        twTab[t].resize((size_t)rowBytes / 8 *
                        (size_t)(v2 ? Bp / 2 : Bp));
        const double zr = std::cos(-2.0 * kPi / (double)Bt);
        const double zi = std::sin(-2.0 * kPi / (double)Bt);
        const int rows = v2 ? Bp / 2 : Bp;
        for (int row = 0; row < rows; ++row) {
            const int k = v2 ? 2 * row : row;
            const double ang = -2.0 * kPi * (double)k / (double)Bt;
            const double br = std::cos(ang);
            const double bi = std::sin(ang); /* zeta^k */
            double b2r = 1.0, b2i = 0.0;     /* zeta^{k+1} */
            if (v2) {
                b2r = br * zr - bi * zi;
                b2i = br * zi + bi * zr;
            }
            double wr = 1.0, wi = 0.0;   /* zeta^{jk} */
            double w2r = 1.0, w2i = 0.0; /* zeta^{(k+1)j} */
            double *slot =
                &twTab[t][(size_t)row * (size_t)(rowBytes / 8)];
            for (int j = 0; j < r; ++j) {
                if (v2) {
                    slot[j * 8 + 0] = wr;
                    slot[j * 8 + 1] = wi;
                    slot[j * 8 + 2] = w2r;
                    slot[j * 8 + 3] = w2i;
                    slot[j * 8 + 4] = wi; /* swap twin */
                    slot[j * 8 + 5] = wr;
                    slot[j * 8 + 6] = w2i;
                    slot[j * 8 + 7] = w2r;
                } else {
                    slot[j * 4 + 0] = wr;
                    slot[j * 4 + 1] = wi;
                    slot[j * 4 + 2] = wi;
                    slot[j * 4 + 3] = hasFma ? wr : -wr;
                }
                const double nwr = wr * br - wi * bi;
                const double nwi = wr * bi + wi * br;
                wr = nwr;
                wi = nwi;
                if (v2) {
                    const double n2r = w2r * b2r - w2i * b2i;
                    const double n2i = w2r * b2i + w2i * b2r;
                    w2r = n2r;
                    w2i = n2i;
                }
            }
        }
    }

    /* LUT block: [gatherTab (N*4B, pad32)][TW_1..] */
    const size_t tabBytes = gatherTab.size() * 4;
    size_t cursor = (tabBytes + 31u) & ~31u;
    for (int t = 1; t < kStages; ++t) {
        cursor += twTab[t].size() * sizeof(double);
        cursor = (cursor + 31u) & ~31u;
    }
    if (borrowLut != nullptr) {
        b.lut = borrowLut; /* caller guarantees the size (search arena) */
        b.ownsLut = false;
    } else {
        b.lut = static_cast<double *>(
            aligned_alloc(64, cursor > 64 ? cursor : 64));
        if (b.lut == nullptr) {
            throw std::bad_alloc();
        }
        b.ownsLut = true;
    }
    double *lut = b.lut;
    char *basep = reinterpret_cast<char *>(lut);
    uint32_t *tabPtr = reinterpret_cast<uint32_t *>(basep);
    for (size_t i = 0; i < gatherTab.size(); ++i) {
        tabPtr[i] = gatherTab[i];
    }
    SchedEmitter::Stage stages[8];
    stages[0].radix = radices[0];
    stages[0].bprev = 1;
    stages[0].b = B[0];
    stages[0].tw = nullptr;
    cursor = (tabBytes + 31u) & ~31u;
    for (int t = 1; t < kStages; ++t) {
        double *dst = reinterpret_cast<double *>(basep + cursor);
        __builtin_memcpy(dst, twTab[t].data(),
                         twTab[t].size() * sizeof(double));
        stages[t].radix = radices[t];
        stages[t].bprev = B[t - 1];
        stages[t].b = B[t];
        stages[t].tw = dst;
        cursor += twTab[t].size() * sizeof(double);
        cursor = (cursor + 31u) & ~31u;
    }

    if (borrowScratch != nullptr) {
        b.scratch = borrowScratch;
        b.ownsScratch = false;
    } else if (kStages > 1) {
        b.scratch =
            static_cast<double *>(aligned_alloc(64, (size_t)N * 16));
        if (b.scratch == nullptr) {
            if (b.ownsLut) {
                free(b.lut);
            }
            b.lut = nullptr;
            throw std::bad_alloc();
        }
        b.ownsScratch = true;
    }

    try {
        b.em = new SchedEmitter(n, tabPtr, b.scratch, stages, kStages,
                                hasFma, groupLen, nt, prefetch);
    } catch (...) {
        if (b.ownsLut) {
            free(b.lut);
        }
        b.lut = nullptr;
        if (b.ownsScratch) {
            free(b.scratch);
            b.scratch = nullptr;
        }
        throw;
    }
    return b;
}

struct SchedCand {
    int rad[8];
    int k;
    int tile; /* R2 F6: max tile elements (0 = no blocking) */
    double model;
};

/* Rough per-slot cost model used ONLY to order the candidate pool (the
 * actual choice is measured).  Position-aware: stage 0 pays the digitrev
 * gather (table lookups + inserts), the final stage pays strided
 * natural-order stores, middle stages are the uniform vector-2 sweep.
 * Vector-2 leaves for r <= 8 (no spill), r = 16 (256B spill cross),
 * radix-32 single-k (no vector-2 possible). */
static double sched_slot_model(int radix, int pos, int k) {
    bool stage0 = (pos == 0);
    bool final = (pos == k - 1) && !stage0;
    switch (radix) {
    case 2:
        return stage0 ? 5.6 : final ? 5.8 : 5.4;
    case 4:
        return stage0 ? 6.0 : final ? 6.3 : 5.8;
    case 8:
        return stage0 ? 6.4 : final ? 6.8 : 6.0;
    case 16:
        return stage0 ? 10.0 : final ? 9.5 : 7.8;
    default: /* 32, single-k */
        return stage0 ? 11.5 : final ? 15.5 : 15.0;
    }
}

static double sched_model(const int *rad, int k) {
    double s = 0.0;
    for (int t = 0; t < k; ++t) {
        s += sched_slot_model(rad[t], t, k);
    }
    return s;
}

/* LUT bytes a decomposition needs: [gatherTab (N*4, pad32)][TW_t]. */
static size_t sched_lut_bytes(int N, const int *radices, int kStages) {
    int B[8];
    B[0] = radices[0];
    for (int t = 1; t < kStages; ++t) {
        B[t] = B[t - 1] * radices[t];
    }
    const size_t tabBytes = (kStages > 1)
                                ? (((size_t)N * 4 + 31u) & ~31u)
                                : 0;
    size_t cursor = tabBytes;
    for (int t = 1; t < kStages; ++t) {
        cursor += (size_t)B[t] * 32; /* Bp*r*32 == B_t*32 either layout */
        cursor = (cursor + 31u) & ~31u;
    }
    return cursor + 64;
}

static void sched_gen(int rem, int *cur, int len,
                      std::vector<SchedCand> &out) {
    if (rem == 0) {
        SchedCand c;
        for (int i = 0; i < len; ++i) {
            c.rad[i] = cur[i];
        }
        c.k = len;
        c.model = 0.0;
        out.push_back(c);
        return;
    }
    if (len >= 8) {
        return;
    }
    const int hi = rem < 5 ? rem : 5;
    for (int p = 1; p <= hi; ++p) {
        cur[len] = 1 << p;
        sched_gen(rem - p, cur, len + 1, out);
    }
}

static bool sched_same(const SchedCand &a, const SchedCand &b) {
    if (a.k != b.k) {
        return false;
    }
    for (int i = 0; i < a.k; ++i) {
        if (a.rad[i] != b.rad[i]) {
            return false;
        }
    }
    return true;
}

static void sched_greedy32(int n, int *rad, int *kOut) {
    int rem = n;
    int k = 0;
    while (rem > 0) {
        const int r = (rem >= 5) ? 32 : (1 << rem);
        rad[k++] = r;
        rem -= (r == 32) ? 5 : rem;
    }
    *kOut = k;
}

/* Enumerate + time candidate (decomposition x block strategy) pairs for
 * N; on success fills radices/kStages/tileOut with the measured-fastest
 * candidate.  forcedTile >= 0 pins the block dimension (only that tile
 * budget is generated per decomposition); forcedTile < 0 expands each
 * decomposition over {2048, 0, 1024} element budgets with identical-
 * grouping variants deduplicated (R2 F6 joint search). */
static bool sched_search(int N, int n, bool hasFma, bool nt, int prefetch,
                         int forcedTile, int *radices, int *kStagesOut,
                         int *tileOut, double *searchNsOut, int *timedOut,
                         int *poolOut, int *genOut) {
    std::vector<SchedCand> all;
    int cur[8];
    sched_gen(n, cur, 0, all);
    *genOut = (int)all.size();
    for (auto &c : all) {
        c.model = sched_model(c.rad, c.k);
    }
    std::stable_sort(all.begin(), all.end(),
                     [](const SchedCand &a, const SchedCand &b) {
                         return a.model < b.model;
                     });

    /* anchors first (R0 default + its reverse + greedy 16-first), then
     * the best-model candidates, capped well past the budget so the
     * budget guard does the final pruning by measurement */
    std::vector<SchedCand> base;
    SchedCand anc[3];
    sched_greedy32(n, anc[0].rad, &anc[0].k);
    anc[1] = anc[0];
    for (int i = 0; i < anc[1].k; ++i) {
        anc[1].rad[i] = anc[0].rad[anc[1].k - 1 - i];
    }
    {
        int rem = n;
        int k = 0;
        while (rem > 0) {
            const int r = (rem >= 4) ? 16 : (1 << rem);
            anc[2].rad[k++] = r;
            rem -= (r == 16) ? 4 : rem;
        }
        anc[2].k = k;
    }
    for (int a = 0; a < 3; ++a) {
        bool dup = false;
        for (const auto &c : base) {
            if (sched_same(c, anc[a])) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            base.push_back(anc[a]);
        }
    }
    for (const auto &c : all) {
        if (base.size() >= 3 + 40) {
            break;
        }
        bool dup = false;
        for (const auto &p : base) {
            if (sched_same(p, c)) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            base.push_back(c);
        }
    }

    /* R2 F6: expand the decomposition pool over tile budgets (elements;
     * 2048/1024/512 = 32/16/8KB data footprint, inside the L1d-derived
     * 24-32KB guidance).  Budget order puts the 32KB deep merge first
     * (the structure the R1 evidence pointed at: more sweeps with
     * L1-local working sets), then the shallower merges (the 8KB tile
     * measured fastest at N=4096-8192), then off (R1 shape).  Variants
     * that derive the SAME tile grouping (including "whole array fits
     * one tile" == off) are deduplicated so the budget guard times
     * distinct schedules only. */
    const int tileModes[4] = {2048, 512, 1024, 0};
    std::vector<SchedCand> pool;
    for (const auto &b : base) {
        if (forcedTile >= 0) {
            SchedCand c = b;
            c.tile = forcedTile;
            pool.push_back(c);
            continue;
        }
        int seen[4][8];
        int nSeen = 0;
        for (int m = 0; m < 4; ++m) {
            int gl[8];
            sched_groups_of(b.rad, b.k, tileModes[m], gl);
            bool dup = false;
            for (int p = 0; p < nSeen; ++p) {
                if (memcmp(seen[p], gl, sizeof(gl)) == 0) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            memcpy(seen[nSeen++], gl, sizeof(gl));
            SchedCand c = b;
            c.tile = tileModes[m];
            pool.push_back(c);
        }
    }
    *poolOut = (int)pool.size();

    double *io = static_cast<double *>(
        aligned_alloc(64, weld_pad64((size_t)N * 16)));
    double *master = static_cast<double *>(
        aligned_alloc(64, weld_pad64((size_t)N * 16)));
    double *scratch =
        static_cast<double *>(aligned_alloc(64, (size_t)N * 16));
    if (io == nullptr || master == nullptr || scratch == nullptr) {
        free(io);
        free(master);
        free(scratch);
        return false;
    }
    {
        uint32_t st = 0x5eed1234u ^ (uint32_t)N;
        for (int i = 0; i < 2 * N; ++i) {
            st = st * 1664525u + 1013904223u;
            master[i] = (double)(st >> 11) / 2097152.0 - 0.5;
        }
    }
    /* pre-fault the shared buffers so the FIRST timed candidate does not
     * pay the first-touch page faults */
    memset(io, 0, (size_t)N * 16);
    memset(scratch, 0, (size_t)N * 16);

    /* one LUT arena sized for the largest candidate (avoids per-candidate
     * alloc/free page-fault churn) */
    size_t arenaSz = 0;
    for (const auto &c : pool) {
        const size_t need = sched_lut_bytes(N, c.rad, c.k);
        if (need > arenaSz) {
            arenaSz = need;
        }
    }
    double *arena = static_cast<double *>(
        aligned_alloc(64, weld_pad64(arenaSz ? arenaSz : 64)));
    if (arena == nullptr) {
        free(io);
        free(master);
        free(scratch);
        return false;
    }
    memset(arena, 0, arenaSz);

    const long iters =
        N <= 512 ? 40 : N <= 2048 ? 16 : N <= 8192 ? 6 : 3;
    /* R2: budget relaxed from 4+8ms*N/32768 to 4+11ms*N/32768 (15ms at
     * 32768) to cover the decomposition x block-strategy expansion;
     * honest search_ns is recorded by the caller (binder allows ~15ms) */
    const double budgetNs = 4.0e6 + 11.0e6 * ((double)N / 32768.0);
    const double t0 = sched_now_ns();
    double bestNs = 1e30;
    int bestIdx = -1;
    int timed = 0;
    for (size_t ci = 0; ci < pool.size(); ++ci) {
        if (bestIdx >= 0 && timed >= 3 &&
            (sched_now_ns() - t0) > budgetNs) {
            break;
        }
        SchedBuilt b;
        bool ok = true;
        try {
            b = schedBuildKernel(N, n, pool[ci].rad, pool[ci].k, hasFma,
                                 scratch, pool[ci].tile, nt, prefetch,
                                 arena);
        } catch (...) {
            ok = false;
        }
        if (!ok || b.em == nullptr) {
            continue;
        }
        b.em->readyRE();
        fft_jit_fn_t fn = reinterpret_cast<fft_jit_fn_t>(
            const_cast<uint8_t *>(b.em->getCode()));
        memcpy(io, master, (size_t)N * 16);
        fn(io);
        fn(io); /* warm */
        double bestC = 1e30;
        for (int rd = 0; rd < 3; ++rd) { /* min over rounds (turbo) */
            const double s = sched_now_ns();
            for (long i = 0; i < iters; ++i) {
                fn(io);
            }
            const double per = (sched_now_ns() - s) / (double)iters;
            if (per < bestC) {
                bestC = per;
            }
        }
        if (bestC < bestNs) {
            bestNs = bestC;
            bestIdx = (int)ci;
        }
        ++timed;
        delete b.em; /* lut + scratch borrowed */
    }
    *searchNsOut = sched_now_ns() - t0;
    *timedOut = timed;
    free(arena);
    free(io);
    free(master);
    free(scratch);
    if (bestIdx < 0) {
        return false;
    }
    for (int i = 0; i < pool[bestIdx].k; ++i) {
        radices[i] = pool[bestIdx].rad[i];
    }
    *kStagesOut = pool[bestIdx].k;
    *tileOut = pool[bestIdx].tile;
    return true;
}

/* ---- topic jit-fp64-conv-fuse: fused conv weld build (L-I) ----
 *
 * Tables + welded SchedEmitterC for one decomposition.  hMul / hSwap are
 * CALLER-OWNED plan-time H tables in their own aligned allocations —
 * separate tables from this LUT block's gather + TW content (single
 * twiddle LUT discipline, [[ARCH-FFT-016]]; MUST NOT build a fourth
 * twiddle subsystem).  Table materialization (gatherTab + TW rows) is a
 * copy-then-edit of schedBuildKernel above, which stays untouched (the
 * fp64 forward path is regression-only). */
struct SchedConvBuilt {
    SchedEmitterC *em;
    double *lut;     /* gatherTab32 + TW tables (owned) */
    double *scratch; /* stage-0 ping buffer (owned) */
};

static SchedConvBuilt schedConvBuildKernel(int N, int n, const int *radices,
                                           int kStages, bool hasFma,
                                           const double *hMul,
                                           const double *hSwap, int maxTile,
                                           bool nt, int prefetch) {
    SchedConvBuilt b = {nullptr, nullptr, nullptr};
    if (kStages < 2) {
        throw std::bad_alloc(); /* the weld needs >= 2 stages */
    }

    int B[8];
    B[0] = radices[0];
    for (int t = 1; t < kStages; ++t) {
        B[t] = B[t - 1] * radices[t];
    }

    int groupLen[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    sched_block_groups(B, kStages, maxTile, groupLen);

    /* Inverse digit-reversal table gatherTab[p] = natural byte offset
     * (16*n; p = digitrev(n)); fused into the stage-0 gather. */
    std::vector<uint32_t> gatherTab;
    gatherTab.resize((size_t)N);
    for (int v = 0; v < N; ++v) {
        int e[8];
        int vv = v;
        for (int i = 0; i < kStages; ++i) {
            e[i] = vv % radices[kStages - 1 - i];
            vv /= radices[kStages - 1 - i];
        }
        int p = 0;
        for (int i = 0; i < kStages; ++i) {
            int m = 1;
            for (int l = 0; l < kStages - 1 - i; ++l) {
                m *= radices[l];
            }
            p += e[i] * m;
        }
        gatherTab[(size_t)p] = (uint32_t)(v * 16);
    }

    /* Per-stage TW tables (same layout/recurrence as the forward path):
     *   vector-2 (radix<=16): 64B slots [W^{jk}, W^{j(k+1)} | swap twin]
     *   single-k (radix 32):  32B slots [W^{jk} | twin (+wr|FMA/-wr)] */
    std::vector<std::vector<double>> twTab(kStages);
    for (int t = 1; t < kStages; ++t) {
        const int r = radices[t];
        const int Bp = B[t - 1];
        const int Bt = B[t];
        const bool v2 = (r <= 16);
        const int rowBytes = v2 ? r * 64 : r * 32;
        twTab[t].resize((size_t)rowBytes / 8 *
                        (size_t)(v2 ? Bp / 2 : Bp));
        const double zr = std::cos(-2.0 * kPi / (double)Bt);
        const double zi = std::sin(-2.0 * kPi / (double)Bt);
        const int rows = v2 ? Bp / 2 : Bp;
        for (int row = 0; row < rows; ++row) {
            const int k = v2 ? 2 * row : row;
            const double ang = -2.0 * kPi * (double)k / (double)Bt;
            const double br = std::cos(ang);
            const double bi = std::sin(ang); /* zeta^k */
            double b2r = 1.0, b2i = 0.0;     /* zeta^{k+1} */
            if (v2) {
                b2r = br * zr - bi * zi;
                b2i = br * zi + bi * zr;
            }
            double wr = 1.0, wi = 0.0;   /* zeta^{jk} */
            double w2r = 1.0, w2i = 0.0; /* zeta^{(k+1)j} */
            double *slot =
                &twTab[t][(size_t)row * (size_t)(rowBytes / 8)];
            for (int j = 0; j < r; ++j) {
                if (v2) {
                    slot[j * 8 + 0] = wr;
                    slot[j * 8 + 1] = wi;
                    slot[j * 8 + 2] = w2r;
                    slot[j * 8 + 3] = w2i;
                    slot[j * 8 + 4] = wi; /* swap twin */
                    slot[j * 8 + 5] = wr;
                    slot[j * 8 + 6] = w2i;
                    slot[j * 8 + 7] = w2r;
                } else {
                    slot[j * 4 + 0] = wr;
                    slot[j * 4 + 1] = wi;
                    slot[j * 4 + 2] = wi;
                    slot[j * 4 + 3] = hasFma ? wr : -wr;
                }
                const double nwr = wr * br - wi * bi;
                const double nwi = wr * bi + wi * br;
                wr = nwr;
                wi = nwi;
                if (v2) {
                    const double n2r = w2r * b2r - w2i * b2i;
                    const double n2i = w2r * b2i + w2i * b2r;
                    w2r = n2r;
                    w2i = n2i;
                }
            }
        }
    }

    /* LUT block: [gatherTab (N*4B, pad32)][TW_1..] */
    const size_t tabBytes = gatherTab.size() * 4;
    size_t cursor = (tabBytes + 31u) & ~31u;
    for (int t = 1; t < kStages; ++t) {
        cursor += twTab[t].size() * sizeof(double);
        cursor = (cursor + 31u) & ~31u;
    }
    b.lut = static_cast<double *>(
        aligned_alloc(64, cursor > 64 ? cursor : 64));
    if (b.lut == nullptr) {
        throw std::bad_alloc();
    }
    char *basep = reinterpret_cast<char *>(b.lut);
    uint32_t *tabPtr = reinterpret_cast<uint32_t *>(basep);
    for (size_t i = 0; i < gatherTab.size(); ++i) {
        tabPtr[i] = gatherTab[i];
    }
    SchedEmitterC::Stage stages[8];
    stages[0].radix = radices[0];
    stages[0].bprev = 1;
    stages[0].b = B[0];
    stages[0].tw = nullptr;
    cursor = (tabBytes + 31u) & ~31u;
    for (int t = 1; t < kStages; ++t) {
        double *dst = reinterpret_cast<double *>(basep + cursor);
        __builtin_memcpy(dst, twTab[t].data(),
                         twTab[t].size() * sizeof(double));
        stages[t].radix = radices[t];
        stages[t].bprev = B[t - 1];
        stages[t].b = B[t];
        stages[t].tw = dst;
        cursor += twTab[t].size() * sizeof(double);
        cursor = (cursor + 31u) & ~31u;
    }

    b.scratch = static_cast<double *>(aligned_alloc(64, (size_t)N * 16));
    if (b.scratch == nullptr) {
        free(b.lut);
        b.lut = nullptr;
        throw std::bad_alloc();
    }

    try {
        b.em = new SchedEmitterC(n, tabPtr, b.scratch, stages, kStages,
                                 hasFma, groupLen, hMul, hSwap, nt,
                                 prefetch);
    } catch (...) {
        free(b.lut);
        b.lut = nullptr;
        free(b.scratch);
        b.scratch = nullptr;
        throw;
    }
    return b;
}

/* ---- topic jit-fp64-linear-conv (DELTA F1): fused OLS build — the
 * sibling of schedConvBuildKernel above (which stays untouched — the
 * fp64 conv weld path is regression-only).  Table materialization is a
 * copy-then-edit: TWO gather tables (the OLS entry edit) + the same TW
 * rows + the plan-owned natural spectrum buffer spec.  No new twiddle
 * subsystem; the H tables come from the create function exactly as in
 * the conv weld. ---- */
struct SchedOlsBuilt {
    SchedEmitterO *em;
    double *lut;     /* gtabA + gtabB + TW tables (owned) */
    double *scratch; /* stage-0 ping buffer (owned) */
    double *spec;    /* natural spectrum transit buffer (owned) */
};

static SchedOlsBuilt schedOlsBuildKernel(int N, int n, const int *radices,
                                         int kStages, bool hasFma,
                                         const double *hMul,
                                         const double *hSwap, int nh,
                                         bool writeDiscard, int maxTile,
                                         bool nt, int prefetch) {
    SchedOlsBuilt b = {nullptr, nullptr, nullptr, nullptr};
    if (kStages < 2) {
        throw std::bad_alloc(); /* the weld needs >= 2 stages */
    }

    int B[8];
    B[0] = radices[0];
    for (int t = 1; t < kStages; ++t) {
        B[t] = B[t - 1] * radices[t];
    }

    int groupLen[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    sched_block_groups(B, kStages, maxTile, groupLen);

    /* Inverse digit-reversal tables.  gtabB = the conv weld's plain
     * mapping (slot p = digitrev(v) <- natural byte offset v*16; used by
     * half B's gather of the natural spectrum).  gtabA = the ROTATED
     * mapping (slot p <- natural ((v + nh-1) mod N) * 16): the OLS
     * window shift sunk into the half-A stage-0 LOAD boundary, so the
     * wrap-around-polluted head lands at the END of the cyclic result
     * and the valid segment is the natural PREFIX [0, N-nh]. */
    std::vector<uint32_t> gatherTab;
    std::vector<uint32_t> gatherTabA;
    gatherTab.resize((size_t)N);
    gatherTabA.resize((size_t)N);
    for (int v = 0; v < N; ++v) {
        int e[8];
        int vv = v;
        for (int i = 0; i < kStages; ++i) {
            e[i] = vv % radices[kStages - 1 - i];
            vv /= radices[kStages - 1 - i];
        }
        int p = 0;
        for (int i = 0; i < kStages; ++i) {
            int m = 1;
            for (int l = 0; l < kStages - 1 - i; ++l) {
                m *= radices[l];
            }
            p += e[i] * m;
        }
        gatherTab[(size_t)p] = (uint32_t)(v * 16);
        gatherTabA[(size_t)p] =
            (uint32_t)(((v + nh - 1) % N) * 16);
    }

    /* Per-stage TW tables: identical to the conv weld (same layout /
     * recurrence — copy-then-edit keeps them byte-equal). */
    std::vector<std::vector<double>> twTab(kStages);
    for (int t = 1; t < kStages; ++t) {
        const int r = radices[t];
        const int Bp = B[t - 1];
        const int Bt = B[t];
        const bool v2 = (r <= 16);
        const int rowBytes = v2 ? r * 64 : r * 32;
        twTab[t].resize((size_t)rowBytes / 8 *
                        (size_t)(v2 ? Bp / 2 : Bp));
        const double zr = std::cos(-2.0 * kPi / (double)Bt);
        const double zi = std::sin(-2.0 * kPi / (double)Bt);
        const int rows = v2 ? Bp / 2 : Bp;
        for (int row = 0; row < rows; ++row) {
            const int k = v2 ? 2 * row : row;
            const double ang = -2.0 * kPi * (double)k / (double)Bt;
            const double br = std::cos(ang);
            const double bi = std::sin(ang); /* zeta^k */
            double b2r = 1.0, b2i = 0.0;     /* zeta^{k+1} */
            if (v2) {
                b2r = br * zr - bi * zi;
                b2i = br * zi + bi * zr;
            }
            double wr = 1.0, wi = 0.0;   /* zeta^{jk} */
            double w2r = 1.0, w2i = 0.0; /* zeta^{(k+1)j} */
            double *slot =
                &twTab[t][(size_t)row * (size_t)(rowBytes / 8)];
            for (int j = 0; j < r; ++j) {
                if (v2) {
                    slot[j * 8 + 0] = wr;
                    slot[j * 8 + 1] = wi;
                    slot[j * 8 + 2] = w2r;
                    slot[j * 8 + 3] = w2i;
                    slot[j * 8 + 4] = wi; /* swap twin */
                    slot[j * 8 + 5] = wr;
                    slot[j * 8 + 6] = w2i;
                    slot[j * 8 + 7] = w2r;
                } else {
                    slot[j * 4 + 0] = wr;
                    slot[j * 4 + 1] = wi;
                    slot[j * 4 + 2] = wi;
                    slot[j * 4 + 3] = hasFma ? wr : -wr;
                }
                const double nwr = wr * br - wi * bi;
                const double nwi = wr * bi + wi * br;
                wr = nwr;
                wi = nwi;
                if (v2) {
                    const double n2r = w2r * b2r - w2i * b2i;
                    const double n2i = w2r * b2i + w2i * b2r;
                    w2r = n2r;
                    w2i = n2i;
                }
            }
        }
    }

    /* LUT block: [gtabA (N*4B, pad32)][gtabB (N*4B, pad32)][TW_1..] */
    const size_t tabABytes =
        (gatherTabA.size() * 4 + 31u) & ~31u;
    const size_t tabBytes = tabABytes +
                           ((gatherTab.size() * 4 + 31u) & ~31u);
    size_t cursor = tabBytes;
    for (int t = 1; t < kStages; ++t) {
        cursor += twTab[t].size() * sizeof(double);
        cursor = (cursor + 31u) & ~31u;
    }
    b.lut = static_cast<double *>(
        aligned_alloc(64, cursor > 64 ? cursor : 64));
    if (b.lut == nullptr) {
        throw std::bad_alloc();
    }
    char *basep = reinterpret_cast<char *>(b.lut);
    uint32_t *tabAPtr = reinterpret_cast<uint32_t *>(basep);
    for (size_t i = 0; i < gatherTabA.size(); ++i) {
        tabAPtr[i] = gatherTabA[i];
    }
    uint32_t *tabBPtr = reinterpret_cast<uint32_t *>(basep + tabABytes);
    for (size_t i = 0; i < gatherTab.size(); ++i) {
        tabBPtr[i] = gatherTab[i];
    }
    SchedEmitterO::Stage stages[8];
    stages[0].radix = radices[0];
    stages[0].bprev = 1;
    stages[0].b = B[0];
    stages[0].tw = nullptr;
    cursor = tabBytes;
    for (int t = 1; t < kStages; ++t) {
        double *dst = reinterpret_cast<double *>(basep + cursor);
        __builtin_memcpy(dst, twTab[t].data(),
                         twTab[t].size() * sizeof(double));
        stages[t].radix = radices[t];
        stages[t].bprev = B[t - 1];
        stages[t].b = B[t];
        stages[t].tw = dst;
        cursor += twTab[t].size() * sizeof(double);
        cursor = (cursor + 31u) & ~31u;
    }

    b.scratch = static_cast<double *>(aligned_alloc(64, (size_t)N * 16));
    if (b.scratch == nullptr) {
        free(b.lut);
        b.lut = nullptr;
        throw std::bad_alloc();
    }
    b.spec = static_cast<double *>(aligned_alloc(64, (size_t)N * 16));
    if (b.spec == nullptr) {
        free(b.scratch);
        b.scratch = nullptr;
        free(b.lut);
        b.lut = nullptr;
        throw std::bad_alloc();
    }

    try {
        b.em = new SchedEmitterO(n, tabAPtr, tabBPtr, b.scratch, b.spec,
                                 stages, kStages, hasFma, groupLen, hMul,
                                 hSwap, nt, prefetch, nh, writeDiscard);
    } catch (...) {
        free(b.spec);
        b.spec = nullptr;
        free(b.scratch);
        b.scratch = nullptr;
        free(b.lut);
        b.lut = nullptr;
        throw;
    }
    return b;
}

/* ---- topic jit-fp64-inverse (amend resume 2026-09-17, DELTA F3): fused
 * INVERSE build — the sibling of schedConvBuildKernel above (which stays
 * untouched — the fp64 conv weld path is regression-only).  Tables +
 * SchedEmitterI for one decomposition.  Table materialization (gatherTab +
 * TW rows) is a copy-then-edit of schedConvBuildKernel; NO H tables — the
 * inverse's boundary ops are pure code (gather-load conj against the rip
 * sign mask, final-store conj*invN against the (invN, -invN) rip constant),
 * so the single twiddle LUT discipline holds (no fourth twiddle subsystem,
 * no inverse twiddle table — the conjugate trick keeps the forward TW
 * semantics directly usable). ---- */
struct SchedInvBuilt {
    SchedEmitterI *em;
    double *lut;     /* gatherTab32 + TW tables (owned) */
    double *scratch; /* stage-0 ping buffer (owned) */
};

static SchedInvBuilt schedInvBuildKernel(int N, int n, const int *radices,
                                         int kStages, bool hasFma, int maxTile,
                                         bool nt, int prefetch) {
    SchedInvBuilt b = {nullptr, nullptr, nullptr};
    if (kStages < 2) {
        throw std::bad_alloc(); /* the fused final needs >= 2 stages */
    }

    int B[8];
    B[0] = radices[0];
    for (int t = 1; t < kStages; ++t) {
        B[t] = B[t - 1] * radices[t];
    }

    int groupLen[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    sched_block_groups(B, kStages, maxTile, groupLen);

    /* Inverse digit-reversal table gatherTab[p] = natural byte offset
     * (16*n; p = digitrev(n)); fused into the stage-0 gather. */
    std::vector<uint32_t> gatherTab;
    gatherTab.resize((size_t)N);
    for (int v = 0; v < N; ++v) {
        int e[8];
        int vv = v;
        for (int i = 0; i < kStages; ++i) {
            e[i] = vv % radices[kStages - 1 - i];
            vv /= radices[kStages - 1 - i];
        }
        int p = 0;
        for (int i = 0; i < kStages; ++i) {
            int m = 1;
            for (int l = 0; l < kStages - 1 - i; ++l) {
                m *= radices[l];
            }
            p += e[i] * m;
        }
        gatherTab[(size_t)p] = (uint32_t)(v * 16);
    }

    /* Per-stage TW tables (same layout/recurrence as the forward path —
     * the conjugate trick consumes the FORWARD twiddles unchanged):
     *   vector-2 (radix<=16): 64B slots [W^{jk}, W^{j(k+1)} | swap twin]
     *   single-k (radix 32):  32B slots [W^{jk} | twin (+wr|FMA/-wr)] */
    std::vector<std::vector<double>> twTab(kStages);
    for (int t = 1; t < kStages; ++t) {
        const int r = radices[t];
        const int Bp = B[t - 1];
        const int Bt = B[t];
        const bool v2 = (r <= 16);
        const int rowBytes = v2 ? r * 64 : r * 32;
        twTab[t].resize((size_t)rowBytes / 8 *
                        (size_t)(v2 ? Bp / 2 : Bp));
        const double zr = std::cos(-2.0 * kPi / (double)Bt);
        const double zi = std::sin(-2.0 * kPi / (double)Bt);
        const int rows = v2 ? Bp / 2 : Bp;
        for (int row = 0; row < rows; ++row) {
            const int k = v2 ? 2 * row : row;
            const double ang = -2.0 * kPi * (double)k / (double)Bt;
            const double br = std::cos(ang);
            const double bi = std::sin(ang); /* zeta^k */
            double b2r = 1.0, b2i = 0.0;     /* zeta^{k+1} */
            if (v2) {
                b2r = br * zr - bi * zi;
                b2i = br * zi + bi * zr;
            }
            double wr = 1.0, wi = 0.0;   /* zeta^{jk} */
            double w2r = 1.0, w2i = 0.0; /* zeta^{(k+1)j} */
            double *slot =
                &twTab[t][(size_t)row * (size_t)(rowBytes / 8)];
            for (int j = 0; j < r; ++j) {
                if (v2) {
                    slot[j * 8 + 0] = wr;
                    slot[j * 8 + 1] = wi;
                    slot[j * 8 + 2] = w2r;
                    slot[j * 8 + 3] = w2i;
                    slot[j * 8 + 4] = wi; /* swap twin */
                    slot[j * 8 + 5] = wr;
                    slot[j * 8 + 6] = w2i;
                    slot[j * 8 + 7] = w2r;
                } else {
                    slot[j * 4 + 0] = wr;
                    slot[j * 4 + 1] = wi;
                    slot[j * 4 + 2] = wi;
                    slot[j * 4 + 3] = hasFma ? wr : -wr;
                }
                const double nwr = wr * br - wi * bi;
                const double nwi = wr * bi + wi * br;
                wr = nwr;
                wi = nwi;
                if (v2) {
                    const double n2r = w2r * b2r - w2i * b2i;
                    const double n2i = w2r * b2i + w2i * b2r;
                    w2r = n2r;
                    w2i = n2i;
                }
            }
        }
    }

    /* LUT block: [gatherTab (N*4B, pad32)][TW_1..] */
    const size_t tabBytes = gatherTab.size() * 4;
    size_t cursor = (tabBytes + 31u) & ~31u;
    for (int t = 1; t < kStages; ++t) {
        cursor += twTab[t].size() * sizeof(double);
        cursor = (cursor + 31u) & ~31u;
    }
    b.lut = static_cast<double *>(
        aligned_alloc(64, cursor > 64 ? cursor : 64));
    if (b.lut == nullptr) {
        throw std::bad_alloc();
    }
    char *basep = reinterpret_cast<char *>(b.lut);
    uint32_t *tabPtr = reinterpret_cast<uint32_t *>(basep);
    for (size_t i = 0; i < gatherTab.size(); ++i) {
        tabPtr[i] = gatherTab[i];
    }
    SchedEmitterI::Stage stages[8];
    stages[0].radix = radices[0];
    stages[0].bprev = 1;
    stages[0].b = B[0];
    stages[0].tw = nullptr;
    cursor = (tabBytes + 31u) & ~31u;
    for (int t = 1; t < kStages; ++t) {
        double *dst = reinterpret_cast<double *>(basep + cursor);
        __builtin_memcpy(dst, twTab[t].data(),
                         twTab[t].size() * sizeof(double));
        stages[t].radix = radices[t];
        stages[t].bprev = B[t - 1];
        stages[t].b = B[t];
        stages[t].tw = dst;
        cursor += twTab[t].size() * sizeof(double);
        cursor = (cursor + 31u) & ~31u;
    }

    b.scratch = static_cast<double *>(aligned_alloc(64, (size_t)N * 16));
    if (b.scratch == nullptr) {
        free(b.lut);
        b.lut = nullptr;
        throw std::bad_alloc();
    }

    try {
        b.em = new SchedEmitterI(n, tabPtr, b.scratch, stages, kStages,
                                 hasFma, groupLen, nt, prefetch);
    } catch (...) {
        free(b.lut);
        b.lut = nullptr;
        free(b.scratch);
        b.scratch = nullptr;
        throw;
    }
    return b;
}

/* ========================================================================
 * topic jit-codelet-weld — SchedEmitterW: ORIGINAL-SIZE non-2^k weld
 * (DELTA F1/F2/F3; draft {#BEH-FFT-016} / {#API-FFT-011} coverage upgrade).
 *
 * Copy-then-edit of SchedEmitterC above (which stays untouched — the pow2
 * conv path is regression-only).  The schedule stays the promoted mixed
 * radix DIT (N = r1*..*rk, stage-0 digitrev gather out-of-place into the
 * scratch, middle sweeps in-place, final stage writes the natural array),
 * but each stage's LEAF now comes from a searched dispatch:
 *
 *   - own vector leaves for pow2 radices (v2 2/4/8/16, single-k 32) —
 *     the promoted code, byte-for-byte same leaf emitters, with the same
 *     parity constraints (twiddled v2 needs even Bp; stage-0 v2 gather
 *     needs an even block count, else a scalar radix-2 gather runs);
 *   - FFTW codelet leaves (L-K) called through ONE plan-time C helper
 *     (weld_run_stage) per stage sweep: t1fv (twiddles fused inside the
 *     leaf, even Bp) or n1fv + a vectorized twiddle pre-sweep (radix 13
 *     has no t1fv; odd-Bp correctness path).
 *
 * Non-2^k geometry (L-L): the digitrev table is the general mixed-radix
 * one (the pow2 engine already generated it generally — only the leaf
 * domain was 2-only); inter-stage twiddles W_{B_t}^{jk} on non-pow2
 * strides ride the t1fv W tables, the TW sweep tables, or the own-leaf TW
 * rows (same single LUT discipline); every load/store stays unaligned
 * (movupd/vmovupd) — no 32B alignment assumption survives non-pow2 block
 * strides (explicit alignment stance, DESIGN failure modes).
 *
 * Fused-convolution boundary (F3), welded at original N (zero padding
 * inflation):
 *   - half A forward; own final stage stores conj(X*H) at the store
 *     boundary (mode 1, SchedEmitterC idiom);
 *   - when the final stage is a codelet (e.g. 5850 = 2*13*9*25 shapes),
 *     the boundary moves: half A stores plain X, and half B's stage-0
 *     gather FUSES the pointwise op at its loads (mode 3: conj(X) *
 *     conj(H) via a 16B sign flip + complex multiply against weld_hc
 *     slots indexed by the gather offset — same load-fusion idea as the
 *     promoted store fusion, opposite side of the io transit);
 *   - when stage 0 is a codelet too (odd N), half A closes with a
 *     CONJMUL sweep and half B with an INVN sweep (weld_run_stage);
 *   - half B final: own stores conj(.) * (1/N) (mode 2); codelet gets an
 *     INVN sweep.  The intermediate spectrum still transits io exactly
 *     once.
 * ======================================================================== */
enum {
    WSK_OWN_V2 = 0, /* own vector-2 leaf (radix 2/4/8/16) */
    WSK_OWN_32,     /* own single-k radix-32 leaf */
    WSK_OWN_S2,     /* own scalar radix-2 stage-0 gather (odd block count) */
    WSK_T1,         /* FFTW t1fv codelet (twiddles inside; even Bp) */
    WSK_N1,         /* FFTW n1fv codelet + twiddle pre-sweep (even Bp) */
    WSK_N1_ODD,     /* n1fv odd-Bp path (tail instances + slack padding) */
    WSK_N1_CONTIG   /* stage-0 contiguous n1fv on the digitrev scratch */
};

class SchedEmitterW final : public Xbyak::CodeGenerator {
public:
    struct Stage {
        int kind;          /* WSK_* */
        int radix;         /* r_t */
        int bprev;         /* B_{t-1} in elements (t=0 -> 1) */
        int b;             /* B_t in elements */
        const double *tw;  /* own stages: TW_t base (nullptr t=0) */
        const weld_stage_desc *desc; /* codelet stages: sweep descriptor */
        const weld_stage_desc *tws;  /* n1 stages: twiddle sweep descriptor */
    };

    /* conv=0: single forward pass (H materialization).  conv=1: two welded
     * halves.  loadFusion: half B's stage-0 gather multiplies conj(H)
     * (used when the final stage is a codelet).
     *
     * topic jit-anyN-fwd-inv (DELTA F1, draft {#API-FFT-013}): invSingle=1
     * emits the INVERSE single-transform pass — the conv half-B geometry
     * with the operator interfaces degenerated per DESIGN §L-Q: entry
     * boundary conj sunk into the stage-0 gather loads (the mode-3 16B
     * sign flip without the conj(H) multiply; codelet stage-0 = the
     * conjugating PERMUTE descriptor), exit boundary conj(.)·(1/N) at the
     * final store (own final = the mode-2 store form itself; codelet
     * final = the WELD_RUN_INVN sweep).  IFFT(x) = conj(F(conj(x)))/N. */
    SchedEmitterW(int N, const uint32_t *gatherTab, const double *scratch,
                  const Stage *stages, int kStages, bool fma, bool conv,
                  bool loadFusion, const double *hMul, const double *hSwap,
                  const double *hcNat, const weld_stage_desc *dPermute,
                  const weld_stage_desc *dConjMul,
                  const weld_stage_desc *dInvN, double *scratch2,
                  bool invSingle = false)
        : Xbyak::CodeGenerator(524288), fma_(fma), N_(N), conv_(conv),
          loadFusion_(loadFusion), scratch_(scratch), hcNat_(hcNat),
          dPermute_(dPermute), dConjMul_(dConjMul), dInvN_(dInvN),
          scratch2_(scratch2), tab_(gatherTab),
          stages_(stages), kStages_(kStages), invSingle_(invSingle),
          entryConj_(invSingle) {
        spillOff_ = -304;
        push(rbp);
        mov(rbp, rsp);
        push(rbx);
        push(r12);
        push(r13);
        push(r14); /* own-final mode-1 hMul anchor */
        push(r15); /* own-final mode-1 hSwap anchor */
        sub(rsp, 272);
        mov(ptr[rbp - 312], rdi); /* park the caller io (rsp == rbp-312) */

        const uint64_t hm = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(hMul) -
            reinterpret_cast<uintptr_t>(scratch));
        const uint64_t hs = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(hSwap) -
            reinterpret_cast<uintptr_t>(scratch));
        mov(r14, hm);
        mov(r15, hs);

        if (invSingle_) {
            /* topic jit-anyN-fwd-inv: ONE inverse pass — the half-B
             * geometry with the entry conj at the stage-0 gather boundary
             * and the exit conj·invN at the final boundary; no half A,
             * no conv operators. */
            emitHalf(2);
        } else if (!conv_) {
            emitHalf(0);
        } else {
            emitHalf(1);
            mov(rdi, ptr[rbp - 312]); /* restore io for half B */
            emitHalf(2);
        }

        vzeroupper();
        lea(rsp, ptr[rbp - 40]);
        pop(r15);
        pop(r14);
        pop(r13);
        pop(r12);
        pop(rbx);
        pop(rbp);
        ret();
        emitPool();
    }

private:
    static int lf_log2(int v) {
        int s = 0;
        while ((1 << s) < v) {
            ++s;
        }
        return s;
    }

    /* ---- JIT -> C call boundary (System V AMD64): descriptor + bases in
     * rdi/rsi/rdx (caller-saved; every live cursor sits in callee-saved
     * registers or the [rbp-312] io park), target via rax, 16B stack
     * alignment restored around the call (body rsp ≡ 8 mod 16). ---- */
    /* base2Sel: 0 = scratch, 1 = caller io (runtime), 2 = scratch2 */
    void weldCall(const weld_stage_desc *d, bool baseIo, int base2Sel) {
        using namespace Xbyak;
        mov(rdi, reinterpret_cast<uint64_t>(d));
        if (baseIo) {
            mov(rsi, ptr[rbp - 312]);
        } else {
            mov(rsi, reinterpret_cast<uint64_t>(scratch_));
        }
        if (base2Sel == 1) {
            mov(rdx, ptr[rbp - 312]);
        } else if (base2Sel == 2) {
            mov(rdx, reinterpret_cast<uint64_t>(scratch2_));
        } else {
            mov(rdx, reinterpret_cast<uint64_t>(scratch_));
        }
        mov(rax, reinterpret_cast<uint64_t>(&weld_run_stage));
        sub(rsp, 8);
        call(rax);
        add(rsp, 8);
    }

    /* src -> io copy (t1fv finals are in-place-only on the scratch; the
     * odd-tail N1 final lands in scratch2) */
    void emitCopyToIo(const double *src) {
        using namespace Xbyak;
        mov(r8, reinterpret_cast<uint64_t>(src));
        mov(r9, ptr[rbp - 312]);
        mov(rcx, (unsigned)(N_ / 2));
        Label cp;
        L(cp);
        vmovupd(ymm0, ptr[r8]);
        vmovupd(ptr[r9], ymm0);
        add(r8, 32);
        add(r9, 32);
        sub(rcx, 1);
        jnz(cp);
        if (N_ % 2 != 0) {
            vmovupd(xmm0, ptr[r8]);
            vmovupd(ptr[r9], xmm0);
        }
    }

    /* mode 0 forward, 1 half A, 2 half B */
    void emitHalf(int mode) {
        using namespace Xbyak;
        finalMode_ = 0;
        loadFuse_ = (mode == 2) && loadFusion_ && hcNat_ != nullptr;
        if (loadFuse_) {
            /* absolute anchor: the gather load runs at [io + off] and the
             * fused conj(H) slot sits at hcNat + off*2 (32B slot per 16B
             * element) — the memory operand [rbx + off*2] covers it. */
            mov(rbx, reinterpret_cast<uint64_t>(hcNat_));
        }

        const Stage &st0 = stages_[0];
        if (st0.kind == WSK_OWN_V2 || st0.kind == WSK_OWN_32) {
            emitStage0Plain(st0);
        } else if (st0.kind == WSK_OWN_S2) {
            emitGatherScalar2(st0);
        } else {
            /* codelet stage 0: digitrev permutation pass + contiguous
             * notwiddled codelet on the permuted scratch */
            weldCall(dPermute_, /*baseIo=*/true, /*base2Sel=*/0);
            weldCall(st0.desc, false, false);
            mov(rdx, ptr[rbp - 312]);
            mov(rdi, reinterpret_cast<uint64_t>(scratch_));
        }

        for (int t = 1; t < kStages_ - 1; ++t) {
            const Stage &st = stages_[t];
            if (st.kind == WSK_OWN_V2 || st.kind == WSK_OWN_32) {
                emitSweepFull(st);
            } else {
                if (st.tws != nullptr) {
                    weldCall(st.tws, /*baseIo=*/false, /*base2Sel=*/0);
                }
                weldCall(st.desc, /*baseIo=*/false, /*base2Sel=*/0);
                mov(rdi, reinterpret_cast<uint64_t>(scratch_));
            }
        }

        const Stage &stf = stages_[kStages_ - 1];
        if (stf.kind == WSK_OWN_V2 || stf.kind == WSK_OWN_32) {
            /* topic jit-anyN-fwd-inv: the inverse single pass keeps the
             * half-B final form (mode 2 = conj(.)·invN store) — that IS
             * the inverse, not a conv operator (DESIGN §L-Q rule 3). */
            emitFinalStage(stf, conv_ ? mode : (invSingle_ ? 2 : 0));
        } else {
            if (stf.tws != nullptr) {
                weldCall(stf.tws, /*baseIo=*/false, /*base2Sel=*/0);
            }
            if (stf.kind == WSK_N1_ODD && scratch2_ != nullptr) {
                /* odd block count + caller-owned out: land the tail
                 * padding in the second scratch, then copy out */
                weldCall(stf.desc, /*baseIo=*/false, /*base2Sel=*/2);
                emitCopyToIo(scratch2_);
            } else if (stf.kind == WSK_T1) {
                /* t1fv is in-place-only (no output pointers): finish on
                 * the scratch, then copy the natural result to io */
                weldCall(stf.desc, /*baseIo=*/false, /*base2Sel=*/0);
                emitCopyToIo(scratch_);
            } else {
                weldCall(stf.desc, /*baseIo=*/false, /*base2Sel=*/1);
            }
            mov(rdi, reinterpret_cast<uint64_t>(scratch_));
            if (mode == 1 && !loadFusion_ && dConjMul_ != nullptr) {
                weldCall(dConjMul_, /*baseIo=*/true, /*base2Sel=*/1);
            }
            if (mode == 2 && dInvN_ != nullptr) {
                weldCall(dInvN_, /*baseIo=*/true, /*base2Sel=*/1);
            }
        }
        loadFuse_ = false;
    }

    /* ---- stage 0, own leaves: digitrev gather fused into the first
     * codelet sweep, out-of-place into the scratch (promoted shape) ---- */
    void emitStage0Plain(const Stage &st) {
        using namespace Xbyak;
        mov(r8, reinterpret_cast<uint64_t>(tab_));
        mov(rsi, reinterpret_cast<uint64_t>(scratch_));
        lea(rdx, ptr[rsi + N_ * 16]);
        Label blk;
        L(blk);
        gatherTab_ = &r8;
        if (st.radix <= 16) {
            emitLeafV2(st.radix, rsi, 16, rsi, false);
            add(r8, st.radix * 8);
            add(rsi, st.radix * 32);
        } else {
            emitLeaf32(r8, 16, false, rsi);
            add(r8, st.radix * 4);
            add(rsi, st.radix * 16);
        }
        gatherTab_ = nullptr;
        cmp(rsi, rdx);
        jb(blk);
        mov(rdx, ptr[rbp - 312]);
        mov(rdi, reinterpret_cast<uint64_t>(scratch_));
    }

    /* ---- scalar radix-2 stage-0 gather: one block per iteration, no
     * block pairing (v2 gather needs an even block count; shapes like
     * 5850 = 2*... have an odd N/2).  Optional mode-3 load fusion. ---- */
    void emitGatherScalar2(const Stage &st) {
        using namespace Xbyak;
        (void)st;
        mov(r8, reinterpret_cast<uint64_t>(tab_));
        mov(rsi, reinterpret_cast<uint64_t>(scratch_));
        lea(rdx, ptr[rsi + N_ * 16]);
        Label blk;
        L(blk);
        mov(eax, dword[r8]);
        vmovupd(xmm8, ptr[rdi + rax]);
        if (loadFuse_) {
            fuseLoadConjMul(xmm8, rax);
        } else if (entryConj_) { /* topic jit-anyN-fwd-inv:
         * inverse entry boundary conj — the mode-3 16B sign
         * flip with no conv operator (DESIGN L-Q degeneracy) */
                vxorpd(xmm8, xmm8, ptr[rip + sign16Lbl_]);
        }
        mov(eax, dword[r8 + 4]);
        vmovupd(xmm9, ptr[rdi + rax]);
        if (loadFuse_) {
            fuseLoadConjMul(xmm9, rax);
        } else if (entryConj_) { /* topic jit-anyN-fwd-inv:
         * inverse entry boundary conj — the mode-3 16B sign
         * flip with no conv operator (DESIGN L-Q degeneracy) */
                vxorpd(xmm9, xmm9, ptr[rip + sign16Lbl_]);
        }
        vaddpd(xmm10, xmm8, xmm9);
        vsubpd(xmm11, xmm8, xmm9);
        vmovupd(ptr[rsi], xmm10);
        vmovupd(ptr[rsi + 16], xmm11);
        add(r8, 8);
        add(rsi, 32);
        cmp(rsi, rdx);
        jb(blk);
        mov(rdx, ptr[rbp - 312]);
        mov(rdi, reinterpret_cast<uint64_t>(scratch_));
    }

    /* mode-3 load fusion on one gathered 16B element: x <- conj(x)*conjH
     * = conj(x*H) (sign flip + complex multiply vs the weld_hc slot
     * indexed by the gather byte offset; FMA3 idiom). */
    void fuseLoadConjMul(const Xbyak::Xmm &x, const Xbyak::Reg64 &off) {
        using namespace Xbyak;
        vxorpd(x, x, ptr[rip + sign16Lbl_]);
        vmovddup(xmm13, x);
        vshufpd(xmm12, x, x, 3);
        vmulpd(xmm12, xmm12, ptr[rbx + off * 2 + 16]);
        vfmaddsub231pd(xmm12, xmm13, ptr[rbx + off * 2]);
        vmovapd(x, xmm12);
    }

    /* ---- one middle stage's full-array sweep (own leaf) ---- */
    void emitSweepFull(const Stage &st) {
        using namespace Xbyak;
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, rdi);
        lea(r8, ptr[rdi + N_ * 16]);
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 16]);
        mov(rax, rsi);
        mov(r10, r9);
        Label kLoop;
        L(kLoop);
        if (st.radix <= 16) {
            emitLeafV2(st.radix, rax, st.bprev * 16, rax, true);
            add(rax, 32);
            add(r10, st.radix * 64);
        } else {
            emitLeaf32(rax, st.bprev * 16, true, rax);
            add(rax, 16);
            add(r10, 32 * 32);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 16);
        cmp(rsi, r8);
        jb(blk);
    }

    /* ---- final stage (own leaf): reads the scratch, writes the natural
     * array with the fused boundary op (mode 1 = conj(X*H), 2 = conj*invN)
     * — SchedEmitterC idiom, rax = scratch k-cursor anchors r14/r15. ---- */
    void emitFinalStage(const Stage &st, int mode) {
        using namespace Xbyak;
        finalMode_ = mode;
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, rdi);
        lea(r8, ptr[rdi + N_ * 16]);
        mov(r12, ptr[rbp - 312]); /* natural block cursor = io */
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 16]);
        mov(rax, rsi);
        mov(r10, r9);
        Label kLoop;
        L(kLoop);
        if (st.radix <= 16) {
            mov(rbx, r12);
            emitLeafV2(st.radix, rax, st.bprev * 16, rbx, true);
            add(rax, 32);
            add(r10, st.radix * 64);
            add(r12, 32);
        } else {
            mov(rbx, r12);
            emitLeaf32(rax, st.bprev * 16, true, rbx);
            add(rax, 16);
            add(r10, 32 * 32);
            add(r12, 16);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 16);
        add(r12, st.b * 16);
        cmp(rsi, r8);
        jb(blk);
        finalMode_ = 0;
    }

    /* ---- fused final-stage boundary op on a (k, k+1) natural pair chunk
     * (SchedEmitterC idiom; mode 1 needs r14/r15 anchors + the rax scratch
     * k-cursor, mode 2 the (invN, -invN) rip constant) ---- */
    void fuseFinalPairInPlace(const Xbyak::Ymm &v, int off) {
        using namespace Xbyak;
        if (finalMode_ == 1) {
            if (fma_) {
                vmovddup(ymm12, v);
                vshufpd(v, v, v, 0xF);
                vmulpd(v, v, ptr[r15 + rax + off]);
                vfmaddsub231pd(v, ymm12, ptr[r14 + rax + off]);
            } else {
                vmulpd(ymm12, v, ptr[r14 + rax + off]);
                vmulpd(ymm13, v, ptr[r15 + rax + off]);
                vhsubpd(v, ymm12, ymm13);
            }
            vxorpd(v, v, ptr[rip + sign2Lbl_]);
        } else if (finalMode_ == 2) {
            vmulpd(v, v, ptr[rip + invNLbl_]);
        }
    }

    /* mode-1 fusion on ONE 16B complex (leaf32 final: its ymm lanes are
     * j-strided DIT positions, not natural-adjacent pairs, so the
     * 32B-pair idiom does not apply — fuse each extracted lane against
     * its own H element).  mode 2 folds into the (invN,-invN) multiply
     * and stays lane-agnostic. */
    void fuseFinalScalar16(const Xbyak::Xmm &x, int off) {
        using namespace Xbyak;
        if (finalMode_ == 1) {
            if (fma_) {
                vmovddup(xmm11, x);            /* (ar, ar) */
                vshufpd(xmm10, x, x, 3);       /* (ai, ai) */
                vmulpd(xmm10, xmm10, ptr[r15 + rax + off]);
                vfmaddsub231pd(xmm10, xmm11, ptr[r14 + rax + off]);
                vmovapd(x, xmm10);
            } else {
                vmulpd(xmm10, x, ptr[r14 + rax + off]);
                vmulpd(xmm11, x, ptr[r15 + rax + off]);
                vhsubpd(x, xmm10, xmm11);
            }
            vxorpd(x, x, ptr[rip + sign16Lbl_]);
        } else if (finalMode_ == 2) {
            /* per-complex (invN, -invN): fold via a 16B multiply against
             * the low half of the 32B invN constant */
            vmulpd(x, x, ptr[rip + invNLbl_]);
        }
    }

    /* ---- vector-2 leaf dispatch (own radix 2/4/8/16) ---- */
    void emitLeafV2(int r, const Xbyak::Reg64 &loadBase, int strideBytes,
                    const Xbyak::Reg64 &storeBase, bool hasTw) {
        using namespace Xbyak;
        if (r >= 4 || (!fma_ && hasTw)) {
            vmovapd(ymm14, ptr[rip + sign2Lbl_]);
        }
        if (r == 16) {
            emitV2Sub(8, loadBase, strideBytes, hasTw, 0);
            for (int j = 0; j < 8; ++j) {
                vmovupd(ptr[rbp + spillOff_ + j * 32], Ymm(j));
            }
            emitV2Sub(8, loadBase, strideBytes, hasTw, 1);
            cross16V2(storeBase, strideBytes);
        } else {
            emitV2Sub(r, loadBase, strideBytes, hasTw, -1);
            storeV2(r, storeBase, strideBytes);
        }
    }

    void emitV2Sub(int r, const Xbyak::Reg64 &loadBase, int strideBytes,
                   bool hasTw, int par) {
        using namespace Xbyak;
        const int nlf = lf_log2(r);
        const int rr = (par >= 0) ? 16 : r;
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};

        for (int u = 0; u < r / 2; ++u) {
            int sa, sb;
            if (par >= 0) {
                sa = 2 * (int)bitrev((uint32_t)(2 * u), nlf) + par;
                sb = 2 * (int)bitrev((uint32_t)(2 * u + 1), nlf) + par;
            } else {
                sa = (int)bitrev((uint32_t)(2 * u), nlf);
                sb = (int)bitrev((uint32_t)(2 * u + 1), nlf);
            }
            if (gatherTab_ != nullptr) {
                const Reg64 &tab = *gatherTab_;
                mov(eax, dword[tab + sa * 4]);
                vmovupd(xmm8, ptr[rdi + rax]);
                if (loadFuse_) {
                    fuseLoadConjMul(xmm8, rax);
                } else if (entryConj_) { /* topic jit-anyN-fwd-inv:
                 * inverse entry boundary conj — the mode-3 16B sign
                 * flip with no conv operator (DESIGN L-Q degeneracy) */
                        vxorpd(xmm8, xmm8, ptr[rip + sign16Lbl_]);
                }
                mov(eax, dword[tab + (rr + sa) * 4]);
                vmovupd(xmm9, ptr[rdi + rax]);
                if (loadFuse_) {
                    fuseLoadConjMul(xmm9, rax);
                } else if (entryConj_) { /* topic jit-anyN-fwd-inv:
                 * inverse entry boundary conj — the mode-3 16B sign
                 * flip with no conv operator (DESIGN L-Q degeneracy) */
                        vxorpd(xmm9, xmm9, ptr[rip + sign16Lbl_]);
                }
                mov(eax, dword[tab + sb * 4]);
                vmovupd(xmm10, ptr[rdi + rax]);
                if (loadFuse_) {
                    fuseLoadConjMul(xmm10, rax);
                } else if (entryConj_) { /* topic jit-anyN-fwd-inv:
                 * inverse entry boundary conj — the mode-3 16B sign
                 * flip with no conv operator (DESIGN L-Q degeneracy) */
                        vxorpd(xmm10, xmm10, ptr[rip + sign16Lbl_]);
                }
                mov(eax, dword[tab + (rr + sb) * 4]);
                vmovupd(xmm11, ptr[rdi + rax]);
                if (loadFuse_) {
                    fuseLoadConjMul(xmm11, rax);
                } else if (entryConj_) { /* topic jit-anyN-fwd-inv:
                 * inverse entry boundary conj — the mode-3 16B sign
                 * flip with no conv operator (DESIGN L-Q degeneracy) */
                        vxorpd(xmm11, xmm11, ptr[rip + sign16Lbl_]);
                }
                vaddpd(xmm12, xmm8, xmm10);
                vsubpd(xmm13, xmm8, xmm10);
                vaddpd(xmm8, xmm9, xmm11);
                vsubpd(xmm10, xmm9, xmm11);
                vinsertf128(P[2 * u], P[2 * u], xmm12, 0);
                vinsertf128(P[2 * u], P[2 * u], xmm8, 1);
                vinsertf128(P[2 * u + 1], P[2 * u + 1], xmm13, 0);
                vinsertf128(P[2 * u + 1], P[2 * u + 1], xmm10, 1);
            } else {
                loadTwV2(ymm10, ptr[loadBase + sa * strideBytes], sa, hasTw);
                loadTwV2(ymm11, ptr[loadBase + sb * strideBytes], sb, hasTw);
                vaddpd(P[2 * u], ymm10, ymm11);
                vsubpd(P[2 * u + 1], ymm10, ymm11);
            }
        }

        for (int st = 2; st <= nlf; ++st) {
            const int m = 1 << st;
            const int h = m >> 1;
            for (int blk = 0; blk < r / m; ++blk) {
                for (int i = 0; i < h; ++i) {
                    const Ymm &bot = P[blk * m + i];
                    const Ymm &top = P[blk * m + h + i];
                    if (i == 0) {
                        vsubpd(ymm12, bot, top);
                        vaddpd(bot, bot, top);
                        vmovapd(top, ymm12);
                    } else if (i == (m >> 2)) {
                        vshufpd(ymm12, top, top, 0x5);
                        vxorpd(ymm12, ymm12, ymm14);
                        vsubpd(top, bot, ymm12);
                        vaddpd(bot, bot, ymm12);
                    } else {
                        mulTwPoolV2(top, st, i);
                        vsubpd(top, bot, ymm12);
                        vaddpd(bot, bot, ymm12);
                    }
                }
            }
        }
    }

    void loadTwV2(const Xbyak::Ymm &dst, const Xbyak::Operand &addr,
                  int slot, bool hasTw) {
        using namespace Xbyak;
        if (slot == 0 || !hasTw) {
            vmovupd(dst, addr);
            return;
        }
        vmovupd(ymm8, addr);
        if (fma_) {
            vmovddup(ymm9, ymm8);
            vshufpd(dst, ymm8, ymm8, 0xF);
            vmulpd(dst, dst, ptr[r10 + slot * 64 + 32]);
            vfmaddsub231pd(dst, ymm9, ptr[r10 + slot * 64]);
        } else {
            vmovupd(ymm12, ptr[r10 + slot * 64]);
            vshufpd(ymm13, ymm12, ymm12, 0x5);
            vxorpd(ymm13, ymm13, ymm14);
            vmulpd(dst, ymm8, ymm12);
            vmulpd(ymm13, ymm8, ymm13);
            vhsubpd(dst, dst, ymm13);
        }
    }

    void mulTwPoolV2(const Xbyak::Ymm &top, int st, int e) {
        using namespace Xbyak;
        if (fma_) {
            vmovddup(ymm13, top);
            vshufpd(ymm12, top, top, 0xF);
            vmulpd(ymm12, ymm12, ptr[rip + twSwapLbl_[st][e]]);
            vfmaddsub231pd(ymm12, ymm13, ptr[rip + twLbl_[st][e]]);
        } else {
            vmovapd(ymm13, ptr[rip + twLbl_[st][e]]);
            vshufpd(ymm15, ymm13, ymm13, 0x5);
            vxorpd(ymm15, ymm15, ymm14);
            vmulpd(ymm12, top, ymm13);
            vmulpd(ymm15, top, ymm15);
            vhsubpd(ymm12, ymm12, ymm15);
        }
    }

    void cross16V2(const Xbyak::Reg64 &storeBase, int strideBytes) {
        using namespace Xbyak;
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};
        for (int j = 0; j < 8; ++j) {
            vmovupd(ymm8, ptr[rbp + spillOff_ + j * 32]);
            if (j == 0) {
                vaddpd(ymm10, ymm8, P[0]);
                vsubpd(ymm11, ymm8, P[0]);
            } else if (j == 4) {
                vshufpd(ymm12, P[4], P[4], 0x5);
                vxorpd(ymm12, ymm12, ymm14);
                vaddpd(ymm10, ymm8, ymm12);
                vsubpd(ymm11, ymm8, ymm12);
            } else {
                mulTwPoolV2(P[j], 4, j);
                vaddpd(ymm10, ymm8, ymm12);
                vsubpd(ymm11, ymm8, ymm12);
            }
            storeV2One(ymm10, j, storeBase, strideBytes, 16);
            storeV2One(ymm11, j + 8, storeBase, strideBytes, 16);
        }
    }

    void storeV2(int r, const Xbyak::Reg64 &storeBase, int strideBytes) {
        using namespace Xbyak;
        for (int p = 0; p < r; ++p) {
            storeV2One(Ymm(p), p, storeBase, strideBytes, r);
        }
    }

    void storeV2One(const Xbyak::Ymm &v, int pos,
                    const Xbyak::Reg64 &storeBase, int strideBytes, int r) {
        using namespace Xbyak;
        if (gatherTab_ != nullptr) {
            vextractf128(xmm12, v, 0);
            vmovupd(ptr[storeBase + pos * 16], xmm12);
            vextractf128(xmm12, v, 1);
            vmovupd(ptr[storeBase + (r + pos) * 16], xmm12);
            return;
        }
        const int off = pos * strideBytes;
        if (finalMode_ != 0) {
            fuseFinalPairInPlace(v, off);
            vmovupd(ptr[storeBase + off], v);
            return;
        }
        vmovupd(ptr[storeBase + off], v);
    }

    /* ---- single-k radix-32 leaf ---- */
    void emitLeaf32(const Xbyak::Reg64 &loadBase, int strideBytes,
                    bool hasTw, const Xbyak::Reg64 &storeBase) {
        using namespace Xbyak;
        vmovapd(ymm14, ptr[rip + sign2Lbl_]);
        emitSubLeaf16Sk(loadBase, strideBytes, hasTw, 0);
        for (int p = 0; p < 8; ++p) {
            vmovupd(ptr[rbp + spillOff_ + p * 32], Ymm(p));
        }
        emitSubLeaf16Sk(loadBase, strideBytes, hasTw, 1);
        for (int p = 0; p < 8; ++p) {
            vmovupd(ymm9, ptr[rbp + spillOff_ + p * 32]);
            if (fma_) {
                vmovddup(ymm11, Ymm(p));
                vshufpd(ymm8, Ymm(p), Ymm(p), 0xF);
                vmulpd(ymm8, ymm8, ptr[rip + w32SwapLbl_[p]]);
                vfmaddsub231pd(ymm8, ymm11, ptr[rip + w32Lbl_[p]]);
            } else {
                vmovupd(ymm10, ptr[rip + w32Lbl_[p]]);
                vshufpd(ymm12, ymm10, ymm10, 0x5);
                vxorpd(ymm12, ymm12, ymm14);
                vmulpd(ymm8, Ymm(p), ymm10);
                vmulpd(ymm12, Ymm(p), ymm12);
                vhsubpd(ymm8, ymm8, ymm12);
            }
            vaddpd(ymm13, ymm9, ymm8);
            vsubpd(ymm15, ymm9, ymm8);
            if (strideBytes == 16) {
                vmovupd(ptr[storeBase + (2 * p) * 16], ymm13);
                vmovupd(ptr[storeBase + (2 * p + 16) * 16], ymm15);
            } else {
                vextractf128(xmm8, ymm13, 0);
                if (finalMode_ != 0) {
                    fuseFinalScalar16(xmm8, (2 * p) * strideBytes);
                }
                vmovupd(ptr[storeBase + (2 * p) * strideBytes], xmm8);
                vextractf128(xmm8, ymm13, 1);
                if (finalMode_ != 0) {
                    fuseFinalScalar16(xmm8, (2 * p + 1) * strideBytes);
                }
                vmovupd(ptr[storeBase + (2 * p + 1) * strideBytes], xmm8);
                vextractf128(xmm8, ymm15, 0);
                if (finalMode_ != 0) {
                    fuseFinalScalar16(xmm8, (2 * p + 16) * strideBytes);
                }
                vmovupd(ptr[storeBase + (2 * p + 16) * strideBytes], xmm8);
                vextractf128(xmm8, ymm15, 1);
                if (finalMode_ != 0) {
                    fuseFinalScalar16(xmm8, (2 * p + 17) * strideBytes);
                }
                vmovupd(ptr[storeBase + (2 * p + 17) * strideBytes], xmm8);
            }
        }
    }

    void emitSubLeaf16Sk(const Xbyak::Reg64 &loadBase, int strideBytes,
                         bool hasTw, int par) {
        using namespace Xbyak;
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};

        for (int p = 0; p < 8; ++p) {
            const int sa = 2 * (int)bitrev((uint32_t)(2 * p), 4) + par;
            const int sb = 2 * (int)bitrev((uint32_t)(2 * p + 1), 4) + par;
            if (gatherTab_ != nullptr) {
                const Reg64 &tab = *gatherTab_;
                mov(eax, dword[tab + sa * 4]);
                vmovupd(xmm8, ptr[rdi + rax]);
                if (loadFuse_) {
                    fuseLoadConjMul(xmm8, rax);
                } else if (entryConj_) { /* topic jit-anyN-fwd-inv:
                 * inverse entry boundary conj — the mode-3 16B sign
                 * flip with no conv operator (DESIGN L-Q degeneracy) */
                        vxorpd(xmm8, xmm8, ptr[rip + sign16Lbl_]);
                }
                mov(eax, dword[tab + sb * 4]);
                vmovupd(xmm9, ptr[rdi + rax]);
                if (loadFuse_) {
                    fuseLoadConjMul(xmm9, rax);
                } else if (entryConj_) { /* topic jit-anyN-fwd-inv:
                 * inverse entry boundary conj — the mode-3 16B sign
                 * flip with no conv operator (DESIGN L-Q degeneracy) */
                        vxorpd(xmm9, xmm9, ptr[rip + sign16Lbl_]);
                }
            } else {
                vmovupd(xmm8, ptr[loadBase + sa * strideBytes]);
                vmovupd(xmm9, ptr[loadBase + sb * strideBytes]);
            }
            if (hasTw && sa != 0) {
                mulTwXmm(xmm10, xmm8, sa);
            }
            if (hasTw && sb != 0) {
                mulTwXmm(xmm12, xmm9, sb);
            }
            const Xmm &a2 = (hasTw && sa != 0) ? xmm10 : xmm8;
            const Xmm &b2 = (hasTw && sb != 0) ? xmm12 : xmm9;
            vaddpd(xmm13, a2, b2);
            vsubpd(xmm15, a2, b2);
            vinsertf128(P[p], P[p], xmm13, 0);
            vinsertf128(P[p], P[p], xmm15, 1);
        }

        for (int blk = 0; blk < 4; ++blk) {
            const Ymm &bot = P[2 * blk];
            const Ymm &top = P[2 * blk + 1];
            vshufpd(ymm8, top, top, 0x5);
            vxorpd(ymm8, ymm8, ymm14);
            vblendpd(ymm8, top, ymm8, 0x0C);
            vsubpd(top, bot, ymm8);
            vaddpd(bot, bot, ymm8);
        }

        for (int st = 3; st <= 4; ++st) {
            const int halfPairs = 1 << (st - 2);
            const int pairsPerBlock = 1 << (st - 1);
            for (int blk = 0; blk < 16 / (1 << st); ++blk) {
                const int bp = blk * pairsPerBlock;
                for (int q = 0; q < halfPairs; ++q) {
                    const Ymm &bot = P[bp + q];
                    const Ymm &top = P[bp + halfPairs + q];
                    if (fma_) {
                        vmovddup(ymm11, top);
                        vshufpd(ymm8, top, top, 0xF);
                        vmulpd(ymm8, ymm8, ptr[rip + lwSwapLbl_[st][q]]);
                        vfmaddsub231pd(ymm8, ymm11, ptr[rip + lwLbl_[st][q]]);
                    } else {
                        vmovapd(ymm10, ptr[rip + lwLbl_[st][q]]);
                        vshufpd(ymm12, ymm10, ymm10, 0x5);
                        vxorpd(ymm12, ymm12, ymm14);
                        vmulpd(ymm8, top, ymm10);
                        vmulpd(ymm12, top, ymm12);
                        vhsubpd(ymm8, ymm8, ymm12);
                    }
                    vsubpd(top, bot, ymm8);
                    vaddpd(bot, bot, ymm8);
                }
            }
        }
    }

    void mulTwXmm(const Xbyak::Xmm &dst, const Xbyak::Xmm &d, int slot) {
        using namespace Xbyak;
        if (fma_) {
            vmovddup(xmm11, d);
            vshufpd(dst, d, d, 3);
            vmulpd(dst, dst, ptr[r10 + slot * 32 + 16]);
            vfmaddsub231pd(dst, xmm11, ptr[r10 + slot * 32]);
        } else {
            vmulpd(dst, d, ptr[r10 + slot * 32]);
            vmulpd(xmm11, d, ptr[r10 + slot * 32 + 16]);
            vhsubpd(dst, dst, xmm11);
        }
    }

    void emitPool() {
        using namespace Xbyak;
        align(32);
        L(sign2Lbl_);
        dq(0);
        dq(0x8000000000000000ull);
        dq(0);
        dq(0x8000000000000000ull);

        align(16);
        L(sign16Lbl_);
        dq(0);
        dq(0x8000000000000000ull);

        const double inv = 1.0 / (double)N_;
        align(32);
        L(invNLbl_);
        dq(bits_from_double(inv));
        dq(bits_from_double(-inv));
        dq(bits_from_double(inv));
        dq(bits_from_double(-inv));

        for (int s = 3; s <= 4; ++s) {
            for (int e = 1; e < (1 << (s - 1)); ++e) {
                if (e == (1 << (s - 2))) {
                    continue;
                }
                const double ang =
                    -2.0 * kPi * (double)e / (double)(1 << s);
                const double wr = std::cos(ang);
                const double wi = std::sin(ang);
                align(32);
                L(twLbl_[s][e]);
                dq(bits_from_double(wr));
                dq(bits_from_double(wi));
                dq(bits_from_double(wr));
                dq(bits_from_double(wi));
                align(32);
                L(twSwapLbl_[s][e]);
                dq(bits_from_double(wi));
                dq(bits_from_double(wr));
                dq(bits_from_double(wi));
                dq(bits_from_double(wr));
            }
        }

        for (int s = 3; s <= 4; ++s) {
            const int halfPairs = 1 << (s - 2);
            for (int q = 0; q < halfPairs; ++q) {
                double c[2], sn[2];
                for (int l = 0; l < 2; ++l) {
                    const int e = 2 * q + l;
                    const double ang =
                        -2.0 * kPi * (double)e / (double)(1 << s);
                    c[l] = std::cos(ang);
                    sn[l] = std::sin(ang);
                }
                align(32);
                L(lwLbl_[s][q]);
                dq(bits_from_double(c[0]));
                dq(bits_from_double(sn[0]));
                dq(bits_from_double(c[1]));
                dq(bits_from_double(sn[1]));
                align(32);
                L(lwSwapLbl_[s][q]);
                dq(bits_from_double(sn[0]));
                dq(bits_from_double(c[0]));
                dq(bits_from_double(sn[1]));
                dq(bits_from_double(c[1]));
            }
        }

        for (int p = 0; p < 8; ++p) {
            double c[2], sn[2];
            for (int l = 0; l < 2; ++l) {
                const int e = 2 * p + l;
                const double ang = -2.0 * kPi * (double)e / 32.0;
                c[l] = std::cos(ang);
                sn[l] = std::sin(ang);
            }
            align(32);
            L(w32Lbl_[p]);
            dq(bits_from_double(c[0]));
            dq(bits_from_double(sn[0]));
            dq(bits_from_double(c[1]));
            dq(bits_from_double(sn[1]));
            align(32);
            L(w32SwapLbl_[p]);
            dq(bits_from_double(sn[0]));
            dq(bits_from_double(c[0]));
            dq(bits_from_double(sn[1]));
            dq(bits_from_double(c[1]));
        }
    }

    static uint64_t bits_from_double(double d) {
        uint64_t u;
        __builtin_memcpy(&u, &d, sizeof(u));
        return u;
    }

    bool fma_;
    int N_;
    bool conv_;
    bool loadFusion_;
    const double *scratch_;
    const double *hcNat_;
    const weld_stage_desc *dPermute_;
    const weld_stage_desc *dConjMul_;
    const weld_stage_desc *dInvN_;
    double *scratch2_;
    int spillOff_ = -304;
    int finalMode_ = 0;
    bool loadFuse_ = false;
    const Xbyak::Reg64 *gatherTab_ = nullptr; /* leaf gather-mode flag */
    const uint32_t *tab_ = nullptr;           /* digitrev gather table */
    const Stage *stages_ = nullptr;
    int kStages_ = 0;
    /* topic jit-anyN-fwd-inv (DELTA F1): inverse single-transform pass
     * flags (entry conj at the stage-0 gather loads; ctor comment). */
    bool invSingle_ = false;
    bool entryConj_ = false;
    Xbyak::Label sign2Lbl_;
    Xbyak::Label sign16Lbl_;
    Xbyak::Label invNLbl_;
    Xbyak::Label twLbl_[5][8];
    Xbyak::Label twSwapLbl_[5][8];
    Xbyak::Label lwLbl_[5][4];
    Xbyak::Label lwSwapLbl_[5][4];
    Xbyak::Label w32Lbl_[8];
    Xbyak::Label w32SwapLbl_[8];
};

/* ---- topic jit-codelet-weld: plan-time build + search ----------------
 *
 * weldBuildKernel materializes, for one decomposition: the general
 * mixed-radix digitrev table, the own-leaf TW rows (promoted layout), the
 * t1fv W tables / TW sweep tables for codelet stages, the codelet stride
 * index arrays, the weld_stage_desc descriptors and the scratch (+ slack
 * for odd-vl padding lanes).  The decomposition search is the promoted
 * enumerate-and-time shape (R2 framework): candidates over the extended
 * factor domain {2,3,4,5,6,7,8,9,10,12,13,15,16,20,25,32}, per-stage
 * leaf-dispatch viability filtered (pow2 factors -> own vector leaves,
 * 3/5-family factors -> codelet leaves, search adjudicated), model-ranked,
 * budget-guarded short-timed.  Env knobs (topic-local, no public API):
 *   FFT_WELD=off                    disable the weld route entirely
 *   FFT_WELD_SEARCH=off|0           skip the search (greedy default)
 *   FFT_WELD_ORDER=r1,r2,...        force a decomposition (verification)
 *   FFT_WELD_LOG=1                  stderr plan diagnostics
 */
struct WeldBuilt {
    SchedEmitterW *em;
    double *arena;    /* tables + descriptors (owned or borrowed) */
    size_t arena_bytes;
    double *scratch;  /* stage-0 ping buffer + slack (owned or borrowed) */
    double *scratch2; /* N1_ODD final out-buffer (owned) */
    bool ownsArena;
    bool ownsScratch;
};

/* per-stage leaf dispatch + viability (the plan-time adjudication input) */
static bool weld_stage_kind(int t, int /*kStages*/, int r, int Bp,
                            int nblocks, int *kind) {
    if (t == 0) {
        const int nb = nblocks; /* N / r0 */
        if (r == 2) {
            *kind = (nb % 2 == 0) ? WSK_OWN_V2 : WSK_OWN_S2;
            return true;
        }
        if (r == 4 || r == 8 || r == 16) {
            if (nb % 2 != 0) {
                return false;
            }
            *kind = WSK_OWN_V2;
            return true;
        }
        if (r == 32) {
            *kind = WSK_OWN_32;
            return true;
        }
        if (weld_clet_n1(r) != nullptr) {
            *kind = WSK_N1_CONTIG;
            return true;
        }
        return false;
    }
    if (r == 32) {
        *kind = WSK_OWN_32;
        return true;
    }
    if (r <= 16 && is_pow2(r)) {
        if (Bp % 2 != 0) {
            return false; /* vector-2 k-pairing needs an even Bp */
        }
        *kind = WSK_OWN_V2;
        return true;
    }
    if (Bp % 2 == 0 && weld_clet_t1(r) != nullptr) {
        *kind = WSK_T1;
        return true;
    }
    if (weld_clet_n1(r) != nullptr) {
        /* odd-Bp codelet final on a caller-owned out buffer gets routed
         * through scratch2 + copy by the builder (padding never leaves
         * our buffers); the model carries the extra pass. */
        *kind = (Bp % 2 != 0) ? WSK_N1_ODD : WSK_N1;
        return true;
    }
    return false;
}

static size_t weld_pad32(size_t v) {
    return (v + 31u) & ~31u;
}

/* arena bytes for one decomposition: mirrors the builder's cursor
 * advances exactly (tables + stride index arrays + 24 descriptors). */
static size_t weld_arena_bytes(int N, const int *radices, int kStages) {
    int B[8];
    B[0] = radices[0];
    for (int t = 1; t < kStages; ++t) {
        B[t] = B[t - 1] * radices[t];
    }
    size_t cur = weld_pad32((size_t)N * 4); /* gatherTab */
    for (int t = 0; t < kStages; ++t) {
        const int r = radices[t];
        const int Bp = (t == 0) ? 1 : B[t - 1];
        int kind = 0;
        weld_stage_kind(t, kStages, r, Bp,
                        (t == 0) ? (N / B[0]) : (N / B[t]), &kind);
        if (t == 0) {
            if (kind == WSK_N1_CONTIG) {
                cur += weld_pad32((size_t)r * 3 * sizeof(long));
            }
            continue; /* own stage-0 leaves carry no tables */
        }
        if (kind == WSK_OWN_V2) {
            cur += weld_pad32((size_t)(Bp / 2) * (size_t)(r * 64));
        } else if (kind == WSK_OWN_32) {
            cur += weld_pad32((size_t)Bp * (size_t)(r * 32));
        } else if (kind == WSK_T1) {
            cur += weld_pad32((size_t)r * 3 * sizeof(long));
            cur += weld_pad32(weld_t1_w_doubles(Bp, r) * 8);
        } else if (Bp % 2 == 0) { /* WSK_N1 sweep (pair layout) */
            cur += weld_pad32((size_t)r * 3 * sizeof(long));
            cur += weld_pad32((size_t)(r - 1) * (size_t)(Bp / 2) * 64);
        } else { /* WSK_N1_ODD sweep (scalar layout) */
            cur += weld_pad32((size_t)r * 3 * sizeof(long));
            cur += weld_pad32((size_t)(r - 1) * (size_t)Bp * 32);
        }
    }
    cur += weld_pad32(24 * sizeof(weld_stage_desc));
    return weld_pad64(cur + 64);
}

static double weld_slot_model(int kind, int radix, int Bp, int pos, int k) {
    bool stage0 = (pos == 0);
    bool final = (pos == k - 1) && !stage0;
    double base;
    switch (radix) {
    case 2:
        base = stage0 ? 5.6 : final ? 5.8 : 5.4;
        break;
    case 3:
        base = 6.1;
        break;
    case 4:
        base = stage0 ? 6.0 : final ? 6.3 : 5.8;
        break;
    case 5:
        base = 6.6;
        break;
    case 6:
        base = 6.9;
        break;
    case 7:
        base = 7.2;
        break;
    case 8:
        base = stage0 ? 6.4 : final ? 6.8 : 6.0;
        break;
    case 9:
        base = 7.5;
        break;
    case 10:
        base = 7.7;
        break;
    case 11:
        /* R1 (F3/L-S): vendored SIMD n1fv_11 (N1+TW sweep) — the R0
         * scalar-leaf penalty 9.6 is retired with the hand leaf; modelled
         * on the machine-leaf progression (between 10 and 12). */
        base = 7.9;
        break;
    case 12:
        base = 8.1;
        break;
    case 13:
        base = 8.3;
        break;
    case 14:
        /* R1 (F3/L-S): vendored SIMD n1fv_14 (N1+TW sweep, composite
         * 2x7); modelled between 13 and 15 on the machine progression. */
        base = 8.45;
        break;
    case 15:
        base = 8.6;
        break;
    case 16:
        base = stage0 ? 10.0 : final ? 9.5 : 7.8;
        break;
    case 20:
        base = 9.4;
        break;
    case 25:
        base = 10.2;
        break;
    default: /* 32, single-k */
        base = stage0 ? 11.5 : final ? 15.5 : 15.0;
        break;
    }
    double cost = base;
    if (kind == WSK_OWN_S2) {
        cost += 2.0; /* unpaired scalar gather */
    } else if (kind == WSK_T1) {
        cost = cost * 1.02 + 0.15;
    } else if (kind == WSK_N1 || kind == WSK_N1_ODD) {
        /* + separate twiddle sweep over (r-1)(Bp-1)/(r*Bp) of the array
         * (~2 element passes per unit fraction, modelled at 5/pass) */
        const double frac = (double)(radix - 1) * (double)(Bp - 1) /
                            ((double)radix * (double)Bp);
        cost += frac * 10.0 + 0.15;
        if (kind == WSK_N1_ODD) {
            cost += 1.0; /* tail-call machinery + slack padding writes */
        }
    } else if (kind == WSK_N1_CONTIG) {
        cost += 10.0; /* digitrev permutation pre-pass */
    }
    if (final && !(kind == WSK_OWN_V2 || kind == WSK_OWN_32)) {
        cost += 5.0;       /* INVN boundary sweep */
        if (pos > 0) {
            /* CONJMUL sweep unless half B's stage-0 load fusion covers it
             * (own stage 0 present) */
        }
    }
    return cost;
}

/* Build tables + descriptors + SchedEmitterW for one decomposition.
 * conv=0 emits the forward-only kernel (H materialization / search
 * timing); conv=1 the welded convolution pair of halves.  outIsCaller:
 * final-stage outputs land in the caller's io (odd-tail N1 finals are
 * routed through scratch2 + copy so padding never leaves our buffers).
 *
 * topic jit-anyN-fwd-inv (DELTA F1): invSingle=1 (with conv=0) emits the
 * INVERSE single-transform kernel — the half-B boundary forms without the
 * conv operators: entry conj at the stage-0 gather loads / conjugating
 * PERMUTE, exit conj(.)·invN at the final store (mode 2) or the INVN
 * sweep on a codelet final. */
static WeldBuilt weldBuildKernel(int N, const int *radices, int kStages,
                                 bool hasFma, bool conv, bool outIsCaller,
                                 const double *hMul, const double *hSwap,
                                 const double *hcNat, double *borrowScratch,
                                 double *borrowArena, bool invSingle = false) {
    WeldBuilt b;
    b.em = nullptr;
    b.arena = nullptr;
    b.arena_bytes = 0;
    b.scratch = nullptr;
    b.scratch2 = nullptr;
    b.ownsArena = (borrowArena == nullptr);
    b.ownsScratch = (borrowScratch == nullptr);
    if (kStages < 2) {
        throw std::bad_alloc(); /* the weld needs >= 2 stages */
    }

    int B[8];
    B[0] = radices[0];
    for (int t = 1; t < kStages; ++t) {
        B[t] = B[t - 1] * radices[t];
    }

    /* stage kinds (search-adjudicated dispatch; re-derived identically
     * here and in the arena sizing) */
    int kinds[8];
    for (int t = 0; t < kStages; ++t) {
        const int nblocks = (t == 0) ? (N / B[0]) : (N / B[t]);
        if (!weld_stage_kind(t, kStages, radices[t],
                             (t == 0) ? 1 : B[t - 1], nblocks, &kinds[t])) {
            throw std::bad_alloc(); /* candidate filtered earlier */
        }
    }

    /* general mixed-radix inverse digit-reversal table */
    std::vector<uint32_t> gatherTab((size_t)N);
    for (int v = 0; v < N; ++v) {
        int e[8];
        int vv = v;
        for (int i = 0; i < kStages; ++i) {
            e[i] = vv % radices[kStages - 1 - i];
            vv /= radices[kStages - 1 - i];
        }
        int p = 0;
        for (int i = 0; i < kStages; ++i) {
            int m = 1;
            for (int l = 0; l < kStages - 1 - i; ++l) {
                m *= radices[l];
            }
            p += e[i] * m;
        }
        gatherTab[(size_t)p] = (uint32_t)(v * 16);
    }

    /* own-leaf TW rows + codelet tables, laid out in one arena */
    b.arena_bytes = weld_arena_bytes(N, radices, kStages);
    b.arena = borrowArena != nullptr
                  ? borrowArena
                  : static_cast<double *>(aligned_alloc(
                        64,
                        weld_pad64(b.arena_bytes > 64 ? b.arena_bytes
                                                      : 64)));
    if (b.arena == nullptr) {
        throw std::bad_alloc();
    }
    char *cur = reinterpret_cast<char *>(b.arena);
    {
        uint32_t *tabPtr = reinterpret_cast<uint32_t *>(cur);
        for (int i = 0; i < N; ++i) {
            tabPtr[i] = gatherTab[(size_t)i];
        }
        cur += weld_pad32((size_t)N * 4);
    }

    SchedEmitterW::Stage stages[8];
    /* descriptors live in the arena tail (kernel lifetime — the emitter
     * bakes their addresses into the machine code); weld_arena_bytes
     * reserves [tables .. tables+descSection+64) so this offset is
     * exactly where the table cursor lands after the stage loop. */
    const size_t descOff =
        b.arena_bytes - 64 - weld_pad32(24 * sizeof(weld_stage_desc));
    weld_stage_desc *descs =
        reinterpret_cast<weld_stage_desc *>(reinterpret_cast<char *>(b.arena) +
                                            descOff);
    int nd = 0;

    /* slack sizing: odd-vl tail padding lanes stay inside our buffers */
    size_t slackDoubles = 64;
    bool needScratch2 = false;
    for (int t = 0; t < kStages; ++t) {
        if (kinds[t] == WSK_N1_CONTIG && (N / radices[0]) % 2 != 0) {
            slackDoubles += (size_t)radices[0] * 2 + 8;
        }
        if (kinds[t] == WSK_N1_ODD) {
            /* every odd-Bp stage closes each block with per-instance
             * vl=1 tail calls whose padding lane writes (r-1)*Bp
             * elements into slack — size for it unconditionally */
            slackDoubles += (size_t)B[t] * 2 + 8;
            if (t == kStages - 1 && outIsCaller && (N / B[t]) % 2 != 0) {
                /* odd block count on a caller-style out buffer (conv OR
                 * forward): the tail's own instance row would run past
                 * the array — route the whole final through scratch2 +
                 * copy */
                needScratch2 = true;
            }
        }
    }

    for (int t = 0; t < kStages; ++t) {
        stages[t].kind = kinds[t];
        stages[t].radix = radices[t];
        stages[t].bprev = (t == 0) ? 1 : B[t - 1];
        stages[t].b = B[t];
        stages[t].tw = nullptr;
        stages[t].desc = nullptr;
        stages[t].tws = nullptr;

        if (kinds[t] == WSK_OWN_V2 || kinds[t] == WSK_OWN_32) {
            if (t == 0) {
                continue; /* no twiddles at stage 0 */
            }
            const int r = radices[t];
            const int Bp = B[t - 1];
            const bool v2 = (kinds[t] == WSK_OWN_V2);
            const int rowBytes = v2 ? r * 64 : r * 32;
            const int rows = v2 ? Bp / 2 : Bp;
            double *dst = reinterpret_cast<double *>(cur);
            const double zr = std::cos(-2.0 * kPi / (double)B[t]);
            const double zi = std::sin(-2.0 * kPi / (double)B[t]);
            for (int row = 0; row < rows; ++row) {
                const int k = v2 ? 2 * row : row;
                const double ang = -2.0 * kPi * (double)k / (double)B[t];
                const double br = std::cos(ang);
                const double bi = std::sin(ang);
                double b2r = 1.0, b2i = 0.0;
                if (v2) {
                    b2r = br * zr - bi * zi;
                    b2i = br * zi + bi * zr;
                }
                double wr = 1.0, wi = 0.0;
                double w2r = 1.0, w2i = 0.0;
                double *slot = dst + (size_t)row * (size_t)(rowBytes / 8);
                for (int j = 0; j < r; ++j) {
                    if (v2) {
                        slot[j * 8 + 0] = wr;
                        slot[j * 8 + 1] = wi;
                        slot[j * 8 + 2] = w2r;
                        slot[j * 8 + 3] = w2i;
                        slot[j * 8 + 4] = wi;
                        slot[j * 8 + 5] = wr;
                        slot[j * 8 + 6] = w2i;
                        slot[j * 8 + 7] = w2r;
                    } else {
                        slot[j * 4 + 0] = wr;
                        slot[j * 4 + 1] = wi;
                        slot[j * 4 + 2] = wi;
                        slot[j * 4 + 3] = hasFma ? wr : -wr;
                    }
                    const double nwr = wr * br - wi * bi;
                    const double nwi = wr * bi + wi * br;
                    wr = nwr;
                    wi = nwi;
                    if (v2) {
                        const double n2r = w2r * b2r - w2i * b2i;
                        const double n2i = w2r * b2i + w2i * b2r;
                        w2r = n2r;
                        w2i = n2i;
                    }
                }
            }
            stages[t].tw = dst;
            cur += weld_pad32((size_t)rows * (size_t)rowBytes);
            continue;
        }

        /* codelet stage: stride index arrays + W / sweep tables + desc */
        const int r = radices[t];
        const int Bp = stages[t].bprev;
        const int Bt = B[t];
        INT *isArr = reinterpret_cast<INT *>(cur);
        INT *osArr = isArr + r;
        INT *rsArr = osArr + r;
        const int elemStride = (kinds[t] == WSK_N1_CONTIG)
                                   ? 2 /* contiguous r-run elements */
                                   : 2 * Bp;
        for (int j = 0; j < r; ++j) {
            isArr[j] = (INT)(elemStride * j);
            osArr[j] = (INT)(elemStride * j);
            rsArr[j] = (INT)(2 * Bp * j);
        }
        cur += weld_pad32((size_t)r * 3 * sizeof(INT));

        weld_stage_desc &d = descs[nd];
        memset(&d, 0, sizeof(d));
        d.radix = r;
        d.bp = Bp;
        d.bstep = (kinds[t] == WSK_N1_CONTIG) ? 0 : (long)(2 * Bt);
        d.nblocks = (kinds[t] == WSK_N1_CONTIG) ? 0 : (long)(N / Bt);
        d.is = isArr;
        d.os = osArr;
        d.rs = rsArr;

        if (kinds[t] == WSK_T1) {
            double *W = reinterpret_cast<double *>(cur);
            weld_t1_w_fill(W, Bp, r);
            cur += weld_pad32(weld_t1_w_doubles(Bp, r) * 8);
            d.kind = WELD_RUN_T1;
            d.t1 = weld_clet_t1(r);
            d.W = W;
            d.vl = Bp; /* me bound */
        } else if (kinds[t] == WSK_N1_CONTIG) {
            d.kind = WELD_RUN_N1_CONTIG;
            d.n1 = weld_clet_n1(r);
            d.vl = N / r;
            d.ivs = 2 * r;
            d.ovs = 2 * r;
            /* slack patched below (needs the scratch pointer) */
        } else {
            /* WSK_N1 / WSK_N1_ODD: twiddle sweep table + codelet sweep */
            double *TW = reinterpret_cast<double *>(cur);
            if (Bp % 2 == 0) {
                /* pair layout: slot(j,kp) at ((j-1)*pairs + kp)*8:
                 * [W^<jk>, W^<j(k+1)> | swap twin] (plan-time direct trig;
                 * execute touches no transcendentals) */
                const int pairs = Bp / 2;
                for (int j = 1; j < r; ++j) {
                    for (int kp = 0; kp < pairs; ++kp) {
                        double *slot =
                            TW + ((size_t)(j - 1) * pairs + kp) * 8;
                        const double th0 = -2.0 * kPi *
                                           (double)(j * (2 * kp)) /
                                           (double)Bt;
                        const double th1 = -2.0 * kPi *
                                           (double)(j * (2 * kp + 1)) /
                                           (double)Bt;
                        const double c0 = std::cos(th0);
                        const double s0 = std::sin(th0);
                        const double c1 = std::cos(th1);
                        const double s1 = std::sin(th1);
                        slot[0] = c0;
                        slot[1] = s0;
                        slot[2] = c1;
                        slot[3] = s1;
                        slot[4] = s0; /* swap twin */
                        slot[5] = c0;
                        slot[6] = s1;
                        slot[7] = c1;
                    }
                }
                cur += weld_pad32((size_t)(r - 1) * (size_t)pairs * 64);
                weld_stage_desc &tw = descs[++nd];
                memset(&tw, 0, sizeof(tw));
                tw.kind = WELD_RUN_TWS_V2;
                tw.radix = r;
                tw.bp = Bp;
                tw.bstep = (long)(2 * Bt);
                tw.nblocks = (long)(N / Bt);
                tw.W = TW;
                tw.twPairs = Bp / 2;
                stages[t].tws = &tw;
            } else {
                /* scalar layout: slot(j,k) at ((j-1)*Bp + k)*4:
                 * (wr, wi, wi, wr) */
                for (int j = 1; j < r; ++j) {
                    for (int k = 0; k < Bp; ++k) {
                        const double ang = -2.0 * kPi *
                                           (double)(j * k) / (double)Bt;
                        double *slot = TW + (size_t)((j - 1) * Bp + k) * 4;
                        slot[0] = std::cos(ang);
                        slot[1] = std::sin(ang);
                        slot[2] = slot[1];
                        slot[3] = slot[0];
                    }
                }
                cur += weld_pad32((size_t)(r - 1) * (size_t)Bp * 32);
                weld_stage_desc &tw = descs[++nd];
                memset(&tw, 0, sizeof(tw));
                tw.kind = WELD_RUN_TWS_SC;
                tw.radix = r;
                tw.bp = Bp;
                tw.bstep = (long)(2 * Bt);
                tw.nblocks = (long)(N / Bt);
                tw.W = TW;
                stages[t].tws = &tw;
            }
            d.kind = (Bp % 2 != 0) ? WELD_RUN_N1_ODD : WELD_RUN_N1;
            d.n1 = weld_clet_n1(r);
            d.vl = Bp;
            d.ivs = 2;
            d.ovs = 2;
            d.base_off = 2 * (Bp - 1);
        }
        stages[t].desc = &d; /* NB: the twiddle-sweep branch advanced nd
                              * above — the stage desc is d itself */
        ++nd;
        if (nd >= 22) {
            throw std::bad_alloc(); /* descriptor budget (8 stages x2) */
        }
    }

    /* scratch (+ slack); the digitrev-permutation descriptor */
    if (borrowScratch != nullptr) {
        b.scratch = borrowScratch;
    } else {
        b.scratch = static_cast<double *>(aligned_alloc(
            64, weld_pad64(((size_t)N * 2 + slackDoubles) * 8)));
        if (b.scratch == nullptr) {
            if (b.ownsArena) {
                free(b.arena);
            }
            b.arena = nullptr;
            throw std::bad_alloc();
        }
    }
    double *slack = b.scratch + 2 * (size_t)N;
    if (needScratch2) {
        b.scratch2 = static_cast<double *>(
            aligned_alloc(64, weld_pad64(((size_t)N * 2 + slackDoubles) * 8)));
        if (b.scratch2 == nullptr) {
            if (b.ownsScratch) {
                free(b.scratch);
            }
            if (b.ownsArena) {
                free(b.arena);
            }
            throw std::bad_alloc();
        }
    }

    /* patch the scratch-relative slack pointer into every descriptor
     * whose runner steers padding-lane stores (contig tail + odd-Bp
     * per-instance tails) */
    for (int t = 0; t < kStages; ++t) {
        weld_stage_desc *sd = const_cast<weld_stage_desc *>(stages[t].desc);
        if (sd != nullptr && (sd->kind == WELD_RUN_N1_ODD ||
                              sd->kind == WELD_RUN_N1_CONTIG)) {
            sd->slack = slack; /* arena-owned descriptor, kernel lifetime */
        }
    }

    weld_stage_desc *dPermute = nullptr;
    if (kinds[0] == WSK_N1_CONTIG) {
        weld_stage_desc &dp = descs[nd];
        memset(&dp, 0, sizeof(dp));
        dp.kind = WELD_RUN_PERMUTE;
        dp.tab = reinterpret_cast<const uint32_t *>(b.arena);
        dp.n = N;
        dp.conj = invSingle ? 1 : 0; /* topic jit-anyN-fwd-inv: inverse
                                        entry boundary conj in the copy */
        dPermute = &dp;
        ++nd;
    }

    weld_stage_desc *dConjMul = nullptr;
    weld_stage_desc *dInvN = nullptr;
    const bool finalCodelet =
        !(kinds[kStages - 1] == WSK_OWN_V2 ||
          kinds[kStages - 1] == WSK_OWN_32);
    const bool stage0Own =
        (kinds[0] == WSK_OWN_V2 || kinds[0] == WSK_OWN_32 ||
         kinds[0] == WSK_OWN_S2);
    const bool loadFusion = conv && finalCodelet && stage0Own;
    /* topic jit-anyN-fwd-inv: the inverse single pass needs the INVN
     * boundary sweep on a codelet final (exit conj·invN) but never the
     * CONJMUL half-A operator. */
    if (finalCodelet && (conv || invSingle)) {
        if (conv && !stage0Own) {
            weld_stage_desc &dc = descs[nd];
            memset(&dc, 0, sizeof(dc));
            dc.kind = WELD_RUN_CONJMUL;
            dc.n = N;
            dc.hMul = hMul;
            dc.hSwap = hSwap;
            dConjMul = &dc;
            ++nd;
        }
        weld_stage_desc &di = descs[nd];
        memset(&di, 0, sizeof(di));
        di.kind = WELD_RUN_INVN;
        di.n = N;
        di.invn = 1.0 / (double)N;
        dInvN = &di;
        ++nd;
        /* odd-tail N1 final on the caller buffer: the emitter routes the
         * sweep into scratch2 and copies out (no descriptor needed) */
    }
    if (nd >= 24) {
        throw std::bad_alloc();
    }

    try {
        b.em = new SchedEmitterW(N,
                                 reinterpret_cast<const uint32_t *>(
                                     b.arena),
                                 b.scratch, stages, kStages, hasFma, conv,
                                 loadFusion, hMul, hSwap, hcNat, dPermute,
                                 dConjMul, dInvN, b.scratch2, invSingle);
    } catch (...) {
        free(b.scratch2);
        b.scratch2 = nullptr;
        if (b.ownsScratch) {
            free(b.scratch);
        }
        b.scratch = nullptr;
        if (b.ownsArena) {
            free(b.arena);
        }
        b.arena = nullptr;
        throw;
    }
    return b;
}


/* ---- weld search: enumerate-and-time over the extended factor domain
 * (promoted R2 framework shape; candidates viability-filtered by the
 * per-stage leaf dispatch, model-ranked, budget-guarded). -------------- */
struct WeldCand {
    int rad[8];
    int k;
    double model;
};

/* topic jit-anyN-fwd-inv R1 (DELTA F3, L-S): 11 = the VENDORED upstream
 * n1fv_11 (replaces the retired R0 hand pair; hand reachable again via
 * FFT_ANYN_LEAVES=hand), 14 = vendored upstream n1fv_14 — both walk the
 * WSK_N1 n1fv+TW-sweep mode like radix 13 (no t1fv_11/t1fv_14 upstream).
 * Coverage set S = this list's prime closure {2,3,5,7,11,13} (14 = 2x7
 * adds no new prime; draft {#API-FFT-013}). */
static const int weld_factors[] = {2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                   12, 13, 14, 15, 16, 20, 25, 32};
static const int weld_n_factors =
    (int)(sizeof(weld_factors) / sizeof(weld_factors[0]));

static bool weld_viable(int N, const int *rad, int k) {
    if (k < 2 || k > 8) {
        return false;
    }
    int B[8];
    B[0] = rad[0];
    for (int t = 1; t < k; ++t) {
        B[t] = B[t - 1] * rad[t];
    }
    if (B[k - 1] != N) {
        return false;
    }
    int kind = 0;
    for (int t = 0; t < k; ++t) {
        const int nblocks = (t == 0) ? (N / B[0]) : (N / B[t]);
        if (!weld_stage_kind(t, k, rad[t], (t == 0) ? 1 : B[t - 1], nblocks,
                             &kind)) {
            return false;
        }
    }
    return true;
}

static double weld_model_of(int N, const int *rad, int k) {
    int B[8];
    B[0] = rad[0];
    for (int t = 1; t < k; ++t) {
        B[t] = B[t - 1] * rad[t];
    }
    double m = 0.0;
    for (int t = 0; t < k; ++t) {
        int kind = 0;
        const int nblocks = (t == 0) ? (N / B[0]) : (N / B[t]);
        weld_stage_kind(t, k, rad[t], (t == 0) ? 1 : B[t - 1], nblocks,
                        &kind);
        m += weld_slot_model(kind, rad[t], (t == 0) ? 1 : B[t - 1], t, k);
    }
    return m;
}

static void weld_gen(int rem, int *cur, int len, int N,
                     std::vector<WeldCand> &out) {
    if (rem == 1) {
        if (len < 2) {
            return; /* the weld needs >= 2 stages */
        }
        WeldCand c;
        for (int i = 0; i < len; ++i) {
            c.rad[i] = cur[i];
        }
        c.k = len;
        c.model = 0.0;
        if (weld_viable(N, c.rad, c.k)) {
            out.push_back(c);
        }
        return;
    }
    if (len >= 8) {
        return;
    }
    for (int fi = 0; fi < weld_n_factors; ++fi) {
        const int r = weld_factors[fi];
        if (rem % r == 0) {
            cur[len] = r;
            weld_gen(rem / r, cur, len + 1, N, out);
        }
    }
}

/* greedy anchor: peal the largest useful factor (prefer a small pow2 at
 * the head for the fused gather, composites descending after) */
static void weld_anchor(int N, bool pow2First, int *rad, int *kOut) {
    int rem = N;
    int k = 0;
    /* head: smallest pow2 > 1 that keeps viability (2 unless 4|N) */
    if (pow2First && rem % 2 == 0) {
        int head = (rem % 4 == 0) ? 4 : 2;
        if (rem / head > 1) {
            rad[k++] = head;
            rem /= head;
        }
    }
    while (rem > 1 && k < 8) {
        int pick = 0;
        for (int fi = weld_n_factors - 1; fi >= 0; --fi) {
            const int r = weld_factors[fi];
            if (!pow2First && r == 2 && rem / r > 1 && rem % r == 0 &&
                pick == 0) {
                /* non-pow2-first anchors still want the tail ownable */
            }
            if (rem % r == 0 && rem / r >= 1 &&
                (rem / r > 1 || r == rem)) {
                if (pick == 0 || r > pick) {
                    /* prefer composites/codelets over pow2 tail here */
                    if (!(r == 2 || r == 4 || r == 8) || rem / r == 1) {
                        pick = r;
                        break;
                    }
                    if (pick == 0) {
                        pick = r;
                    }
                }
            }
        }
        if (pick == 0) {
            break;
        }
        rad[k++] = pick;
        rem /= pick;
    }
    if (rem != 1 || k < 2 || k > 8) {
        *kOut = 0;
        return;
    }
    if (!weld_viable(N, rad, k)) {
        *kOut = 0;
        return;
    }
    *kOut = k;
}

static bool weld_same(const WeldCand &a, const WeldCand &b) {
    if (a.k != b.k) {
        return false;
    }
    for (int i = 0; i < a.k; ++i) {
        if (a.rad[i] != b.rad[i]) {
            return false;
        }
    }
    return true;
}

static bool weld_search(int N, bool hasFma, int *radices, int *kStagesOut,
                        double *searchNsOut, int *timedOut, int *poolOut,
                        int *genOut) {
    std::vector<WeldCand> all;
    int cur[8];
    weld_gen(N, cur, 0, N, all);
    *genOut = (int)all.size();
    for (auto &c : all) {
        c.model = weld_model_of(N, c.rad, c.k);
    }
    std::stable_sort(all.begin(), all.end(),
                     [](const WeldCand &a, const WeldCand &b) {
                         return a.model < b.model;
                     });

    std::vector<WeldCand> pool;
    {
        int rad[8];
        int k = 0;
        weld_anchor(N, true, rad, &k);
        if (k > 0) {
            WeldCand c;
            for (int i = 0; i < k; ++i) {
                c.rad[i] = rad[i];
            }
            c.k = k;
            c.model = 0.0;
            pool.push_back(c);
        }
        weld_anchor(N, false, rad, &k);
        if (k > 0) {
            WeldCand c;
            for (int i = 0; i < k; ++i) {
                c.rad[i] = rad[i];
            }
            c.k = k;
            c.model = 0.0;
            bool dup = false;
            for (const auto &p : pool) {
                if (weld_same(p, c)) {
                    dup = true;
                }
            }
            if (!dup) {
                pool.push_back(c);
            }
        }
    }
    for (const auto &c : all) {
        if (pool.size() >= 26) {
            break;
        }
        bool dup = false;
        for (const auto &p : pool) {
            if (weld_same(p, c)) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            pool.push_back(c);
        }
    }
    *poolOut = (int)pool.size();

    double *io = static_cast<double *>(
        aligned_alloc(64, weld_pad64((size_t)N * 16)));
    double *master = static_cast<double *>(
        aligned_alloc(64, weld_pad64((size_t)N * 16)));
    /* shared search scratch: 2N data + full odd-Bp tail slack (the worst
     * candidate's padding lane can span ~2*Bt ~= 2N doubles) */
    double *scratch = static_cast<double *>(
        aligned_alloc(64, weld_pad64(((size_t)N * 4 + 512) * 8)));
    if (io == nullptr || master == nullptr || scratch == nullptr) {
        free(io);
        free(master);
        free(scratch);
        return false;
    }
    {
        uint32_t st = 0x3eed5678u ^ (uint32_t)N;
        for (int i = 0; i < 2 * N; ++i) {
            st = st * 1664525u + 1013904223u;
            master[i] = (double)(st >> 11) / 2097152.0 - 0.5;
        }
    }
    memset(io, 0, (size_t)N * 16);
    memset(scratch, 0, ((size_t)N * 4 + 512) * 8);

    size_t arenaSz = 0;
    for (const auto &c : pool) {
        const size_t need = weld_arena_bytes(N, c.rad, c.k);
        if (need > arenaSz) {
            arenaSz = need;
        }
    }
    double *arena = static_cast<double *>(
        aligned_alloc(64, weld_pad64(arenaSz ? arenaSz : 64)));
    if (arena == nullptr) {
        free(io);
        free(master);
        free(scratch);
        return false;
    }
    memset(arena, 0, arenaSz);

    const long iters = N <= 512 ? 40 : N <= 2048 ? 16 : N <= 8192 ? 6 : 3;
    const double budgetNs = 4.0e6 + 11.0e6 * ((double)N / 32768.0);
    const double t0 = sched_now_ns();
    double bestNs = 1e30;
    int bestIdx = -1;
    int timed = 0;
    for (size_t ci = 0; ci < pool.size(); ++ci) {
        if (bestIdx >= 0 && timed >= 3 &&
            (sched_now_ns() - t0) > budgetNs) {
            break;
        }
        WeldBuilt b;
        bool ok = true;
        try {
            /* forward-only timing (H-independent): the conv kernel is
             * two such halves + boundary ops; ranking correlates. */
            b = weldBuildKernel(N, pool[ci].rad, pool[ci].k, hasFma,
                                /*conv=*/false,
                                /*outIsCaller=*/true /* io has no slack:
                                  odd-tail finals must route through
                                  scratch2 + copy */,
                                nullptr, nullptr, nullptr, scratch, arena);
        } catch (...) {
            ok = false;
        }
        if (!ok || b.em == nullptr) {
            continue;
        }
        b.em->readyRE();
        fft_jit_fn_t fn = reinterpret_cast<fft_jit_fn_t>(
            const_cast<uint8_t *>(b.em->getCode()));
        memcpy(io, master, (size_t)N * 16);
        fn(io);
        fn(io); /* warm */
        double bestC = 1e30;
        for (int rd = 0; rd < 3; ++rd) {
            const double s2 = sched_now_ns();
            for (long i = 0; i < iters; ++i) {
                fn(io);
            }
            const double per = (sched_now_ns() - s2) / (double)iters;
            if (per < bestC) {
                bestC = per;
            }
        }
        if (bestC < bestNs) {
            bestNs = bestC;
            bestIdx = (int)ci;
        }
        ++timed;
        delete b.em; /* arena + scratch borrowed */
    }
    *searchNsOut = sched_now_ns() - t0;
    *timedOut = timed;
    free(arena);
    free(io);
    free(master);
    free(scratch);
    if (bestIdx < 0) {
        return false;
    }
    for (int i = 0; i < pool[bestIdx].k; ++i) {
        radices[i] = pool[bestIdx].rad[i];
    }
    *kStagesOut = pool[bestIdx].k;
    return true;
}

/* quick weldability probe for the host routing (cheap factorization
 * check; the full build still validates everything). */
static bool weld_factorable(int N) {
    if (N < 6 || N > 32768 || is_pow2(N)) {
        return false;
    }
    int rem = N;
    while (rem > 1) {
        bool found = false;
        for (int fi = 0; fi < weld_n_factors; ++fi) {
            if (rem % weld_factors[fi] == 0) {
                rem /= weld_factors[fi];
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

} // namespace
/* ==========================================================================
 * topic jit-ps-sched-fuse — fp32 (ps) SchedEmitter + fused-conv emitter
 *
 * L-A (DELTA F1): parameterized port of the promoted pd SchedEmitter
 * (src/jit/fft_jit.cpp, topic jit-mixed-radix-codelet) to the float path:
 *   - element = 8B float complex; one ymm = 4 complexes = the SAME DIT
 *     position of FOUR parallel instances (k-quad) — the pd vector-2
 *     (k-pair) doubled;
 *   - radix-{2,4,8,16} stages run vector-4 leaves (stage 0: vector-4 over
 *     block quadruples through the digitrev gather table); radix-16 =
 *     DFT-8(even)+DFT-8(odd)+W_16 cross with a 256B spill frame;
 *     radix-32 = k-pair leaves (two DIT positions x a k PAIR per ymm:
 *     DFT-16 even/odd subleaves + W_32 cross, 256B spill);
 *   - ONE twiddle table: vector-4 rows carry 64B slots
 *     ([W^{jk}..W^{j(k+3)} interleaved | pre-swapped twins]), radix-32
 *     rows carry 64B slots ([W^{jk},W^{j(k+1)} | twins]); the twin sign
 *     follows the plan-time FMA3 gate ([[CON-FFT-002]]); the non-FMA path
 *     uses the verified vmulps x2 + vhsubps + vshufps 0xD8 idiom;
 *   - plan-time "decomposition x tile" enumerate+time search with the pd
 *     budget guard and FFT_SCHED_* debug knob semantics (ps re-selects by
 *     fp32 measurement; tile budgets rescaled to the 8B element:
 *     4096/2048/1024 elements = 32/16/8KB footprints);
 *   - N <= 16 keeps the promoted SmallEmitter straight-line kernels; the
 *     legacy ps three-tier dispatch (Pool/Lut radix-2) stays reachable as
 *     the FFT_PS_SCHED=off topic-local comparison build.
 *
 * L-B (DELTA F2/F3) — fused execute (the point of this topic):
 *   - conv: ONE kernel welds the forward half and the conjugate-trick
 *     inverse half.  The forward final stage multiplies X[k]*H[k] and
 *     conjugates AT THE STORE BOUNDARY (per-store-vector H quad + twin
 *     from plan-time tables, natural-order indexed); the inverse half's
 *     stage-0 gather consumes the natural conj(X*H) directly (no wrapper
 *     sweep between halves); the inverse final stage stores conj(.)·invN
 *     via one vmulps by an (invN, -invN) rip constant.  The three-segment
 *     wrapper's pointwise / conj / conj·invN sweeps are eliminated; the
 *     intermediate spectrum transits io exactly once (write by the
 *     forward final, gather-read by the inverse stage-0) instead of
 *     write + 2x read-modify-write + read.
 *   - gauss: the G[n] multiply sinks to the stage-0 GATHER load boundary
 *     (rows duplicated (g,g,g,g) per gather row, pre-arranged in gather
 *     order — one vmulps per gathered element, no window sweep).
 *   - filter-mul: not exposed on the public surface (include/fft.h has no
 *     filter entry) — skipped, recorded honestly.
 * ======================================================================== */
namespace {

class SchedEmitterF final : public Xbyak::CodeGenerator {
public:
    struct Stage {
        int radix;       /* r_t */
        int bprev;       /* B_{t-1} elements (t=0 -> 1) */
        int b;           /* B_t elements */
        const float *tw; /* TW_t base (64B slots); nullptr for t=0 */
    };
    struct Half {
        const Stage *stages;
        int kStages;
        const int *groupLen;       /* tile-group lens (R2-F6 port) */
        const uint32_t *blkTab;    /* R1: stage-0 BLOCK-BASE table — one
                                      natural byte offset per scratch block
                                      (N/r1 entries); slot sa of block b
                                      reads natural base_b + sa*s0Stride */
        int s0Stride;              /* (N/r1)*8 — within-block slot stride */
        const float *gt;           /* natural-order window rows (16B/row)
                                      or nullptr */
        int finalMode;             /* 0 plain; 1 = store X*H then conj
                                      (conv part A); 2 = store conj·invN
                                      (conv part B) */
        const float *hMul;         /* finalMode 1: natural H (N complex) */
        const float *hSwap;        /* finalMode 1: pre-swapped H twin */
        const float *gH;           /* topic jit-joint-search-batched (L-E
                                      load placement): natural-order H rows
                                      (16B/row: (Hr,-Hi,Hi,-Hr)) multiplied
                                      at the stage-0 gather LOAD boundary of
                                      this half — produces conj(x*H) directly
                                      (conv part B gather); or nullptr */
    };

    /* topic jit-joint-search-batched: batchBlocks > 0 emits the L-F
     * single-core batch form — the (single) half body is wrapped in an
     * in-kernel block loop over batchBlocks blocks; the block stride (in
     * float complex ELEMENTS) is a RUNTIME argument (rsi of the
     * fft_jit_fn2_t entry).  Only valid with nHalves == 1 (windowed
     * forward, no conv halves).
     *
     * topic R2 (amend r2, DELTA F5 / L-G): batchSlim > 0 emits the
     * SLIMMED block loop as a DUAL-entry kernel — the generic
     * runtime-stride entry at offset 0 (the contract-safe fn2) plus a
     * contiguous stride==N-specialized entry at contigEntryOff_ whose
     * block advance is the baked immediate N*8 (no per-block stride
     * load+shift; the wrapper dispatches on stride == N).  Block
     * invariants are hoisted OUT of the block loop: the window-row base
     * (r14 — the only stage-0 table register not clobbered by the block
     * body) is set once before the loop, and the generic entry
     * precomputes stride*8 once (per block: a single add rdi,[frame]
     * instead of load+shl+add). */
    SchedEmitterF(int n, const float *scratch, const Half *halves, int nHalves,
                  bool fma, bool nt, int prefetch, int batchBlocks = 0,
                  int batchSlim = 0)
        : Xbyak::CodeGenerator(batchSlim ? 786432 : 262144), fma_(fma),
          nt_(nt && ((1 << n) >= 4096)), prefetch_(prefetch), n_(n),
          scratch_(scratch), halves_(halves), nHalves_(nHalves),
          batchBlocks_(batchBlocks), batchSlim_(batchSlim) {
        const float inv = 1.0f / (float)(1 << n);
        invN_[0] = inv;
        invN_[1] = -inv;
        if (batchBlocks_ > 0 && nHalves_ != 1) {
            /* the batch loop parks the block base in rdx, which the conv
             * part-B half re-purposes — unreachable by construction. */
            throw std::bad_alloc();
        }
        if (batchSlim_ > 0 && batchBlocks_ <= 0) {
            /* the slim dual entry only exists for the single-core batch
             * form — unreachable by construction (fail loudly). */
            throw std::bad_alloc();
        }
        if (batchBlocks_ > 0) {
            if (batchSlim_ > 0) {
                emitBatchFn(false); /* generic runtime-stride entry */
                contigEntryOff_ = getSize();
                emitBatchFn(true); /* contiguous stride==N fast entry */
            } else {
                emitBatchFn(false);
            }
        } else {
            planFrame();
            for (int h = 0; h < nHalves_; ++h) {
                half_ = h;
                emitHalf();
            }
            epilogue();
        }
        emitPool();
    }

    /* byte offset of the contiguous stride==N entry inside the code
     * buffer (slim dual-entry kernels; 0 for the single-entry forms —
     * the generic entry itself lives at getCode()+0). */
    size_t contigEntryOff() const { return contigEntryOff_; }

private:
    static int lf_log2(int v) {
        int s = 0;
        while ((1 << s) < v) {
            ++s;
        }
        return s;
    }

    /* ---- frame plan: callee-saved pushes + spill frame ---- */
    void planFrame() {
        bool needSpill = false;
        needCursors_ = false;
        needWin_ = false;
        needH_ = false;
        needLoop_ = (batchBlocks_ > 0);
        for (int h = 0; h < nHalves_; ++h) {
            const Half &H = halves_[h];
            if (H.kStages > 1) {
                needCursors_ = true;
            }
            for (int t = 0; t < H.kStages; ++t) {
                if (H.stages[t].radix >= 16) {
                    needSpill = true;
                }
            }
            if (H.gt != nullptr || H.gH != nullptr) {
                needWin_ = true;
            }
            if (H.finalMode == 1) {
                needH_ = true;
            }
        }
        spillBias_ = needLoop_ ? 32 : 0; /* [0..31] parks the batch stride */
        frameBytes_ = (needSpill ? 256 : 0) + spillBias_;
        pushAdj_ = (needCursors_ ? 24 : 0) +
                   ((needWin_ || needH_ || needLoop_) ? 16 : 0);
        spillOff_ = -(pushAdj_ + frameBytes_);
        push(rbp);
        mov(rbp, rsp);
        if (needCursors_) {
            push(rbx);
            push(r12);
            push(r13);
        }
        if (needWin_ || needH_ || needLoop_) {
            push(r14);
            push(r15);
        }
        if (frameBytes_ > 0) {
            sub(rsp, frameBytes_);
        }
    }

    void epilogue() {
        vzeroupper();
        if (pushAdj_ > 0) {
            lea(rsp, ptr[rbp - pushAdj_]);
            if (needWin_ || needH_ || needLoop_) {
                pop(r15);
                pop(r14);
            }
            if (needCursors_) {
                pop(r13);
                pop(r12);
                pop(rbx);
            }
        } else {
            lea(rsp, ptr[rbp]); /* frame-only (single-codelet) path */
        }
        pop(rbp);
        ret();
    }

    /* ---- topic R2 (amend r2, DELTA F5 / L-G): the single-core batch
     * function (prologue + block loop + epilogue), emitted once for the
     * incumbent form and TWICE for the slim dual-entry form.  Block
     * invariants hoisted out of the loop (the per-block saving vs the
     * incumbent: the window-row base reload at every stage-0 top and the
     * load+shl+add stride advance; everything else re-materialized inside
     * the body is a per-use cursor clobbered by the sweeps and cannot be
     * hoisted without a second full frame of callee-saves). ---- */
    void emitBatchFn(bool contig) {
        using namespace Xbyak;
        const int N = 1 << n_;
        planFrame();
        if (needWin_ && halves_[0].gt != nullptr) {
            /* hoisted: window rows base once per CALL, not per block
             * (r14 is read by the stage-0 gather and clobbered by nothing
             * else in the block body). */
            mov(r14, reinterpret_cast<uint64_t>(halves_[0].gt));
            gtPreSet_ = true;
        }
        if (!contig) {
            /* rsi = stride (elements) — precompute stride*8 ONCE and park
             * it at the frame head; per block the tail is a single
             * add rdi, [frame]. */
            mov(rax, rsi);
            shl(rax, 3);
            mov(ptr[rbp + spillOff_], rax);
        }
        mov(r15, (uint64_t)(unsigned)batchBlocks_);
        Label blk;
        L(blk);
        half_ = 0;
        emitHalf();
        /* the stage-0 epilogue parks the block base in rdx; advance to
         * io + stride*8 and loop (contig: the baked immediate N*8). */
        mov(rdi, rdx);
        if (contig) {
            add(rdi, (uint32_t)((size_t)N * 8));
        } else {
            add(rdi, ptr[rbp + spillOff_]);
        }
        dec(r15);
        jnz(blk);
        gtPreSet_ = false;
        epilogue();
    }

    /* ---- one transform half ---- */
    void emitHalf() {
        using namespace Xbyak;
        const Half &H = halves_[half_];
        const int N = 1 << n_;

        if (H.kStages < 2) {
            /* unreachable by construction: schedF_valid excludes k < 2
             * decompositions (the leaf32 k-pair layout needs a parallel
             * k dimension).  Fail the emission loudly. */
            throw std::bad_alloc();
        }

        /* half > 0 (fused conv part B): rdi currently parks the scratch
         * base; the ORIGINAL io base lives in rdx — restore it so the
         * stage-0 gather reads the natural conj(X*H) written by part A's
         * final stage. */
        if (half_ > 0) {
            mov(rdi, rdx);
        }
        int t = 0;
        if (H.groupLen[0] >= 2) {
            emitStage0Group(H, N);
            t = H.groupLen[0];
        } else {
            emitStage0Plain(H, N);
            t = 1;
        }
        while (t < H.kStages - 1) {
            const int d = H.groupLen[t] >= 2 ? H.groupLen[t] : 1;
            if (d >= 2) {
                emitTileGroup(H, t, d, N);
            } else {
                emitSweepFull(H.stages[t], N, /*allowNt=*/true);
            }
            t += d;
        }
        emitFinalStage(H, N);
    }

    /* ---- stage 0, unblocked: digitrev gather fused into the first
     * codelet sweep, out-of-place into the scratch.  R1: block-base form —
     * the leaf loads slot sa of scratch block b from io[blkTab[b] +
     * sa*s0Stride] (base registers rbx/r9/r10/r11 hold one block QUAD's
     * natural bases; the per-element digitrev table is gone). ---- */
    void emitStage0Plain(const Half &H, int N) {
        using namespace Xbyak;
        const Stage &st = H.stages[0];
        const Reg64 base[4] = {rbx, r9, r10, r11};
        if (st.radix == 32) {
            /* block-PAIR form (leaf32 k-pair = two scratch blocks) */
            mov(r8, reinterpret_cast<uint64_t>(H.blkTab));
            if (H.gt != nullptr) {
                if (!gtPreSet_) { /* R2 slim: hoisted before the block loop */
                    mov(r14, reinterpret_cast<uint64_t>(H.gt));
                }
                gtBase_ = &r14;
            }
            mov(rsi, reinterpret_cast<uint64_t>(scratch_));
            lea(rdx, ptr[rsi + N * 8]);
            ntSweep_ = nt_;
            Label blk;
            L(blk);
            mov(ebx, dword[r8]);
            mov(r9d, dword[r8 + 4]);
            gBase_ = base;
            s0Stride_ = H.s0Stride;
            emitLeaf32(rsi, 8, false, rsi, true);
            gBase_ = nullptr;
            s0Stride_ = 0;
            add(r8, 8);
            add(rsi, 64 * 8);
            cmp(rsi, rdx);
            jb(blk);
            gtBase_ = nullptr;
            gHBase_ = nullptr;
            ntSweep_ = false;
            if (nt_) {
                sfence();
            }
            mov(rdx, rdi); /* park the io base */
            mov(rdi, reinterpret_cast<uint64_t>(scratch_));
            return;
        }
        mov(r8, reinterpret_cast<uint64_t>(H.blkTab));
        if (H.gt != nullptr) {
            if (!gtPreSet_) { /* R2 slim: hoisted before the block loop */
                mov(r14, reinterpret_cast<uint64_t>(H.gt));
            }
            gtBase_ = &r14;
        }
        if (H.gH != nullptr) {
            mov(r14, reinterpret_cast<uint64_t>(H.gH));
            gHBase_ = &r14;
        }
        mov(rsi, reinterpret_cast<uint64_t>(scratch_));
        lea(rdx, ptr[rsi + N * 8]);
        ntSweep_ = nt_;
        Label blk;
        L(blk);
        mov(ebx, dword[r8]);
        mov(r9d, dword[r8 + 4]);
        mov(r10d, dword[r8 + 8]);
        mov(r11d, dword[r8 + 12]);
        gBase_ = base;
        s0Stride_ = H.s0Stride;
        emitLeafV4(st.radix, rsi, rsi, false);
        gBase_ = nullptr;
        s0Stride_ = 0;
        add(r8, 16); /* 4 block-base entries per leaf */
        add(rsi, st.radix * 32); /* 4 blocks x r1 elements x 8B */
        cmp(rsi, rdx);
        jb(blk);
        gtBase_ = nullptr;
        gHBase_ = nullptr;
        ntSweep_ = false;
        if (nt_) {
            sfence();
        }
        mov(rdx, rdi); /* park the io base */
        mov(rdi, reinterpret_cast<uint64_t>(scratch_));
    }

    /* ---- stage-0 tile group (R2-F6 port): gather the tile's rows, write
     * the tile scratch, run stages 1..d-1 confined to the tile ---- */
    void emitStage0Group(const Half &H, int N) {
        using namespace Xbyak;
        const Stage &st = H.stages[0];
        const int d = H.groupLen[0];
        const int T = H.stages[d - 1].b; /* tile elements */
        const Reg64 base[4] = {rbx, r9, r10, r11};
        bool ntUsed = false;
        mov(r13, reinterpret_cast<uint64_t>(H.blkTab));
        if (H.gt != nullptr) {
            if (!gtPreSet_) { /* R2 slim: hoisted before the block loop */
                mov(r14, reinterpret_cast<uint64_t>(H.gt));
            }
            gtBase_ = &r14;
        }
        if (H.gH != nullptr) {
            mov(r14, reinterpret_cast<uint64_t>(H.gH));
            gHBase_ = &r14;
        }
        mov(r12, reinterpret_cast<uint64_t>(scratch_));
        lea(rcx, ptr[r12 + N * 8]);
        Label tile;
        L(tile);
        emitPrefetchHead(r12, T * 8);
        mov(rsi, r12);
        lea(rdx, ptr[r12 + T * 8]);
        Label blk;
        L(blk);
        if (st.radix == 32) {
            mov(ebx, dword[r13]);
            mov(r9d, dword[r13 + 4]);
            gBase_ = base;
            s0Stride_ = H.s0Stride;
            emitLeaf32(rsi, 8, false, rsi, true);
            gBase_ = nullptr;
            s0Stride_ = 0;
            add(r13, 8);
            add(rsi, 64 * 8);
        } else {
            mov(ebx, dword[r13]);
            mov(r9d, dword[r13 + 4]);
            mov(r10d, dword[r13 + 8]);
            mov(r11d, dword[r13 + 12]);
            gBase_ = base;
            s0Stride_ = H.s0Stride;
            emitLeafV4(st.radix, rsi, rsi, false);
            gBase_ = nullptr;
            s0Stride_ = 0;
            add(r13, 16);
            add(rsi, st.radix * 32);
        }
        cmp(rsi, rdx);
        jb(blk);
        for (int s = 1; s < d; ++s) {
            const bool last = (s == d - 1);
            emitSweepInTile(H.stages[s], T * 8, last);
            ntUsed |= (nt_ && last);
        }
        add(r12, T * 8);
        cmp(r12, rcx);
        jb(tile);
        gtBase_ = nullptr;
        gHBase_ = nullptr;
        if (ntUsed) {
            sfence();
        }
        mov(rdx, rdi);
        mov(rdi, reinterpret_cast<uint64_t>(scratch_));
    }

    /* ---- one stage's sweep confined to the current tile ---- */
    void emitSweepInTile(const Stage &st, int tileBytes, bool lastOfGroup) {
        using namespace Xbyak;
        ntSweep_ = nt_ && lastOfGroup;
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, r12);
        lea(r8, ptr[r12 + tileBytes]);
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 8]);
        mov(rax, rsi);
        mov(r10, r9);
        Label kLoop;
        L(kLoop);
        if (st.radix == 32) {
            curStride_ = st.bprev * 8;
            emitLeaf32(rax, curStride_, true, rax);
            add(rax, 16); /* one k-pair */
            add(r10, 32 * 64);
        } else {
            curStride_ = st.bprev * 8;
            emitLeafV4(st.radix, rax, rax, true);
            add(rax, 32); /* one k-quad */
            add(r10, st.radix * 64);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 8);
        cmp(rsi, r8);
        jb(blk);
        ntSweep_ = false;
    }

    /* ---- one stage's unblocked full-array sweep ---- */
    void emitSweepFull(const Stage &st, int N, bool allowNt) {
        using namespace Xbyak;
        ntSweep_ = nt_ && allowNt;
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, rdi);
        lea(r8, ptr[rdi + N * 8]);
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 8]);
        mov(rax, rsi);
        mov(r10, r9);
        Label kLoop;
        L(kLoop);
        curStride_ = st.bprev * 8;
        if (st.radix == 32) {
            emitLeaf32(rax, curStride_, true, rax);
            add(rax, 16);
            add(r10, 32 * 64);
        } else {
            emitLeafV4(st.radix, rax, rax, true);
            add(rax, 32);
            add(r10, st.radix * 64);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 8);
        cmp(rsi, r8);
        jb(blk);
        ntSweep_ = false;
        if (nt_ && allowNt) {
            sfence();
        }
    }

    /* ---- middle tile group (stages t..t+d-1, t >= 1) ---- */
    void emitTileGroup(const Half &H, int t, int d, int N) {
        using namespace Xbyak;
        const int T = H.stages[t + d - 1].b;
        bool ntUsed = false;
        mov(r12, rdi);
        lea(rcx, ptr[rdi + N * 8]);
        Label tile;
        L(tile);
        emitPrefetchHead(r12, T * 8);
        for (int s = t; s < t + d; ++s) {
            const bool last = (s == t + d - 1);
            emitSweepInTile(H.stages[s], T * 8, last);
            ntUsed |= (nt_ && last);
        }
        add(r12, T * 8);
        cmp(r12, rcx);
        jb(tile);
        if (ntUsed) {
            sfence();
        }
    }

    /* ---- final stage: reads the scratch, writes the natural array ---- */
    void emitFinalStage(const Half &H, int N) {
        using namespace Xbyak;
        const Stage &st = H.stages[H.kStages - 1];
        finalMode_ = H.finalMode;
        if (finalMode_ == 1) {
            /* r14/r15 = H tables relative to the scratch base (the final
             * stage's rax walks the scratch k-cursor) */
            mov(r14, reinterpret_cast<uint64_t>(
                         reinterpret_cast<uintptr_t>(H.hMul) -
                         reinterpret_cast<uintptr_t>(scratch_)));
            mov(r15, reinterpret_cast<uint64_t>(
                         reinterpret_cast<uintptr_t>(H.hSwap) -
                         reinterpret_cast<uintptr_t>(scratch_)));
        }
        mov(r9, reinterpret_cast<uint64_t>(st.tw));
        mov(rsi, rdi);
        lea(r8, ptr[rdi + N * 8]);
        mov(r12, rdx); /* natural block cursor (io parked in rdx) */
        Label blk;
        L(blk);
        lea(r11, ptr[rsi + st.bprev * 8]);
        mov(rax, rsi);
        mov(r10, r9);
        Label kLoop;
        L(kLoop);
        curStride_ = st.bprev * 8;
        if (st.radix == 32) {
            mov(rbx, r12);
            emitLeaf32(rax, curStride_, true, rbx);
            add(rax, 16);
            add(r10, 32 * 64);
            add(r12, 16);
        } else {
            mov(rbx, r12);
            emitLeafV4(st.radix, rax, rbx, true);
            add(rax, 32);
            add(r10, st.radix * 64);
            add(r12, 32);
        }
        cmp(rax, r11);
        jb(kLoop);
        add(rsi, st.b * 8);
        add(r12, st.b * 8);
        cmp(rsi, r8);
        jb(blk);
        finalMode_ = 0;
    }

    void emitPrefetchHead(const Xbyak::Reg64 &base, int tileBytes) {
        if (prefetch_ <= 0) {
            return;
        }
        for (int i = 0; i < 8; ++i) {
            prefetcht0(ptr[base + tileBytes + i * 64]);
        }
    }

    /* ===================== vector-4 leaves (radix 2/4/8/16) ============
     * P[j] = DIT position j of FOUR parallel instances (k-quad, or block
     * quad at stage 0), interleaved (re,im) x4 per ymm.  Loads come from
     * loadBase + slot*curStride_ (twiddled stages) or through the
     * gatherTab_ cursor (stage 0: 8B movq per element + optional window
     * multiply AT THE LOAD BOUNDARY); stores go to storeBase +
     * pos*curStride_ (middle in-place / final natural) or per-block 8B
     * extracts into the scratch (stage 0). */
    void emitLeafV4(int r, const Xbyak::Reg64 &loadBase,
                    const Xbyak::Reg64 &storeBase, bool hasTw) {
        if (r == 16) {
            emitV4Sub(8, loadBase, hasTw, 0);
            for (int j = 0; j < 8; ++j) {
                vmovups(ptr[rbp + spillOff_ + spillBias_ + j * 32], Xbyak::Ymm(j));
            }
            emitV4Sub(8, loadBase, hasTw, 1);
            cross16V4(storeBase);
        } else {
            emitV4Sub(r, loadBase, hasTw, -1);
            storeV4(r, storeBase);
        }
    }

    /* r-point DFT in per-position v4 registers P[0..r-1] (r <= 8).
     * par < 0: whole leaf (slot(pos) = bitrev(pos, log2 r));
     * par >= 0: sub-leaf of the radix-16 leaf (slot(pos) =
     * 2*bitrev(pos,3)+par).  Gather row stride rr = r1 (16 for the
     * radix-16 subleaf pair). */
    /* topic jit-joint-search-batched (L-E load placement): d (ONE complex
     * in the low 64b, upper zeroed by the VEX vmovq gather load) *= H row
     * (16B per natural element: (Hr,-Hi,Hi,-Hr)) such that d becomes
     * conj(d*H) directly — no separate conjugate sweep.  Temps are the
     * P slots Xmm(r-2)/Xmm(r-1) of THIS leaf, which the m=2 fuse writes
     * only after the gather (dead at load time). */
    void mulGatherH(const Xbyak::Xmm &d, const Xbyak::Reg64 &gh,
                    const Xbyak::Reg64 &bs, int da, int r) {
        using namespace Xbyak;
        const Xmm t1 = Xmm(r - 2);
        const Xmm t2 = Xmm(r - 1);
        vmovsldup(t1, d);
        vmovshdup(t2, d);
        if (fma_) {
            vmulps(t2, t2, ptr[gh + bs * 2 + (2 * da + 8)]);
            vfmaddsub231ps(t2, t1, ptr[gh + bs * 2 + 2 * da]);
            vmovaps(d, t2);
        } else {
            vmulps(t2, t2, ptr[gh + bs * 2 + (2 * da + 8)]);
            vmulps(d, t1, ptr[gh + bs * 2 + 2 * da]);
            vaddsubps(d, d, t2);
        }
    }

    void emitV4Sub(int r, const Xbyak::Reg64 &loadBase, bool hasTw, int par) {
        using namespace Xbyak;
        const int nlf = lf_log2(r);
        const int stride = curStride_;
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};

        /* ---- load (+twiddle / +window) + m=2 fuse ---- */
        for (int u = 0; u < r / 2; ++u) {
            int sa, sb;
            if (par >= 0) {
                sa = 2 * (int)bitrev((uint32_t)(2 * u), nlf) + par;
                sb = 2 * (int)bitrev((uint32_t)(2 * u + 1), nlf) + par;
            } else {
                sa = (int)bitrev((uint32_t)(2 * u), nlf);
                sb = (int)bitrev((uint32_t)(2 * u + 1), nlf);
            }
            if (gBase_ != nullptr) {
                /* R1 block-base gather: slot sa of scratch block i lives
                 * at io[base_i + sa*s0Stride_] — one table entry per
                 * BLOCK instead of per element; window rows are natural
                 * order (16B rows), indexed by the same offsets x2. */
                const int da = sa * s0Stride_;
                const int db = sb * s0Stride_;
                for (int i = 0; i < 4; ++i) {
                    const Xbyak::Reg64 &bs = gBase_[i];
                    /* VEX vmovq ZEROES the upper 64 bits (legacy movq
                     * leaves stale garbage that stalls adds on denormal
                     * assists) */
                    vmovq(Xmm(8 + i), ptr[rdi + bs + da]);
                    vmovq(Xmm(12 + i), ptr[rdi + bs + db]);
                    if (gtBase_ != nullptr) {
                        const Xbyak::Reg64 &gt = *gtBase_;
                        vmulps(Xmm(8 + i), Xmm(8 + i),
                               ptr[gt + bs * 2 + 2 * da]);
                        vmulps(Xmm(12 + i), Xmm(12 + i),
                               ptr[gt + bs * 2 + 2 * db]);
                    }
                    if (gHBase_ != nullptr) {
                        /* topic jit-joint-search-batched (L-E load
                         * placement): complex multiply by the natural-order
                         * H rows AT THE LOAD BOUNDARY, producing conj(x*H)
                         * directly (part A stored plain X; this is part
                         * B's stage-0 gather of the spectrum). */
                        const Xbyak::Reg64 &gh = *gHBase_;
                        mulGatherH(Xmm(8 + i), gh, bs, da, r);
                        mulGatherH(Xmm(12 + i), gh, bs, db, r);
                    }
                }
                /* m=2 fuse per block; block 0 lands in the P lows */
                vaddps(Xmm(2 * u), Xmm(8), Xmm(12));
                vsubps(Xmm(2 * u + 1), Xmm(8), Xmm(12));
                vaddps(xmm8, xmm9, xmm13);    /* b1 sum */
                vsubps(xmm9, xmm9, xmm13);    /* b1 diff */
                vaddps(xmm13, xmm10, xmm14);  /* b2 sum */
                vsubps(xmm10, xmm10, xmm14);  /* b2 diff */
                vaddps(xmm14, xmm11, xmm15);  /* b3 sum */
                vsubps(xmm11, xmm11, xmm15);  /* b3 diff */
                /* P[2u] = (sum0, sum1 | sum2, sum3) */
                vpunpcklqdq(xmm12, Xmm(2 * u), xmm8);
                vpunpcklqdq(xmm15, xmm13, xmm14);
                vinsertf128(P[2 * u], P[2 * u], xmm12, 0);
                vinsertf128(P[2 * u], P[2 * u], xmm15, 1);
                /* P[2u+1] = (diff0, diff1 | diff2, diff3) */
                vpunpcklqdq(xmm12, Xmm(2 * u + 1), xmm9);
                vpunpcklqdq(xmm15, xmm10, xmm11);
                vinsertf128(P[2 * u + 1], P[2 * u + 1], xmm12, 0);
                vinsertf128(P[2 * u + 1], P[2 * u + 1], xmm15, 1);
            } else {
                loadTwV4(ymm10, ptr[loadBase + sa * stride], sa, hasTw);
                loadTwV4(ymm11, ptr[loadBase + sb * stride], sb, hasTw);
                vaddps(P[2 * u], ymm10, ymm11);
                vsubps(P[2 * u + 1], ymm10, ymm11);
            }
        }

        /* the sign mask (ymm14) is loaded AFTER the gather fuse (which
         * uses xmm14/15 as load temps) */
        if (r >= 4) {
            vmovaps(ymm14, ptr[rip + sign2Lbl_]);
        }

        /* ---- leaf stages 2..nlf ---- */
        for (int st = 2; st <= nlf; ++st) {
            const int m = 1 << st;
            const int h = m >> 1;
            for (int blk = 0; blk < r / m; ++blk) {
                for (int i = 0; i < h; ++i) {
                    const Ymm &bot = P[blk * m + i];
                    const Ymm &top = P[blk * m + h + i];
                    if (i == 0) {
                        vsubps(ymm12, bot, top);
                        vaddps(bot, bot, top);
                        vmovaps(top, ymm12);
                    } else if (i == (m >> 2)) {
                        /* W_m^{m/4} = -i: swap + sign flip */
                        vshufps(ymm12, top, top, 0xB1);
                        vxorps(ymm12, ymm12, ymm14);
                        vsubps(top, bot, ymm12);
                        vaddps(bot, bot, ymm12);
                    } else {
                        mulTwPoolV4(ymm12, top, st, i);
                        vsubps(top, bot, ymm12);
                        vaddps(bot, bot, ymm12);
                    }
                }
            }
        }
    }

    /* load the k-quad at addr; twiddle by TW row slot (64B slots at
     * r10 + slot*64: [W quad interleaved | twin]) when slot != 0. */
    void loadTwV4(const Xbyak::Ymm &dst, const Xbyak::Operand &addr,
                  int slot, bool hasTw) {
        using namespace Xbyak;
        if (slot == 0 || !hasTw) {
            vmovups(dst, addr);
            return;
        }
        /* temps ymm8/ymm12 only: the two loadTwV4 destinations in the
         * m=2 fuse are ymm10/ymm11 and MUST survive both calls. */
        vmovups(ymm8, addr);
        if (fma_) {
            vmovshdup(dst, ymm8); /* (di,di) per complex */
            vmovsldup(ymm12, ymm8); /* (dr,dr) per complex */
            vmulps(dst, dst, ptr[r10 + slot * 64 + 32]);
            vfmaddsub231ps(dst, ymm12, ptr[r10 + slot * 64]);
        } else {
            vmulps(dst, ymm8, ptr[r10 + slot * 64]);
            vmulps(ymm12, ymm8, ptr[r10 + slot * 64 + 32]);
            vhsubps(dst, dst, ymm12);
            vshufps(dst, dst, dst, 0xD8);
        }
    }

    /* dst = top * W_{2^st}^e from the rip pool (broadcast quads). */
    void mulTwPoolV4(const Xbyak::Ymm &dst, const Xbyak::Ymm &top, int st,
                     int e) {
        using namespace Xbyak;
        if (fma_) {
            vmovshdup(dst, top);
            vmovsldup(ymm13, top);
            vmulps(dst, dst, ptr[rip + twSwapLbl_[st][e]]);
            vfmaddsub231ps(dst, ymm13, ptr[rip + twLbl_[st][e]]);
        } else {
            vmulps(dst, top, ptr[rip + twLbl_[st][e]]);
            vmulps(ymm15, top, ptr[rip + twSwapLbl_[st][e]]);
            vhsubps(dst, dst, ymm15);
            vshufps(dst, dst, dst, 0xD8);
        }
    }

    /* radix-16 cross: y[j] = e[j] + W_16^j·o[j], y[j+8] = e[j] − W_16^j·o[j]
     * (j = 0..7); e[] reloaded from the spill frame, o[] in P. */
    void cross16V4(const Xbyak::Reg64 &storeBase) {
        using namespace Xbyak;
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};
        for (int j = 0; j < 8; ++j) {
            vmovups(ymm8, ptr[rbp + spillOff_ + spillBias_ + j * 32]);
            if (j == 0) {
                vaddps(ymm10, ymm8, P[0]);
                vsubps(ymm11, ymm8, P[0]);
            } else if (j == 4) {
                vshufps(ymm12, P[4], P[4], 0xB1);
                vxorps(ymm12, ymm12, ymm14);
                vaddps(ymm10, ymm8, ymm12);
                vsubps(ymm11, ymm8, ymm12);
            } else {
                mulTwPoolV4(ymm12, P[j], 4, j);
                vaddps(ymm10, ymm8, ymm12);
                vsubps(ymm11, ymm8, ymm12);
            }
            storeV4One(ymm10, j, storeBase, 16);
            storeV4One(ymm11, j + 8, storeBase, 16);
        }
    }

    void storeV4(int r, const Xbyak::Reg64 &storeBase) {
        for (int p = 0; p < r; ++p) {
            storeV4One(Xbyak::Ymm(p), p, storeBase, r);
        }
    }

    /* store position pos of a v4 leaf.  Stage-0 gather mode: the ymm's 4
     * complexes are 4 BLOCKS' position pos -> 8B extracts into the 4 block
     * regions of the scratch.  Final stage (finalMode_): the fused
     * store-boundary op runs first (X*H + conj, or conj·invN). */
    void storeV4One(const Xbyak::Ymm &v, int pos,
                    const Xbyak::Reg64 &storeBase, int r) {
        using namespace Xbyak;
        if (gBase_ != nullptr) {
            vextractf128(xmm12, v, 0);
            pextrq(ptr[storeBase + (size_t)pos * 8], xmm12, 0);
            pextrq(ptr[storeBase + (size_t)(r + pos) * 8], xmm12, 1);
            vextractf128(xmm12, v, 1);
            pextrq(ptr[storeBase + (size_t)(2 * r + pos) * 8], xmm12, 0);
            pextrq(ptr[storeBase + (size_t)(3 * r + pos) * 8], xmm12, 1);
            return;
        }
        const int off = pos * curStride_;
        if (finalMode_ == 1) {
            /* X[k..k+3]·H[k..k+3] at the store boundary, then conj */
            if (fma_) {
                vmovshdup(ymm12, v);
                vmovsldup(ymm13, v);
                vmulps(ymm12, ymm12, ptr[r15 + rax + off]);
                vfmaddsub231ps(ymm12, ymm13, ptr[r14 + rax + off]);
            } else {
                vmulps(ymm12, v, ptr[r14 + rax + off]);
                vmulps(ymm13, v, ptr[r15 + rax + off]);
                vhsubps(ymm12, ymm12, ymm13);
                vshufps(ymm12, ymm12, ymm12, 0xD8);
            }
            vxorps(ymm12, ymm12, ptr[rip + sign2Lbl_]);
            if (ntSweep_) {
                vmovntps(ptr[storeBase + off], ymm12);
            } else {
                vmovups(ptr[storeBase + off], ymm12);
            }
            return;
        }
        if (finalMode_ == 2) {
            vmulps(ymm12, v, ptr[rip + invNLbl_]);
            if (ntSweep_) {
                vmovntps(ptr[storeBase + off], ymm12);
            } else {
                vmovups(ptr[storeBase + off], ymm12);
            }
            return;
        }
        if (ntSweep_) {
            vmovntps(ptr[storeBase + off], v);
        } else {
            vmovups(ptr[storeBase + off], v);
        }
    }

    /* ===================== radix-32 leaf (k-pair) =====================
     * P[p] = (posA of k, posA of k+1 | posB of k, posB of k+1) — one ymm
     * carries TWO adjacent DIT positions of a k PAIR (pd leaf32 shape
     * doubled).  DFT-32 = DFT-16(even slots) + DFT-16(odd slots) + W_32
     * cross; each DFT-16 runs in the 8 P registers (pair-chunk rip-pool
     * twiddles); the even DFT-16 spills to the frame.  16B loads carry
     * one position of (k, k+1). */
    void emitLeaf32(const Xbyak::Reg64 &loadBase, int strideBytes,
                    bool hasTw, const Xbyak::Reg64 &storeBase,
                    bool gather = false) {
        using namespace Xbyak;
        vmovaps(ymm14, ptr[rip + sign2Lbl_]);
        curStride_ = strideBytes;
        leaf32Base_ = &loadBase;
        emitSubLeaf16Kp(loadBase, hasTw, 0, gather);
        for (int p = 0; p < 8; ++p) {
            vmovups(ptr[rbp + spillOff_ + spillBias_ + p * 32], Ymm(p));
        }
        emitSubLeaf16Kp(loadBase, hasTw, 1, gather);
        for (int p = 0; p < 8; ++p) {
            vmovups(ymm9, ptr[rbp + spillOff_ + spillBias_ + p * 32]);
            mulTwW32Kp(ymm8, Ymm(p), p);
            vaddps(ymm13, ymm9, ymm8);
            vsubps(ymm15, ymm9, ymm8);
            store32Chunk(ymm13, 2 * p, storeBase, gather);
            store32Chunk(ymm15, 2 * p + 16, storeBase, gather);
        }
    }

    /* DFT-16 over slots {2i+par} in k-pair P registers.  R1 gather=true:
     * the "k-pair" is a scratch block PAIR (blocks A/B bases in
     * gBase_[0]/gBase_[1]); slot loads come from io[base + slot*
     * s0Stride_] as two 8B loads combined (stage 0 is untwiddled). */
    void emitSubLeaf16Kp(const Xbyak::Reg64 &loadBase, bool hasTw, int par,
                         bool gather = false) {
        using namespace Xbyak;
        const Ymm P[8] = {ymm0, ymm1, ymm2, ymm3,
                          ymm4, ymm5, ymm6, ymm7};
        for (int p = 0; p < 8; ++p) {
            const int sa = 2 * (int)bitrev((uint32_t)(2 * p), 4) + par;
            const int sb = 2 * (int)bitrev((uint32_t)(2 * p + 1), 4) + par;
            if (gather) {
                const int da = sa * s0Stride_;
                const int db = sb * s0Stride_;
                vmovq(xmm8, ptr[rdi + gBase_[0] + da]);
                vmovq(xmm10, ptr[rdi + gBase_[1] + da]);
                vpunpcklqdq(xmm8, xmm8, xmm10);
                vmovq(xmm9, ptr[rdi + gBase_[0] + db]);
                vmovq(xmm10, ptr[rdi + gBase_[1] + db]);
                vpunpcklqdq(xmm9, xmm9, xmm10);
            } else {
                vmovups(xmm8, ptr[loadBase + sa * curStride_]);
                vmovups(xmm9, ptr[loadBase + sb * curStride_]);
            }
            if (hasTw && sa != 0) {
                mulTwKpXmm(xmm8, sa);
            }
            if (hasTw && sb != 0) {
                mulTwKpXmm(xmm9, sb);
            }
            vaddps(xmm12, xmm8, xmm9); /* sum pair (k, k+1) */
            vsubps(xmm13, xmm8, xmm9); /* diff pair (k, k+1) */
            vinsertf128(P[p], P[p], xmm12, 0);
            vinsertf128(P[p], P[p], xmm13, 1);
        }

        /* stage 2 (m=4): top HIGH 128 (posB) × (−i) blended */
        for (int blk = 0; blk < 4; ++blk) {
            const Ymm &bot = P[2 * blk];
            const Ymm &top = P[2 * blk + 1];
            vshufps(ymm8, top, top, 0xB1);
            vxorps(ymm8, ymm8, ymm14);
            vblendps(ymm8, top, ymm8, 0xF0);
            vsubps(top, bot, ymm8);
            vaddps(bot, bot, ymm8);
        }

        /* stages 3..4 (m=8, m=16): pair chunks from the rip pool (W^{2q},
         * W^{2q+1} per 128 half) */
        for (int st = 3; st <= 4; ++st) {
            const int halfPairs = 1 << (st - 2);
            const int pairsPerBlock = 1 << (st - 1);
            for (int blk = 0; blk < 16 / (1 << st); ++blk) {
                const int bp = blk * pairsPerBlock;
                for (int q = 0; q < halfPairs; ++q) {
                    const Ymm &bot = P[bp + q];
                    const Ymm &top = P[bp + halfPairs + q];
                    mulTwPoolKpSt(ymm8, top, st, q);
                    vsubps(top, bot, ymm8);
                    vaddps(bot, bot, ymm8);
                }
            }
        }
    }

    /* dst(xmm, 16B = one position of k,k+1) = d * TW[slot]; 64B slots at
     * r10 + slot*64 = [W^{jk}, W^{j(k+1)} interleaved | twins]. */
    void mulTwKpXmm(const Xbyak::Xmm &dst, int slot) {
        using namespace Xbyak;
        if (fma_) {
            vmovsldup(xmm10, dst);
            vmovshdup(xmm11, dst);
            vmulps(xmm11, xmm11, ptr[r10 + slot * 64 + 32]);
            vfmaddsub231ps(xmm11, xmm10, ptr[r10 + slot * 64]);
            vmovaps(dst, xmm11);
        } else {
            vmulps(xmm12, dst, ptr[r10 + slot * 64]);
            vmulps(xmm13, dst, ptr[r10 + slot * 64 + 32]);
            vhsubps(xmm12, xmm12, xmm13);
            vshufps(dst, xmm12, xmm12, 0xD8);
        }
    }

    /* dst(ymm) = top * leaf-stage pair chunk (W^{2q}, W^{2q+1}) broadcast
     * to both 128 halves (+ twins). */
    void mulTwPoolKpSt(const Xbyak::Ymm &dst, const Xbyak::Ymm &top, int st,
                       int q) {
        using namespace Xbyak;
        if (fma_) {
            vmovshdup(dst, top);
            vmovsldup(ymm11, top);
            vmulps(dst, dst, ptr[rip + lwSwapLbl_[st][q]]);
            vfmaddsub231ps(dst, ymm11, ptr[rip + lwLbl_[st][q]]);
        } else {
            vmulps(dst, top, ptr[rip + lwLbl_[st][q]]);
            vmulps(ymm12, top, ptr[rip + lwSwapLbl_[st][q]]);
            vhsubps(dst, dst, ymm12);
            vshufps(dst, dst, dst, 0xD8);
        }
    }

    /* dst(ymm) = top * W_32 cross chunk p = (W_32^{2p}, W_32^{2p+1}). */
    void mulTwW32Kp(const Xbyak::Ymm &dst, const Xbyak::Ymm &top, int p) {
        using namespace Xbyak;
        if (fma_) {
            vmovshdup(dst, top);
            vmovsldup(ymm11, top);
            vmulps(dst, dst, ptr[rip + w32SwapLbl_[p]]);
            vfmaddsub231ps(dst, ymm11, ptr[rip + w32Lbl_[p]]);
        } else {
            vmulps(dst, top, ptr[rip + w32Lbl_[p]]);
            vmulps(ymm12, top, ptr[rip + w32SwapLbl_[p]]);
            vhsubps(dst, dst, ymm12);
            vshufps(dst, dst, dst, 0xD8);
        }
    }

    /* store the (k, k+1) position-pair result: extract0 = pos j of
     * (k,k+1) at [storeBase + j*stride], extract1 = pos j+1.  R1 gather
     * mode (stage 0): the pair is a scratch block PAIR — extract0 =
     * (pos j blkA | pos j blkB) -> two 8B stores at [storeBase + j*8]
     * and [storeBase + 256 + j*8] (block B's region, 32 elems x 8B). */
    void store32Chunk(const Xbyak::Ymm &v, int j,
                      const Xbyak::Reg64 &storeBase, bool gather = false) {
        using namespace Xbyak;
        const int off0 = j * curStride_;
        const int off1 = (j + 1) * curStride_;
        vextractf128(xmm8, v, 0);
        vextractf128(xmm9, v, 1);
        if (gather) {
            /* extract0 = (pos j blkA | pos j blkB): the low 64b goes to
             * block A's region, the HIGH 64b (pextrq) to block B's. */
            vmovlps(ptr[storeBase + j * 8], xmm8);
            vpextrq(ptr[storeBase + 256 + j * 8], xmm8, 1);
            vmovlps(ptr[storeBase + (j + 1) * 8], xmm9);
            vpextrq(ptr[storeBase + 256 + (j + 1) * 8], xmm9, 1);
            return;
        }
        if (finalMode_ == 1) {
            mulHKpXmm(xmm8, off0);
            mulHKpXmm(xmm9, off1);
        } else if (finalMode_ == 2) {
            vmulps(xmm8, xmm8, ptr[rip + invN16Lbl_]);
            vmulps(xmm9, xmm9, ptr[rip + invN16Lbl_]);
        }
        if (ntSweep_) {
            movntps(ptr[storeBase + off0], xmm8);
            movntps(ptr[storeBase + off1], xmm9);
        } else {
            vmovups(ptr[storeBase + off0], xmm8);
            vmovups(ptr[storeBase + off1], xmm9);
        }
    }

    /* xmm (16B, one position of k,k+1) *= H pair at
     * [r14 + loadBase + off], then conjugate (conv part A boundary). */
    void mulHKpXmm(const Xbyak::Xmm &d, int off) {
        using namespace Xbyak;
        if (fma_) {
            vmovsldup(xmm10, d);
            vmovshdup(xmm11, d);
            vmulps(xmm11, xmm11, ptr[r15 + *leaf32Base_ + off]);
            vfmaddsub231ps(xmm11, xmm10, ptr[r14 + *leaf32Base_ + off]);
            vmovaps(d, xmm11);
        } else {
            vmulps(xmm12, d, ptr[r14 + *leaf32Base_ + off]);
            vmulps(xmm13, d, ptr[r15 + *leaf32Base_ + off]);
            vhsubps(xmm12, xmm12, xmm13);
            vshufps(d, xmm12, xmm12, 0xD8);
        }
        vxorps(d, d, ptr[rip + sign2Lbl_]);
    }

    /* ---- rip-relative constant pool (fp32) ---- */
    void emitPool() {
        using namespace Xbyak;
        align(32);
        L(sign2Lbl_); /* (0, -0) per complex x4 */
        for (int i = 0; i < 4; ++i) {
            dd(0u);
            dd(0x80000000u);
        }

        align(32);
        L(invNLbl_); /* (invN, -invN) x4 */
        for (int i = 0; i < 4; ++i) {
            dd(float_bits(invN_[0]));
            dd(float_bits(invN_[1]));
        }
        align(16);
        L(invN16Lbl_); /* (invN, -invN) x2 */
        for (int i = 0; i < 2; ++i) {
            dd(float_bits(invN_[0]));
            dd(float_bits(invN_[1]));
        }

        /* W_{2^s}^e broadcast quads (s=3..4; e=0 skipped (=1), e=2^{s-2}
         * skipped (= -i fold)); s=4 doubles as the radix-16 cross. */
        for (int s = 3; s <= 4; ++s) {
            for (int e = 1; e < (1 << (s - 1)); ++e) {
                if (e == (1 << (s - 2))) {
                    continue;
                }
                const double ang =
                    -2.0 * kPi * (double)e / (double)(1 << s);
                const float wr = (float)std::cos(ang);
                const float wi = (float)std::sin(ang);
                const float tw = fma_ ? wr : -wr;
                align(32);
                L(twLbl_[s][e]);
                for (int i = 0; i < 4; ++i) {
                    dd(float_bits(wr));
                    dd(float_bits(wi));
                }
                align(32);
                L(twSwapLbl_[s][e]);
                for (int i = 0; i < 4; ++i) {
                    dd(float_bits(wi));
                    dd(float_bits(tw));
                }
            }
        }

        /* k-pair leaf chunks: chunk q of stage s = (W^{2q}, W^{2q+1})
         * broadcast to both 128 halves (+ twins). */
        for (int s = 3; s <= 4; ++s) {
            const int halfPairs = 1 << (s - 2);
            for (int q = 0; q < halfPairs; ++q) {
                float c[2], sn[2], tw[2];
                for (int l = 0; l < 2; ++l) {
                    const int e = 2 * q + l;
                    const double ang =
                        -2.0 * kPi * (double)e / (double)(1 << s);
                    c[l] = (float)std::cos(ang);
                    sn[l] = (float)std::sin(ang);
                    tw[l] = fma_ ? c[l] : -c[l];
                }
                align(32);
                L(lwLbl_[s][q]);
                /* (W^{2q} dup over the k-pair | W^{2q+1} dup): one
                 * twiddle PER 128-BIT POSITION LANE, duplicated across
                 * the k-pair complexes inside the lane. */
                for (int rep = 0; rep < 2; ++rep) {
                    dd(float_bits(c[0]));
                    dd(float_bits(sn[0]));
                }
                for (int rep = 0; rep < 2; ++rep) {
                    dd(float_bits(c[1]));
                    dd(float_bits(sn[1]));
                }
                align(32);
                L(lwSwapLbl_[s][q]);
                for (int rep = 0; rep < 2; ++rep) {
                    dd(float_bits(sn[0]));
                    dd(float_bits(tw[0]));
                }
                for (int rep = 0; rep < 2; ++rep) {
                    dd(float_bits(sn[1]));
                    dd(float_bits(tw[1]));
                }
            }
        }

        /* W_32 cross chunks: chunk p = (W_32^{2p}, W_32^{2p+1}). */
        for (int p = 0; p < 8; ++p) {
            float c[2], sn[2], tw[2];
            for (int l = 0; l < 2; ++l) {
                const int e = 2 * p + l;
                const double ang = -2.0 * kPi * (double)e / 32.0;
                c[l] = (float)std::cos(ang);
                sn[l] = (float)std::sin(ang);
                tw[l] = fma_ ? c[l] : -c[l];
            }
            align(32);
            L(w32Lbl_[p]);
            for (int rep = 0; rep < 2; ++rep) {
                dd(float_bits(c[0]));
                dd(float_bits(sn[0]));
            }
            for (int rep = 0; rep < 2; ++rep) {
                dd(float_bits(c[1]));
                dd(float_bits(sn[1]));
            }
            align(32);
            L(w32SwapLbl_[p]);
            for (int rep = 0; rep < 2; ++rep) {
                dd(float_bits(sn[0]));
                dd(float_bits(tw[0]));
            }
            for (int rep = 0; rep < 2; ++rep) {
                dd(float_bits(sn[1]));
                dd(float_bits(tw[1]));
            }
        }
    }

    static uint32_t float_bits(float f) {
        uint32_t u;
        __builtin_memcpy(&u, &f, sizeof(u));
        return u;
    }

    bool fma_;
    bool nt_ = false;
    int prefetch_ = 0;
    bool ntSweep_ = false;
    int n_;
    const float *scratch_;
    const Half *halves_;
    int nHalves_;
    int half_ = 0;
    int finalMode_ = 0;
    bool needCursors_ = false;
    bool needWin_ = false;
    bool needH_ = false;
    bool needLoop_ = false; /* topic jit-joint-search-batched: batch mode */
    int batchBlocks_ = 0;   /* batch mode: baked block count (else 0) */
    int batchSlim_ = 0;     /* topic R2: slim dual-entry batch form */
    size_t contigEntryOff_ = 0; /* topic R2: contiguous entry offset */
    bool gtPreSet_ = false; /* topic R2: gt base hoisted before the loop */
    int frameBytes_ = 0;
    int pushAdj_ = 24;
    int spillOff_ = 0;
    int spillBias_ = 0; /* batch: 32 (stride park slot before the spills) */
    int curStride_ = 16; /* bytes between DIT slots (v4 leaves) */
    float invN_[2] = {0.0f, -0.0f};
    /* R1 stage-0 gather: block bases held in registers (v4 quad: 4 regs;
     * leaf32 block-pair: first 2), loads at [rdi + base + slot*stride] */
    const Xbyak::Reg64 *gBase_ = nullptr; /* [4] block base regs */
    int s0Stride_ = 0;                    /* slot stride bytes */
    const Xbyak::Reg64 *gtBase_ = nullptr;
    const Xbyak::Reg64 *gHBase_ = nullptr; /* L-E load placement rows */
    const Xbyak::Reg64 *leaf32Base_ = nullptr;
    Xbyak::Label sign2Lbl_;
    Xbyak::Label invNLbl_;
    Xbyak::Label invN16Lbl_;
    Xbyak::Label twLbl_[5][8];
    Xbyak::Label twSwapLbl_[5][8];
    Xbyak::Label lwLbl_[5][4];
    Xbyak::Label lwSwapLbl_[5][4];
    Xbyak::Label w32Lbl_[8];
    Xbyak::Label w32SwapLbl_[8];
};

/* ---- fp32 plan-side: tables + decomposition search (pd framework port) */

struct SchedFBuilt {
    SchedEmitterF *em;
    float *lut;     /* gatherTab [+ gt rows] + TW [+ hMul/hSwap] */
    float *scratch; /* stage-0 ping buffer (N*8B) */
    bool ownsLut;
    bool ownsScratch;
    int slim;       /* topic R2: 1 = slim dual-entry batch single form */
};

static inline size_t schedF_pad32(size_t b) { return (b + 31u) & ~31u; }

static inline size_t schedF_pad64(size_t b) { return (b + 63u) & ~63u; }

/* lut layout (bytes): [gatherTab N*4 pad32][gt N*16 pad32 (windowed)]
 * [TW_t per stage t>=1, pad32 each][hMul N*8 pad32][hSwap N*8 pad32]. */
static size_t schedF_tw_bytes(int N, const int *radices, int kStages) {
    (void)N;
    int B[8];
    B[0] = radices[0];
    for (int t = 1; t < kStages; ++t) {
        B[t] = B[t - 1] * radices[t];
    }
    size_t cur = 0;
    for (int t = 1; t < kStages; ++t) {
        const int r = radices[t];
        const bool v4 = (r <= 16);
        const int rows = v4 ? B[t - 1] / 4 : B[t - 1] / 2;
        const int rowBytes = v4 ? r * 64 : 32 * 64;
        cur = schedF_pad32(cur + (size_t)rows * (size_t)rowBytes);
    }
    return cur;
}

/* R1: BLOCK-BASE gather table.  Scratch block b's slot sa reads natural
 * element base_b + sa*(N/r1) (verified: holds for every radix chain), so
 * the stage-0 gather needs ONE natural byte offset per scratch block
 * (N/r1 entries) instead of one per element. */
static void schedF_make_blktab(std::vector<uint32_t> &tab, int N,
                               const int *radices, int kStages) {
    const int r1 = radices[0];
    const int NB = N / r1;
    tab.assign((size_t)NB, 0);
    for (int v = 0; v < NB; ++v) {
        /* natural v (high r1-digit = 0) lands at scratch position p;
         * its block gets base v*8 */
        int e[8];
        int vv = v;
        for (int i = 0; i < kStages; ++i) {
            e[i] = vv % radices[kStages - 1 - i];
            vv /= radices[kStages - 1 - i];
        }
        int p = 0;
        for (int i = 0; i < kStages; ++i) {
            int m = 1;
            for (int l = 0; l < kStages - 1 - i; ++l) {
                m *= radices[l];
            }
            p += e[i] * m;
        }
        tab[(size_t)(p / r1)] = (uint32_t)(v * 8); /* fp32 complex = 8B */
    }
}

/* window rows in NATURAL order: row p = (g,g,g,g) for natural index p —
 * multiplied at the stage-0 load boundary (offsets follow the data). */
static void schedF_make_gt(float *dst, int N, const float *G) {
    for (int p = 0; p < N; ++p) {
        const float g = G[p];
        dst[4 * p + 0] = g;
        dst[4 * p + 1] = g;
        dst[4 * p + 2] = g;
        dst[4 * p + 3] = g;
    }
}

/* TW tables: per-stage rows of 64B slots.  vector-4 rows (r <= 16): one
 * row per k-QUAD, r slots (position j), slot = [W^{jk0..jk0+3}
 * interleaved | twins (wi, +/-wr)].  radix-32 rows: one row per k-PAIR,
 * 32 slots, slot = [W^{jk0}, W^{jk0+1} interleaved (16B used) | twins
 * (16B at +32)].  Double recurrence (drift << fp32 eps -> exact to
 * float rounding); twin sign follows the FMA3 gate. */
static size_t schedF_fill_tw(float *base, const int *radices, int kStages,
                             bool hasFma, const float **twOut) {
    int B[8];
    B[0] = radices[0];
    for (int t = 1; t < kStages; ++t) {
        B[t] = B[t - 1] * radices[t];
    }
    size_t cur = 0;
    for (int t = 1; t < kStages; ++t) {
        const int r = radices[t];
        const int Bt = B[t];
        const bool v4 = (r <= 16);
        const int inst = v4 ? 4 : 2;
        const int rows = v4 ? B[t - 1] / 4 : B[t - 1] / 2;
        const int rowFloats = (v4 ? r * 64 : 32 * 64) / 4;
        twOut[t] = base + cur / 4;
        float *row = base + cur / 4;
        for (int q = 0; q < rows; ++q) {
            const int k0 = q * inst;
            double zr[4], zi[4];
            double wr[4] = {1.0, 1.0, 1.0, 1.0};
            double wi[4] = {0.0, 0.0, 0.0, 0.0};
            for (int c = 0; c < inst; ++c) {
                const double ang =
                    -2.0 * kPi * (double)(k0 + c) / (double)Bt;
                zr[c] = std::cos(ang);
                zi[c] = std::sin(ang);
            }
            for (int j = 0; j < r; ++j) {
                /* 64B slot = [W quad interleaved (inst complexes, 32B
                 * for v4 / 16B used for k-pair) | twin quad at +32B]:
                 * W_c at floats 2c/2c+1, twin_c at floats 8+2c/9+2c —
                 * exactly what loadTwV4 / mulTwKpXmm read. */
                float *slot = row + (size_t)j * 16;
                for (int c = 0; c < inst; ++c) {
                    slot[2 * c + 0] = (float)wr[c];
                    slot[2 * c + 1] = (float)wi[c];
                    slot[8 + 2 * c + 0] = (float)wi[c];
                    slot[8 + 2 * c + 1] =
                        (float)(hasFma ? wr[c] : -wr[c]);
                }
                for (int c = 0; c < inst; ++c) {
                    const double nr = wr[c] * zr[c] - wi[c] * zi[c];
                    const double ni = wr[c] * zi[c] + wi[c] * zr[c];
                    wr[c] = nr;
                    wi[c] = ni;
                }
            }
            row += rowFloats;
        }
        cur = schedF_pad32(cur + (size_t)rows * (size_t)rowFloats * 4);
    }
    return cur;
}

static SchedFBuilt schedFBuildKernel(int N, int n, const int *radices,
                                     int kStages, bool hasFma,
                                     float *borrowScratch, int maxTile,
                                     bool nt, int prefetch, const float *G,
                                     const float *H /* interleaved N*2 or
                                                      nullptr */,
                                     float *borrowLut = nullptr,
                                     int loadH = 0,
                                     int batchBlocks = 0,
                                     int batchSlim = 0) {
    SchedFBuilt b = {nullptr, nullptr, nullptr, false, false, 0};

    int B[8];
    B[0] = radices[0];
    for (int t = 1; t < kStages; ++t) {
        B[t] = B[t - 1] * radices[t];
    }
    int groupLen[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    sched_block_groups(B, kStages, maxTile, groupLen);

    std::vector<uint32_t> blkTab;
    if (kStages > 1) {
        schedF_make_blktab(blkTab, N, radices, kStages);
    }
    const bool windowed = (G != nullptr);
    const bool fused = (H != nullptr);
    const bool loadHMode = fused && loadH != 0;
    if (windowed && radices[0] == 32) {
        /* the leaf32 block-PAIR gather has no per-element window form;
         * schedF_valid(noR1_32) keeps these out of the pool — the
         * forced-order debug knob lands here, fail loudly. */
        throw std::bad_alloc();
    }
    if (loadHMode && radices[0] == 32) {
        /* same exclusion for the conv load-boundary rows (L-E): the
         * leaf32 pair gather has no per-element complex-H form. */
        throw std::bad_alloc();
    }
    if (batchBlocks > 0 && (fused || kStages < 2)) {
        /* the batch block loop parks the block base in rdx — reserved by
         * the conv part-B half; and a single-codelet plan owns no
         * scratch to ping through. */
        throw std::bad_alloc();
    }

    const size_t tabBytes = schedF_pad64(blkTab.size() * 4);
    const size_t gtBytes = windowed ? schedF_pad64((size_t)N * 16) : 0;
    const size_t twBytes = schedF_pad64(schedF_tw_bytes(N, radices, kStages));
    const size_t hBytes = fused ? 2 * schedF_pad64((size_t)N * 8) : 0;
    const size_t total = tabBytes + gtBytes + twBytes + hBytes + 64;

    if (borrowLut != nullptr) {
        b.lut = borrowLut;
        b.ownsLut = false;
    } else {
        b.lut = static_cast<float *>(aligned_alloc(64, total));
        if (b.lut == nullptr) {
            throw std::bad_alloc();
        }
        b.ownsLut = true;
    }
    char *base = reinterpret_cast<char *>(b.lut);
    uint32_t *tabPtr = reinterpret_cast<uint32_t *>(base);
    for (size_t i = 0; i < blkTab.size(); ++i) {
        tabPtr[i] = blkTab[i];
    }
    const float *gt = nullptr;
    if (windowed) {
        float *gtp = reinterpret_cast<float *>(base + tabBytes);
        schedF_make_gt(gtp, N, G);
        gt = gtp;
    }
    const float *twv[8] = {nullptr};
    schedF_fill_tw(reinterpret_cast<float *>(base + tabBytes + gtBytes),
                   radices, kStages, hasFma, twv);
    const float *hMul = nullptr;
    const float *hSwap = nullptr;
    const float *gH = nullptr;
    if (loadHMode) {
        /* topic jit-joint-search-batched (L-E load placement): natural-
         * order 16B rows (Hr,-Hi,Hi,-Hr); the stage-0 gather multiply
         * yields conj(x*H) directly (mulGatherH). */
        float *gh = reinterpret_cast<float *>(base + tabBytes + gtBytes +
                                              twBytes);
        for (int i = 0; i < N; ++i) {
            const float hr = H[2 * i];
            const float hi = H[2 * i + 1];
            gh[4 * i + 0] = hr;
            gh[4 * i + 1] = -hi;
            gh[4 * i + 2] = hi;
            gh[4 * i + 3] = -hr;
        }
        gH = gh;
    } else if (fused) {
        float *hm = reinterpret_cast<float *>(base + tabBytes + gtBytes +
                                              twBytes);
        float *hs = reinterpret_cast<float *>(
            base + tabBytes + gtBytes + twBytes + schedF_pad64((size_t)N * 8));
        for (int i = 0; i < N; ++i) {
            const float hr = H[2 * i];
            const float hi = H[2 * i + 1];
            hm[2 * i] = hr;
            hm[2 * i + 1] = hi;
            hs[2 * i] = hi;
            hs[2 * i + 1] = hasFma ? hr : -hr;
        }
        hMul = hm;
        hSwap = hs;
    }

    if (borrowScratch != nullptr) {
        b.scratch = borrowScratch;
        b.ownsScratch = false;
    } else if (kStages > 1) {
        b.scratch = static_cast<float *>(aligned_alloc(64, (size_t)N * 8));
        if (b.scratch == nullptr) {
            if (b.ownsLut) {
                free(b.lut);
            }
            b.lut = nullptr;
            throw std::bad_alloc();
        }
        b.ownsScratch = true;
    }

    SchedEmitterF::Stage stages[8];
    for (int t = 0; t < kStages; ++t) {
        stages[t].radix = radices[t];
        stages[t].bprev = (t == 0) ? 1 : B[t - 1];
        stages[t].b = B[t];
        stages[t].tw = (t == 0) ? nullptr : twv[t];
    }
    SchedEmitterF::Half halves[2];
    halves[0].stages = stages;
    halves[0].kStages = kStages;
    halves[0].groupLen = groupLen;
    halves[0].blkTab = (kStages > 1) ? tabPtr : nullptr;
    halves[0].s0Stride = (kStages > 1) ? (N / radices[0]) * 8 : 0;
    halves[0].gt = gt;
    halves[0].finalMode = (fused && !loadHMode) ? 1 : 0;
    halves[0].hMul = hMul;
    halves[0].hSwap = hSwap;
    halves[0].gH = nullptr;
    int nHalves = 1;
    if (fused) {
        halves[1] = halves[0];
        halves[1].finalMode = 2;
        halves[1].hMul = nullptr;
        halves[1].hSwap = nullptr;
        halves[1].gH = loadHMode ? gH : nullptr;
        nHalves = 2;
    }

    try {
        b.em = new SchedEmitterF(n, b.scratch, halves, nHalves, hasFma, nt,
                                 prefetch, batchBlocks, batchSlim);
        b.slim = (batchBlocks > 0 && batchSlim != 0) ? 1 : 0;
    } catch (...) {
        if (b.ownsLut) {
            free(b.lut);
        }
        b.lut = nullptr;
        if (b.ownsScratch) {
            free(b.scratch);
            b.scratch = nullptr;
        }
        throw;
    }
    return b;
}

/* ---- R1 candidate space: first radix in {2,4,8,16,32} (the stage-0
 * block-quad / block-pair gather covers all five; r1=32 excluded for
 * windowed plans — no per-element window form in the pair layout),
 * later stages {2,4,8,16,32}; single-codelet [32] still excluded (the
 * k-pair leaf32 needs a parallel block dimension). ---- */
static void schedF_greedy16(int n, int *rad, int *kOut) {
    int rem = n;
    int k = 0;
    while (rem > 0) {
        const int r = (rem >= 4) ? 16 : (1 << rem);
        rad[k++] = r;
        rem -= (r == 16) ? 4 : rem;
    }
    *kOut = k;
}

static void schedF_greedy32(int n, int *rad, int *kOut) {
    int rem = n;
    int k = 0;
    while (rem > 0) {
        const int r = (rem >= 5) ? 32 : (1 << rem);
        rad[k++] = r;
        rem -= (r == 32) ? 5 : rem;
    }
    *kOut = k;
}

static bool schedF_valid(const SchedCand &c, int n, bool noR1_32 = false) {
    const int N = 1 << n;
    if (c.k < 2) {
        /* single-codelet [32] is EXCLUDED: the leaf32 k-pair layout has
         * no meaning without a parallel k dimension (N=32 always takes a
         * k=2 decomposition like [8,4]). */
        return false;
    }
    const int r1 = c.rad[0];
    if (r1 == 2 || r1 == 4 || r1 == 8 || r1 == 16) {
        /* the stage-0 vector-4 leaf gathers a block QUAD: the block
         * count N/r1 must be a multiple of 4. */
        if (((N / r1) % 4) != 0) {
            return false;
        }
    } else if (r1 == 32) {
        if (noR1_32) {
            return false; /* windowed: no gather-window form in the pair */
        }
        /* the leaf32 gather runs a scratch block PAIR: even block count
         * >= 2. */
        if (((N / 32) % 2) != 0 || (N / 32) < 2) {
            return false;
        }
    } else {
        return false;
    }
    /* parallel-dimension rule for stages t >= 1 (R1: r1=2 chains hit
     * this — the v4 k-quad leaf needs bprev % 4 == 0, the leaf32 k-pair
     * needs bprev % 2 == 0; R0's r1 in {4,8,16} guaranteed it implicitly):
     */
    int B = r1;
    for (int t = 1; t < c.k; ++t) {
        if (c.rad[t] <= 16 ? (B % 4) != 0 : (B % 2) != 0) {
            return false;
        }
        B *= c.rad[t];
    }
    return true;
}

static void schedF_gen(int rem, int *cur, int len,
                       std::vector<SchedCand> &out) {
    if (rem == 0) {
        SchedCand c;
        for (int i = 0; i < len; ++i) {
            c.rad[i] = cur[i];
        }
        c.k = len;
        c.model = 0.0;
        out.push_back(c);
        return;
    }
    if (len >= 8) {
        return;
    }
    if (len == 0) {
        for (int p = 1; p <= 5 && p <= rem; ++p) {
            cur[0] = 1 << p;
            schedF_gen(rem - p, cur, 1, out);
        }
        return;
    }
    const int hi = rem < 5 ? rem : 5;
    for (int p = 1; p <= hi; ++p) {
        cur[len] = 1 << p;
        schedF_gen(rem - p, cur, len + 1, out);
    }
}

/* R1: rough ps-specific slot ordering model (pool ORDER only — the
 * choice is measured).  Stage 0 = block-base gather (r1=32 = pair
 * leaf32), middle/final = v4 k-quad leaves / k-pair leaf32. */
static double schedF_slot_model(int radix, int pos, int k) {
    const bool stage0 = (pos == 0);
    const bool final = (pos == k - 1) && !stage0;
    switch (radix) {
    case 2:
        return stage0 ? 5.2 : final ? 5.6 : 5.2;
    case 4:
        return stage0 ? 5.3 : final ? 5.8 : 5.4;
    case 8:
        return stage0 ? 5.5 : final ? 6.4 : 5.7;
    case 16:
        return stage0 ? 7.4 : final ? 8.2 : 7.4;
    default: /* 32: stage-0 pair-gather leaf32 / k-pair leaf32 mid */
        return stage0 ? 8.4 : final ? 12.0 : 10.5;
    }
}

static double schedF_model(const int *rad, int k) {
    double s = 0.0;
    for (int t = 0; t < k; ++t) {
        s += schedF_slot_model(rad[t], t, k);
    }
    return s;
}

static size_t schedF_lut_bytes(int N, const int *radices, int kStages) {
    int B[8];
    B[0] = radices[0];
    for (int t = 1; t < kStages; ++t) {
        B[t] = B[t - 1] * radices[t];
    }
    const size_t tabBytes =
        (kStages > 1)
            ? (((size_t)(N / radices[0]) * 4 + 63u) & ~63u)
            : 0;
    return tabBytes + schedF_pad64(schedF_tw_bytes(N, radices, kStages)) +
           64;
}

static bool schedF_search(int N, int n, bool hasFma, bool nt, int prefetch,
                          int forcedTile, bool noR1_32, int *radices,
                          int *kStagesOut, int *tileOut, double *searchNsOut,
                          int *timedOut, int *poolOut, int *genOut,
                          int *refinedOut) {
    std::vector<SchedCand> all;
    int cur[8];
    schedF_gen(n, cur, 0, all);
    *genOut = (int)all.size();
    for (auto &c : all) {
        c.model = schedF_model(c.rad, c.k);
    }
    std::stable_sort(all.begin(), all.end(),
                     [](const SchedCand &a, const SchedCand &b) {
                         return a.model < b.model;
                     });

    /* anchors: greedy 16-first + reversed + greedy 8-first + greedy
     * 32-first + reversed-32 (R1: the 32-heavy corner the R0 pd-model
     * ordering starved), then model-ordered candidates under a PER-r1
     * quota so every first-radix class actually reaches the timer. */
    std::vector<SchedCand> base;
    SchedCand anc[5];
    schedF_greedy16(n, anc[0].rad, &anc[0].k);
    anc[1] = anc[0];
    for (int i = 0; i < anc[1].k; ++i) {
        anc[1].rad[i] = anc[0].rad[anc[1].k - 1 - i];
    }
    {
        int rem = n;
        int k = 0;
        while (rem > 0) {
            const int r = (rem >= 3) ? 8 : (1 << rem);
            anc[2].rad[k++] = r;
            rem -= (r == 8) ? 3 : rem;
        }
        anc[2].k = k;
    }
    schedF_greedy32(n, anc[3].rad, &anc[3].k);
    anc[4] = anc[3];
    for (int i = 0; i < anc[4].k; ++i) {
        anc[4].rad[i] = anc[3].rad[anc[4].k - 1 - i];
    }
    for (int a = 0; a < 5; ++a) {
        if (!schedF_valid(anc[a], n, noR1_32)) {
            continue;
        }
        bool dup = false;
        for (const auto &c : base) {
            if (sched_same(c, anc[a])) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            base.push_back(anc[a]);
        }
    }
    {
        /* per-r1 quota: 9 per first-radix class (r1 in {2,4,8,16,32}) */
        const int quotaCls[6] = {0, 0, 9, 9, 9, 9}; /* idx = log2(r1) */
        int taken[6] = {0, 0, 0, 0, 0, 0};
        for (const auto &c : all) {
            const int cls = ilog2(c.rad[0]);
            if (cls > 5 || taken[cls] >= quotaCls[cls]) {
                continue;
            }
            if (!schedF_valid(c, n, noR1_32)) {
                continue;
            }
            bool dup = false;
            for (const auto &p : base) {
                if (sched_same(p, c)) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                base.push_back(c);
                ++taken[cls];
            }
        }
    }

    /* R1 tile budgets in ELEMENTS (fp32 = 8B/element): 4096/1024/2048
     * (R0 set) + 512 (4KB mid-band L1 tiles) + off. */
    const int tileModes[5] = {4096, 1024, 2048, 512, 0};
    std::vector<SchedCand> pool;
    for (const auto &b : base) {
        if (forcedTile >= 0) {
            SchedCand c = b;
            c.tile = forcedTile;
            pool.push_back(c);
            continue;
        }
        int seen[5][8];
        int nSeen = 0;
        for (int m = 0; m < 5; ++m) {
            int gl[8];
            sched_groups_of(b.rad, b.k, tileModes[m], gl);
            bool dup = false;
            for (int p = 0; p < nSeen; ++p) {
                if (memcmp(seen[p], gl, sizeof(gl)) == 0) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            memcpy(seen[nSeen++], gl, sizeof(gl));
            SchedCand c = b;
            c.tile = tileModes[m];
            pool.push_back(c);
        }
    }
    *poolOut = (int)pool.size();

    float *io = static_cast<float *>(aligned_alloc(64, (size_t)N * 8));
    float *master = static_cast<float *>(aligned_alloc(64, (size_t)N * 8));
    float *scratch = static_cast<float *>(aligned_alloc(64, (size_t)N * 8));
    if (io == nullptr || master == nullptr || scratch == nullptr) {
        free(io);
        free(master);
        free(scratch);
        return false;
    }
    {
        uint32_t st = 0x5eed1234u ^ (uint32_t)N;
        for (int i = 0; i < 2 * N; ++i) {
            st = st * 1664525u + 1013904223u;
            master[i] = (float)(st >> 11) / 2097152.0f - 0.5f;
        }
    }
    memset(io, 0, (size_t)N * 8);
    memset(scratch, 0, (size_t)N * 8);

    size_t arenaSz = 0;
    for (const auto &c : pool) {
        const size_t need = schedF_lut_bytes(N, c.rad, c.k);
        if (need > arenaSz) {
            arenaSz = need;
        }
    }
    float *arena =
        static_cast<float *>(aligned_alloc(64, arenaSz ? arenaSz : 64));
    if (arena == nullptr) {
        free(io);
        free(master);
        free(scratch);
        return false;
    }
    memset(arena, 0, arenaSz);

    const long iters = N <= 512 ? 40 : N <= 2048 ? 16 : N <= 8192 ? 6 : 3;
    const double budgetNs = 4.0e6 + 11.0e6 * ((double)N / 32768.0);
    const double t0 = sched_now_ns();

    /* time one candidate (build + warm + min over rd rounds) */
    auto timeCand = [&](const SchedCand &c, int rounds, double *nsOut,
                        bool *okOut) -> void {
        SchedFBuilt b;
        *okOut = true;
        try {
            b = schedFBuildKernel(N, n, c.rad, c.k, hasFma, scratch, c.tile,
                                  nt, prefetch, nullptr, nullptr, arena);
        } catch (...) {
            *okOut = false;
            return;
        }
        if (b.em == nullptr) {
            *okOut = false;
            return;
        }
        b.em->readyRE();
        fft_jit_fn_t fn = reinterpret_cast<fft_jit_fn_t>(
            const_cast<uint8_t *>(b.em->getCode()));
        memcpy(io, master, (size_t)N * 8);
        fn(io);
        fn(io); /* warm */
        double bestC = 1e30;
        for (int rd = 0; rd < rounds; ++rd) {
            const double s = sched_now_ns();
            for (long i = 0; i < iters; ++i) {
                fn(io);
            }
            const double per = (sched_now_ns() - s) / (double)iters;
            if (per < bestC) {
                bestC = per;
            }
        }
        *nsOut = bestC;
        delete b.em; /* lut + scratch borrowed */
    };

    std::vector<std::pair<double, int>> timedRes; /* (ns, pool idx) */
    double bestNs = 1e30;
    int bestIdx = -1;
    int timed = 0;
    for (size_t ci = 0; ci < pool.size(); ++ci) {
        if (bestIdx >= 0 && timed >= 3 &&
            (sched_now_ns() - t0) > budgetNs) {
            break;
        }
        double ns = 1e30;
        bool ok = false;
        timeCand(pool[ci], 3, &ns, &ok);
        if (!ok) {
            continue;
        }
        timedRes.push_back({ns, (int)ci});
        if (ns < bestNs) {
            bestNs = ns;
            bestIdx = (int)ci;
        }
        ++timed;
    }

    /* ---- R1 L-D diagnosis + mid-band refinement (F5) ---- */
    *refinedOut = 0;
    const bool midN = (N >= 512 && N <= 4096);
    const bool diagOn = getenv("FFT_SCHED_DIAG") != nullptr ||
                        getenv("FFT_SCHED_LOG") != nullptr;
    std::sort(timedRes.begin(), timedRes.end());
    if (midN && timedRes.size() >= 5) {
        /* top-5 short-test spread vs R0 noise bit (~5%): a spread below
         * the noise floor means the short test cannot separate the top
         * candidates ("short-test mis-pick" risk) -> re-time the top-5
         * with 3x rounds and keep the refined winner. */
        const double topGap =
            timedRes[4].first / timedRes[0].first - 1.0;
        if (diagOn) {
            fprintf(stderr,
                    "[schedf-diag] N=%d pre-refine top5 gap=%.1f%%:", N,
                    topGap * 100.0);
            for (int i = 0; i < 5; ++i) {
                const SchedCand &c = pool[timedRes[i].second];
                fprintf(stderr, " %d[", i + 1);
                for (int t = 0; t < c.k; ++t) {
                    fprintf(stderr, "%s%d", t ? "x" : "", c.rad[t]);
                }
                fprintf(stderr, "/t%d]=%.0fns", c.tile, timedRes[i].first);
            }
            fprintf(stderr, "\n");
        }
        if (topGap < 0.05) {
            double bestR = 1e30;
            int bestRIdx = -1;
            for (int i = 0; i < 5; ++i) {
                const int ci = timedRes[i].second;
                double ns = 1e30;
                bool ok = false;
                timeCand(pool[ci], 9, &ns, &ok);
                if (!ok) {
                    continue;
                }
                if (diagOn) {
                    const SchedCand &c = pool[ci];
                    fprintf(stderr,
                            "[schedf-diag] N=%d refine %d[", N, i + 1);
                    for (int t = 0; t < c.k; ++t) {
                        fprintf(stderr, "%s%d", t ? "x" : "", c.rad[t]);
                    }
                    fprintf(stderr, "/t%d] 3r=%.0f -> 9r=%.0f\n", c.tile,
                            timedRes[i].first, ns);
                }
                if (ns < bestR) {
                    bestR = ns;
                    bestRIdx = ci;
                }
            }
            if (bestRIdx >= 0) {
                bestIdx = bestRIdx;
                *refinedOut = 1;
            }
        }
    }

    *searchNsOut = sched_now_ns() - t0;
    *timedOut = timed;
    free(arena);
    free(io);
    free(master);
    free(scratch);
    if (bestIdx < 0) {
        return false;
    }
    for (int i = 0; i < pool[bestIdx].k; ++i) {
        radices[i] = pool[bestIdx].rad[i];
    }
    *kStagesOut = pool[bestIdx].k;
    *tileOut = pool[bestIdx].tile;
    return true;
}

struct SchedFPlan {
    int radices[8];
    int kStages;
    int tile;
    double searchNs;
    int timed;
    int poolSz;
    int genSz;
    bool forced;
    int refined; /* R1 L-D: 1 = mid-band top-5 refinement fired */
};

/* knob parsing mirrors the pd framework: FFT_SCHED_ORDER (16first |
 * smallfirst | explicit radix list), FFT_SCHED_SEARCH=off, FFT_SCHED_BLOCK
 * (off | auto = 4096 elements | explicit). */
static bool schedF_pick(int N, int n, bool hasFma, bool nt, int prefetch,
                        bool noR1_32, SchedFPlan &out) {
    out.kStages = 0;
    out.tile = -1;
    out.searchNs = 0.0;
    out.timed = 0;
    out.poolSz = 0;
    out.genSz = 0;
    out.forced = false;
    out.refined = 0;
    const char *ord = getenv("FFT_SCHED_ORDER");
    if (ord != nullptr && *ord != '\0') {
        if (strcmp(ord, "smallfirst") == 0) {
            schedF_greedy16(n, out.radices, &out.kStages);
            for (int a = 0; a < out.kStages / 2; ++a) {
                const int tmp = out.radices[a];
                out.radices[a] = out.radices[out.kStages - 1 - a];
                out.radices[out.kStages - 1 - a] = tmp;
            }
            out.forced = true;
        } else if (strcmp(ord, "16first") == 0 ||
                   strcmp(ord, "32first") == 0) {
            schedF_greedy16(n, out.radices, &out.kStages);
            out.forced = true;
        } else {
            int rr[8];
            int nr = 0;
            long prod = 1;
            bool ok = true;
            const char *s = ord;
            while (*s != '\0') {
                char *end = nullptr;
                const long v = strtol(s, &end, 10);
                if (end == s || nr >= 8 ||
                    (v != 2 && v != 4 && v != 8 && v != 16 && v != 32)) {
                    ok = false;
                    break;
                }
                rr[nr++] = (int)v;
                prod *= v;
                s = end;
                if (*s == ',') {
                    ++s;
                } else if (*s != '\0') {
                    ok = false;
                    break;
                }
            }
            SchedCand probe;
            if (ok && prod == N && nr >= 1) {
                for (int i = 0; i < nr; ++i) {
                    probe.rad[i] = rr[i];
                }
                probe.k = nr;
                if (schedF_valid(probe, n, noR1_32)) {
                    for (int i = 0; i < nr; ++i) {
                        out.radices[i] = rr[i];
                    }
                    out.kStages = nr;
                    out.forced = true;
                }
            }
        }
    }
    int forcedTile = -1;
    const char *blk = getenv("FFT_SCHED_BLOCK");
    if (blk != nullptr && *blk != '\0') {
        if (strcmp(blk, "off") == 0 || strcmp(blk, "0") == 0) {
            forcedTile = 0;
        } else if (strcmp(blk, "auto") == 0) {
            forcedTile = 4096;
        } else {
            char *end = nullptr;
            const long v = strtol(blk, &end, 10);
            if (end != blk && *end == '\0' && v >= 64 && v <= 32768) {
                forcedTile = (int)v;
            }
        }
    }
    if (!out.forced && n >= 6) {
        const char *srch = getenv("FFT_SCHED_SEARCH");
        const bool off =
            srch != nullptr && (strcmp(srch, "0") == 0 ||
                                strcmp(srch, "off") == 0);
        if (!off) {
            schedF_search(N, n, hasFma, nt, prefetch, forcedTile, noR1_32,
                          out.radices, &out.kStages, &out.tile,
                          &out.searchNs, &out.timed, &out.poolSz,
                          &out.genSz, &out.refined);
        }
    }
    if (out.kStages == 0) {
        schedF_greedy16(n, out.radices, &out.kStages);
    }
    {
        SchedCand probe;
        for (int i = 0; i < out.kStages; ++i) {
            probe.rad[i] = out.radices[i];
        }
        probe.k = out.kStages;
        if (!schedF_valid(probe, n, noR1_32)) {
            /* greedy16 (e.g. N=32 r1=16 -> 2 blocks) falls back to the
             * greedy 8-first chain, then a 4-first chain. */
            int rem = n;
            int k = 0;
            while (rem > 0) {
                const int r = (rem >= 3) ? 8 : (1 << rem);
                out.radices[k++] = r;
                rem -= (r == 8) ? 3 : rem;
            }
            out.kStages = k;
            probe.k = k;
            for (int i = 0; i < k; ++i) {
                probe.rad[i] = out.radices[i];
            }
            if (!schedF_valid(probe, n)) {
                int rem2 = n;
                int k2 = 0;
                while (rem2 > 0) {
                    const int r = (rem2 >= 2) ? 4 : 2;
                    out.radices[k2++] = r;
                    rem2 -= (r == 4) ? 2 : 1;
                }
                out.kStages = k2;
            }
        }
    }
    if (out.tile < 0) {
        out.tile = forcedTile >= 0 ? forcedTile : 0;
    }
    return out.kStages >= 1;
}

} // namespace

static int fill_gauss_window(int N, float *out);

/* ========================================================================
 * topic jit-joint-search-batched (L-E + L-F): plan-time JOINT searches.
 *
 * L-E: the fusion PLACEMENT {store boundary, load boundary, standalone}
 *      joins the plan-time search space as a THIRD dimension on top of
 *      the base "decomposition x tile" two dims (first instance = the
 *      conv H pointwise multiply; the batched window multiply joins the
 *      same pool via the batch search below).  The analytic prescreen
 *      models only the coarse placement delta (boundary multiply ~ 0
 *      extra sweep, just table layout; standalone ~ +1 full N-element
 *      sweep = +2N 8B accesses) — it ONLY orders the pool; every
 *      judgment is a measured short test (R1 lesson: a wider pool is
 *      not a win until the timer says so).  Placement-class round-robin
 *      + per-class best tracking keep every class reaching the timer
 *      under the budget gate (short-test resolution guard), and the
 *      base top-5 refinement re-test is retained.
 * L-F: the batched windowed-forward kernel jointly searches
 *      (placement {load, standalone} x decomposition x tile x batch form
 *      {single-core in-kernel block loop, per-block host loop}) for the
 *      given (N, B); B enters the model through the amortized short test
 *      (call-overhead and cross-block table warmth are what the form
 *      dimension trades).
 * ======================================================================== */

static fft_jit_search_stats g_conv_search_stats;
static fft_jit_search_stats g_batch_search_stats;

extern "C" const fft_jit_search_stats *fft_jit_conv_search_last(void) {
    return &g_conv_search_stats;
}

extern "C" const fft_jit_search_stats *fft_jit_batch_search_last(void) {
    return &g_batch_search_stats;
}

/* LUT bytes a joint candidate needs (the base schedF_lut_bytes ignores
 * the gt / H table regions the fused forms add). */
static size_t joint_lut_bytes(int N, const int *radices, int kStages,
                              bool windowed, bool hTables) {
    int B[8];
    B[0] = radices[0];
    for (int t = 1; t < kStages; ++t) {
        B[t] = B[t - 1] * radices[t];
    }
    const size_t tabBytes =
        (kStages > 1)
            ? (((size_t)(N / radices[0]) * 4 + 63u) & ~63u)
            : 0;
    const size_t gtBytes = windowed ? schedF_pad64((size_t)N * 16) : 0;
    const size_t twBytes =
        schedF_pad64(schedF_tw_bytes(N, radices, kStages));
    const size_t hBytes = hTables ? 2 * schedF_pad64((size_t)N * 8) : 0;
    return tabBytes + gtBytes + twBytes + hBytes + 64;
}

/* The base (decomposition x tile) candidate pool — same anchors +
 * per-r1 quota + tile dedup as the base schedF_search, factored out for
 * the joint searches (schedF_search itself is untouched). */
static std::vector<SchedCand> schedF_joint_base_pool(int n, int forcedTile) {
    std::vector<SchedCand> all;
    int cur[8];
    schedF_gen(n, cur, 0, all);
    for (auto &c : all) {
        c.model = schedF_model(c.rad, c.k);
    }
    std::stable_sort(all.begin(), all.end(),
                     [](const SchedCand &a, const SchedCand &b) {
                         return a.model < b.model;
                     });

    std::vector<SchedCand> base;
    SchedCand anc[5];
    schedF_greedy16(n, anc[0].rad, &anc[0].k);
    anc[1] = anc[0];
    for (int i = 0; i < anc[1].k; ++i) {
        anc[1].rad[i] = anc[0].rad[anc[1].k - 1 - i];
    }
    {
        int rem = n;
        int k = 0;
        while (rem > 0) {
            const int r = (rem >= 3) ? 8 : (1 << rem);
            anc[2].rad[k++] = r;
            rem -= (r == 8) ? 3 : rem;
        }
        anc[2].k = k;
    }
    schedF_greedy32(n, anc[3].rad, &anc[3].k);
    anc[4] = anc[3];
    for (int i = 0; i < anc[4].k; ++i) {
        anc[4].rad[i] = anc[3].rad[anc[4].k - 1 - i];
    }
    for (int a = 0; a < 5; ++a) {
        if (!schedF_valid(anc[a], n)) {
            continue;
        }
        bool dup = false;
        for (const auto &c : base) {
            if (sched_same(c, anc[a])) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            base.push_back(anc[a]);
        }
    }
    {
        const int quotaCls[6] = {0, 0, 9, 9, 9, 9}; /* idx = log2(r1) */
        int taken[6] = {0, 0, 0, 0, 0, 0};
        for (const auto &c : all) {
            const int cls = ilog2(c.rad[0]);
            if (cls > 5 || taken[cls] >= quotaCls[cls]) {
                continue;
            }
            if (!schedF_valid(c, n)) {
                continue;
            }
            bool dup = false;
            for (const auto &p : base) {
                if (sched_same(p, c)) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                base.push_back(c);
                ++taken[cls];
            }
        }
    }

    const int tileModes[5] = {4096, 1024, 2048, 512, 0};
    std::vector<SchedCand> pool;
    for (const auto &b : base) {
        if (forcedTile >= 0) {
            SchedCand c = b;
            c.tile = forcedTile;
            pool.push_back(c);
            continue;
        }
        int seen[5][8];
        int nSeen = 0;
        for (int m = 0; m < 5; ++m) {
            int gl[8];
            sched_groups_of(b.rad, b.k, tileModes[m], gl);
            bool dup = false;
            for (int p = 0; p < nSeen; ++p) {
                if (memcmp(seen[p], gl, sizeof(gl)) == 0) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            memcpy(seen[nSeen++], gl, sizeof(gl));
            SchedCand c = b;
            c.tile = tileModes[m];
            pool.push_back(c);
        }
    }
    return pool;
}

static void joint_read_knobs(bool *nt, bool *pf, int *forcedTile) {
    *nt = false;
    *pf = false;
    *forcedTile = -1;
    const char *nts = getenv("FFT_NT_STORE");
    if (nts != nullptr && (strcmp(nts, "1") == 0 || strcmp(nts, "on") == 0)) {
        *nt = true;
    }
    const char *pfs = getenv("FFT_PREFETCH");
    if (pfs != nullptr && (strcmp(pfs, "1") == 0 || strcmp(pfs, "on") == 0 ||
                           strcmp(pfs, "tile") == 0)) {
        *pf = true;
    }
    const char *blk = getenv("FFT_SCHED_BLOCK");
    if (blk != nullptr && *blk != '\0') {
        if (strcmp(blk, "off") == 0 || strcmp(blk, "0") == 0) {
            *forcedTile = 0;
        } else if (strcmp(blk, "auto") == 0) {
            *forcedTile = 4096;
        } else {
            char *end = nullptr;
            const long v = strtol(blk, &end, 10);
            if (end != blk && *end == '\0' && v >= 64 && v <= 32768) {
                *forcedTile = (int)v;
            }
        }
    }
}

/* Mirror of the host's unfused three-segment conv composition (the
 * standalone placement's execute path) — same sweep shapes, -O2. */
static void joint_threeseg_conv(fft_jit_fn_t fn, const float *H, float *io,
                                int N) {
    fn(io);
    for (int i = 0; i < N; ++i) {
        const float xr = io[2 * i];
        const float xi = io[2 * i + 1];
        const float hr = H[2 * i];
        const float hi = H[2 * i + 1];
        io[2 * i] = xr * hr - xi * hi;
        io[2 * i + 1] = xr * hi + xi * hr;
    }
    for (int i = 0; i < N; ++i) {
        io[2 * i + 1] = -io[2 * i + 1];
    }
    fn(io);
    const float invN = 1.0f / (float)N;
    for (int i = 0; i < N; ++i) {
        io[2 * i + 1] = -io[2 * i + 1];
        io[2 * i] *= invN;
        io[2 * i + 1] *= invN;
    }
}

/* Standalone window sweep over one block (float-pair loop; the compiler
 * vectorizes it — the same shape the host wrapper runs). */
static void joint_window_sweep(const float *G2, float *blk, int N) {
    for (int i = 0; i < 2 * N; ++i) {
        blk[i] *= G2[i];
    }
}

/* topic R2 (amend r2, DELTA F5 / L-G): batch-level STREAMING pre-sweep —
 * ONE contiguous pass over all B blocks (the stride == N layout; the
 * search times exactly that layout), two blocks interleaved to keep the
 * multiply loop wide.  The composed leg's window-sweep shape, but it
 * rides a single execute (no per-block call face). */
static void joint_window_sweep_batch(const float *G2, float *io, int N,
                                     long B) {
    const long n2 = 2L * (long)N;
    long b = 0;
    for (; b + 2 <= B; b += 2) {
        float *p0 = io + b * n2;
        float *p1 = p0 + n2;
        for (long i = 0; i < n2; ++i) {
            p0[i] *= G2[i];
            p1[i] *= G2[i];
        }
    }
    for (; b < B; ++b) {
        float *p = io + b * n2;
        for (long i = 0; i < n2; ++i) {
            p[i] *= G2[i];
        }
    }
}

/* Wrap a freshly-built SchedFBuilt (owning its lut/scratch) into a
 * fft_jit_kernel, or tear it down. */
static fft_jit_kernel *joint_wrap_kernel(SchedFBuilt &b, int windowed,
                                         int batchSingle) {
    fft_jit_kernel *k = new (std::nothrow) fft_jit_kernel();
    if (k == nullptr) {
        delete b.em;
        if (b.ownsLut) {
            free(b.lut);
        }
        if (b.ownsScratch) {
            free(b.scratch);
        }
        return nullptr;
    }
    k->emitter = b.em;
    k->lut = b.lut;
    k->lut_d = nullptr;
    k->win = nullptr;
    k->win_dup = nullptr;
    k->scratch_d = nullptr;
    k->scratch_f = b.scratch;
    k->fn = nullptr;
    k->fn2 = nullptr;
    k->fn2c = nullptr;
    k->batch_single = 0;
    k->fn2 = nullptr;
    k->code_size = 0;
    k->windowed = windowed ? 1 : 0;
    k->precision = 0;
    k->batch_single = batchSingle ? 1 : 0;
    try {
        k->emitter->readyRE();
    } catch (...) {
        fft_jit_kernel_destroy(k);
        return nullptr;
    }
    if (batchSingle) {
        k->fn2 = reinterpret_cast<fft_jit_fn2_t>(
            const_cast<uint8_t *>(k->emitter->getCode()));
        if (b.slim) {
            /* topic R2 (F5/L-G): the slim dual-entry form — expose the
             * contiguous stride==N-specialized entry next to the generic
             * runtime-stride fn2 (the wrapper dispatches on stride==N). */
            k->fn2c = reinterpret_cast<fft_jit_fn1c_t>(
                const_cast<uint8_t *>(k->emitter->getCode()) +
                b.em->contigEntryOff());
        }
    } else {
        k->fn = reinterpret_cast<fft_jit_fn_t>(
            const_cast<uint8_t *>(k->emitter->getCode()));
    }
    k->code_size = static_cast<int>(k->emitter->getSize());
    return k;
}

/* ---------------- L-E: conv placement joint search ---------------- */

static fft_jit_kernel *joint_conv_create(int N, const float *H,
                                         int forcedPlace) {
    const int n = ilog2(N);
    const bool hasFma =
        cpu_has_fma() && getenv("FFT_SCHED_NOFMA") == nullptr;
    bool nt, pf;
    int forcedTile;
    joint_read_knobs(&nt, &pf, &forcedTile);

    fft_jit_search_stats &st = g_conv_search_stats;
    memset(&st, 0, sizeof(st));

    int classes[3];
    int nCls = 0;
    if (forcedPlace >= 0) {
        classes[nCls++] = forcedPlace;
    } else {
        const char *js = getenv("FFT_JOINT_SEARCH");
        const bool off =
            js != nullptr && (strcmp(js, "0") == 0 || strcmp(js, "off") == 0);
        if (off) {
            classes[nCls++] = FFT_PLACE_STORE; /* base two-dim behavior */
        } else {
            classes[nCls++] = FFT_PLACE_STORE;
            classes[nCls++] = FFT_PLACE_LOAD;
            classes[nCls++] = FFT_PLACE_STANDALONE;
        }
    }

    std::vector<SchedCand> pool[3];
    {
        auto base = schedF_joint_base_pool(n, forcedTile);
        for (const auto &c : base) {
            for (int e = 0; e < nCls; ++e) {
                if (classes[e] == FFT_PLACE_LOAD && c.rad[0] == 32) {
                    continue; /* no pair-gather H-row form */
                }
                pool[e].push_back(c);
            }
        }
    }
    int poolTotal = 0;
    for (int e = 0; e < nCls; ++e) {
        poolTotal += (int)pool[e].size();
    }
    if (poolTotal == 0) {
        return nullptr;
    }

    float *io = static_cast<float *>(aligned_alloc(64, (size_t)N * 8));
    float *master = static_cast<float *>(aligned_alloc(64, (size_t)N * 8));
    float *scratch = static_cast<float *>(aligned_alloc(64, (size_t)N * 8));
    if (io == nullptr || master == nullptr || scratch == nullptr) {
        free(io);
        free(master);
        free(scratch);
        return nullptr;
    }
    {
        uint32_t s = 0x5eed5678u ^ (uint32_t)N;
        for (int i = 0; i < 2 * N; ++i) {
            s = s * 1664525u + 1013904223u;
            master[i] = (float)(s >> 11) / 2097152.0f - 0.5f;
        }
    }
    memset(io, 0, (size_t)N * 8);
    memset(scratch, 0, (size_t)N * 8);

    size_t arenaSz = 0;
    for (int e = 0; e < nCls; ++e) {
        for (const auto &c : pool[e]) {
            const size_t need = joint_lut_bytes(N, c.rad, c.k, false,
                                                classes[e] !=
                                                    FFT_PLACE_STANDALONE);
            if (need > arenaSz) {
                arenaSz = need;
            }
        }
    }
    float *arena =
        static_cast<float *>(aligned_alloc(64, arenaSz ? arenaSz : 64));
    if (arena == nullptr) {
        free(io);
        free(master);
        free(scratch);
        return nullptr;
    }
    memset(arena, 0, arenaSz);

    const long iters = N <= 512 ? 32 : N <= 2048 ? 12 : N <= 8192 ? 5 : 3;
    /* H-A(a): the joint search runs under the SAME budget gate formula
     * as the base two-dim search (placement classes compete for the
     * timer via round-robin + quotas inside the budget, not by raising
     * it) — structurally bounds joint search cost to ~single-dim cost. */
    const double budgetNs = 4.0e6 + 11.0e6 * ((double)N / 32768.0);

    auto timeCand = [&](int place, const SchedCand &c, int rounds,
                        double *nsOut, bool *okOut) -> void {
        SchedFBuilt b;
        *okOut = true;
        try {
            if (place == FFT_PLACE_STANDALONE) {
                b = schedFBuildKernel(N, n, c.rad, c.k, hasFma, scratch,
                                      c.tile, nt, pf, nullptr, nullptr,
                                      arena);
            } else {
                b = schedFBuildKernel(N, n, c.rad, c.k, hasFma, scratch,
                                      c.tile, nt, pf, nullptr, H, arena,
                                      place == FFT_PLACE_LOAD ? 1 : 0);
            }
        } catch (...) {
            *okOut = false;
            return;
        }
        if (b.em == nullptr) {
            *okOut = false;
            return;
        }
        b.em->readyRE();
        fft_jit_fn_t fn = reinterpret_cast<fft_jit_fn_t>(
            const_cast<uint8_t *>(b.em->getCode()));
        memcpy(io, master, (size_t)N * 8);
        if (place == FFT_PLACE_STANDALONE) {
            joint_threeseg_conv(fn, H, io, N);
            joint_threeseg_conv(fn, H, io, N); /* warm */
        } else {
            fn(io);
            fn(io); /* warm */
        }
        double bestC = 1e30;
        for (int rd = 0; rd < rounds; ++rd) {
            const double s0 = sched_now_ns();
            for (long i = 0; i < iters; ++i) {
                if (place == FFT_PLACE_STANDALONE) {
                    joint_threeseg_conv(fn, H, io, N);
                } else {
                    fn(io);
                }
            }
            const double per = (sched_now_ns() - s0) / (double)iters;
            if (per < bestC) {
                bestC = per;
            }
        }
        *nsOut = bestC;
        delete b.em; /* lut + scratch borrowed */
    };

    struct TEntry {
        double ns;
        int e;
        int ci;
    };
    std::vector<TEntry> timedRes;
    double bestPerCls[3] = {1e30, 1e30, 1e30};
    double bestNs = 1e30;
    int bestE = -1;
    int bestCi = -1;
    int timed = 0;
    size_t cursor[3] = {0, 0, 0};
    int rr = 0;
    const double t0 = sched_now_ns();
    for (;;) {
        int pick = -1;
        for (int q = 0; q < nCls; ++q) {
            const int e = (rr + q) % nCls;
            if (cursor[e] < pool[e].size()) {
                pick = e;
                break;
            }
        }
        if (pick < 0) {
            break;
        }
        ++rr;
        if (bestE >= 0 && timed >= 3 &&
            (sched_now_ns() - t0) > budgetNs) {
            break;
        }
        const int ci = (int)cursor[pick];
        ++cursor[pick];
        double ns = 1e30;
        bool ok = false;
        timeCand(classes[pick], pool[pick][ci], 3, &ns, &ok);
        if (!ok) {
            continue;
        }
        timedRes.push_back({ns, pick, ci});
        if (ns < bestPerCls[pick]) {
            bestPerCls[pick] = ns;
        }
        if (ns < bestNs) {
            bestNs = ns;
            bestE = pick;
            bestCi = ci;
        }
        ++timed;
    }

    /* top-5 refinement (base R1 pattern): mid-N, spread below the ~5%
     * short-test noise bit -> re-time the top-5 with 3x rounds. */
    st.refined = 0;
    const bool midN = (N >= 512 && N <= 4096);
    std::sort(timedRes.begin(), timedRes.end(),
              [](const TEntry &a, const TEntry &b) { return a.ns < b.ns; });
    if (midN && timedRes.size() >= 5) {
        const double topGap = timedRes[4].ns / timedRes[0].ns - 1.0;
        if (topGap < 0.05) {
            double bestR = 1e30;
            int bestRE = -1;
            int bestRCi = -1;
            for (int i = 0; i < 5; ++i) {
                const TEntry &te = timedRes[i];
                double ns = 1e30;
                bool ok = false;
                timeCand(classes[te.e], pool[te.e][te.ci], 9, &ns, &ok);
                if (!ok) {
                    continue;
                }
                if (ns < bestR) {
                    bestR = ns;
                    bestRE = te.e;
                    bestRCi = te.ci;
                }
            }
            if (bestRE >= 0) {
                bestE = bestRE;
                bestCi = bestRCi;
                st.refined = 1;
            }
        }
    }

    st.search_ns = sched_now_ns() - t0;
    st.timed = timed;
    st.pool = poolTotal;
    st.forced = (forcedPlace >= 0 || nCls == 1) ? 1 : 0;
    free(arena);
    free(io);
    free(master);
    free(scratch);
    if (bestE < 0) {
        return nullptr;
    }
    const SchedCand &win = pool[bestE][bestCi];
    for (int i = 0; i < win.k; ++i) {
        st.radices[i] = win.rad[i];
    }
    st.k = win.k;
    st.tile = win.tile;
    st.place = classes[bestE];
    for (int e = 0; e < nCls; ++e) {
        st.best_place[classes[e]] = bestPerCls[e];
    }

    if (getenv("FFT_SCHED_LOG") != nullptr) {
        fprintf(stderr, "[joint-conv] N=%d place=%s plan=", N,
                st.place == FFT_PLACE_STORE
                    ? "store"
                    : st.place == FFT_PLACE_LOAD ? "load" : "standalone");
        for (int t = 0; t < st.k; ++t) {
            fprintf(stderr, "%s%d", t ? "x" : "", st.radices[t]);
        }
        fprintf(stderr,
                "/t%d pool=%d timed=%d search_ns=%.0f refined=%d "
                "best[store=%.0f load=%.0f sa=%.0f]%s\n",
                st.tile, st.pool, st.timed, st.search_ns, st.refined,
                st.best_place[0], st.best_place[1], st.best_place[2],
                st.forced ? " (forced)" : "");
    }

    SchedFBuilt b;
    try {
        if (st.place == FFT_PLACE_STANDALONE) {
            b = schedFBuildKernel(N, n, st.radices, st.k, hasFma, nullptr,
                                  st.tile, nt, pf, nullptr, nullptr);
        } else {
            b = schedFBuildKernel(N, n, st.radices, st.k, hasFma, nullptr,
                                  st.tile, nt, pf, nullptr, H, nullptr,
                                  st.place == FFT_PLACE_LOAD ? 1 : 0);
        }
    } catch (...) {
        return nullptr;
    }
    return joint_wrap_kernel(b, 0, 0);
}

/* ---------------- L-F: batched windowed joint search ---------------- */

/* special timedRes slots (>= SLOT_LEGACY are skipped by the top-5
 * re-time: single-shot candidates) */
static const int SLOT_LEGACY = 8;
static const int SLOT_BARE = 9;

static fft_jit_kernel *joint_batch_create(int N, int B) {
    const int n = ilog2(N);
    const bool hasFma =
        cpu_has_fma() && getenv("FFT_SCHED_NOFMA") == nullptr;
    bool nt, pf;
    int forcedTile;
    joint_read_knobs(&nt, &pf, &forcedTile);

    fft_jit_search_stats &st = g_batch_search_stats;
    memset(&st, 0, sizeof(st));
    st.btest = 0;

    /* topic-local debug knobs: pin the placement / batch-form dims.
     * R2 (amend r2, INTERFACE Config): FFT_WINPLACE=gather|presweep|
     * standalone is the window-placement knob (FFT_BATCH_PLACE=load|
     * standalone stays as the legacy alias); FFT_BATCH_FORM gains slim
     * (the R2 slimmed dual-entry single-core form) and bare (the F6/L-H
     * base-shape anchor: base two-dim search decomposition + presweep
     * sweep + per-block plain kernel). */
    int forcedPlace = -1;
    int forcedForm = -1;
    {
        const char *v = getenv("FFT_WINPLACE");
        if (v == nullptr) {
            v = getenv("FFT_BATCH_PLACE"); /* legacy alias */
        }
        if (v != nullptr) {
            if (strcmp(v, "load") == 0 || strcmp(v, "gather") == 0) {
                forcedPlace = FFT_PLACE_LOAD;
            } else if (strcmp(v, "presweep") == 0 ||
                       strcmp(v, "pre") == 0) {
                forcedPlace = FFT_PLACE_PRESWEEP;
            } else if (strcmp(v, "standalone") == 0 ||
                       strcmp(v, "sa") == 0) {
                forcedPlace = FFT_PLACE_STANDALONE;
            }
        }
        const char *w = getenv("FFT_BATCH_FORM");
        if (w != nullptr) {
            if (strcmp(w, "single") == 0) {
                forcedForm = FFT_BATCH_FORM_SINGLE;
            } else if (strcmp(w, "loop") == 0) {
                forcedForm = FFT_BATCH_FORM_LOOP;
            } else if (strcmp(w, "slim") == 0) {
                forcedForm = FFT_BATCH_FORM_SLIM;
            } else if (strcmp(w, "bare") == 0) {
                forcedForm = 3; /* FORM_FORCE_BARE (not a stored form) */
            }
        }
    }

    /* N <= 8 has no schedulable chain (k >= 2 validity): fixed legacy
     * per-block fused windowed kernel (the SmallEmitter gauss form). */
    if (N <= 8) {
        fft_jit_kernel *k = fft_jit_kernel_create_gauss_window(N);
        if (k == nullptr || fft_jit_kernel_fn(k) == nullptr) {
            fft_jit_kernel_destroy(k);
            return nullptr;
        }
        st.place = FFT_PLACE_LOAD;
        st.form = FFT_BATCH_FORM_LOOP;
        st.forced = 1;
        st.k = 0;
        st.tile = 0;
        return k;
    }

    struct BCls {
        int place;
        int form;
    };
    /* R2 (amend r2): the batch-form dimension gains the SLIM single-core
     * variant (DELTA F5 / L-G — hoisted block invariants + dual-entry
     * contiguous stride==N specialization) and the window placement gains
     * PRESWEEP (streaming pre-sweep + windowless kernel).  The four r1
     * incumbent classes stay in the pool unchanged (regression guard on
     * the search selections). */
    const BCls allCls[7] = {
        {FFT_PLACE_LOAD, FFT_BATCH_FORM_SINGLE},
        {FFT_PLACE_LOAD, FFT_BATCH_FORM_SLIM},
        {FFT_PLACE_LOAD, FFT_BATCH_FORM_LOOP},
        {FFT_PLACE_STANDALONE, FFT_BATCH_FORM_SINGLE},
        {FFT_PLACE_STANDALONE, FFT_BATCH_FORM_LOOP},
        {FFT_PLACE_PRESWEEP, FFT_BATCH_FORM_SLIM},
        {FFT_PLACE_PRESWEEP, FFT_BATCH_FORM_LOOP}};
    BCls cls[8];
    int nCls = 0;
    for (int i = 0; i < 7; ++i) {
        if (forcedPlace >= 0 && allCls[i].place != forcedPlace) {
            continue;
        }
        if (forcedForm >= 0 && forcedForm < 3 &&
            allCls[i].form != forcedForm) {
            continue;
        }
        if (forcedForm == 3) {
            continue; /* bare anchor only */
        }
        cls[nCls++] = allCls[i];
    }

    std::vector<SchedCand> pool[8];
    {
        auto base = schedF_joint_base_pool(n, forcedTile);
        for (const auto &c : base) {
            for (int e = 0; e < nCls; ++e) {
                if (cls[e].place == FFT_PLACE_LOAD && c.rad[0] == 32) {
                    continue; /* no pair-gather window form */
                }
                pool[e].push_back(c);
            }
        }
    }
    int poolTotal = 0;
    for (int e = 0; e < nCls; ++e) {
        poolTotal += (int)pool[e].size();
    }

    /* DESIGN L-F: at small N the legacy straight-line windowed kernel
     * (SmallEmitter register-resident body at N <= 16, PoolEmitter at
     * N = 32) joins the pool as an extra (load, per-block-loop)
     * candidate — "small N goes the SmallEmitter straight-line batch
     * form", decided by the same short test. */
    fft_jit_kernel *legacyK = nullptr;
    int legacyOk = 0;
    if (N <= 32 && (forcedPlace < 0 || forcedPlace == FFT_PLACE_LOAD) &&
        (forcedForm < 0 || forcedForm == FFT_BATCH_FORM_LOOP)) {
        legacyK = fft_jit_kernel_create_gauss_window(N);
        if (legacyK != nullptr && fft_jit_kernel_fn(legacyK) != nullptr) {
            legacyOk = 1;
            ++poolTotal;
        } else if (legacyK != nullptr) {
            fft_jit_kernel_destroy(legacyK);
            legacyK = nullptr;
        }
    }

    /* topic R2 (amend r2, DELTA F6 / L-H): the BASE-SHAPE anchor — run
     * the base two-dim (decomposition x tile) single-block search and
     * enter its winner as a GUARANTEED-TIMED batch candidate (presweep
     * sweep + plain per-block kernel = the ps bare-kernel shape), so the
     * batch context can fall back to the bare shape instead of being
     * forced to choose between degraded batch-only forms.  Its full cost
     * (the base search itself) counts into search_ns honestly. */
    const double searchT0 = sched_now_ns();
    SchedFPlan basePlan;
    int haveBare = 0;
    if (forcedForm < 0 || forcedForm == 3) {
        if (schedF_pick(N, n, hasFma, nt, pf, /*noR1_32=*/false, basePlan)) {
            haveBare = 1;
            ++poolTotal;
        }
    }
    if (poolTotal == 0) {
        return nullptr;
    }

    /* G + duplicated-pair sweep table (sigma = N/8, sum = 1). */
    float *G =
        static_cast<float *>(aligned_alloc(64, ((size_t)N * 4 + 63u) & ~63u));
    float *G2 =
        static_cast<float *>(aligned_alloc(64, ((size_t)N * 8 + 63u) & ~63u));
    if (G == nullptr || G2 == nullptr || fill_gauss_window(N, G) != 0) {
        free(G);
        free(G2);
        if (legacyK != nullptr) {
            fft_jit_kernel_destroy(legacyK);
        }
        return nullptr;
    }
    for (int i = 0; i < N; ++i) {
        G2[2 * i] = G[i];
        G2[2 * i + 1] = G[i];
    }

    /* B enters the search through the amortized short test.  R2 (amend
     * r2, DELTA F6 / L-H): the short-test block count keeps an explicit
     * floor of >= 4 blocks (B >= 4) and the large-N caps are widened
     * (16384 band: 4 -> 8) so the per-block amortized number covers the
     * cross-block steady state (table warmth, icache/branch settle,
     * streaming ramp) instead of extrapolating from a couple of blocks;
     * the actual Btest is recorded in the stats. */
    int Btest = B;
    if (N <= 64) {
        Btest = B < 256 ? B : 256;
    } else if (N <= 512) {
        Btest = B < 64 ? B : 64;
    } else if (N <= 4096) {
        Btest = B < 16 ? B : 16;
    } else if (N <= 16384) {
        Btest = B < 8 ? B : 8; /* R2 L-H: >= 4 with steady-state headroom */
    } else {
        Btest = B < 4 ? B : 4;
    }
    if (Btest < 4 && B >= 4) {
        Btest = 4; /* R2 L-H explicit floor */
    }
    if (Btest < 1) {
        Btest = 1;
    }
    st.btest = Btest;

    float *io = static_cast<float *>(
        aligned_alloc(64, (size_t)Btest * (size_t)N * 8));
    float *master = static_cast<float *>(
        aligned_alloc(64, (size_t)Btest * (size_t)N * 8));
    float *scratch = static_cast<float *>(aligned_alloc(64, (size_t)N * 8));
    if (io == nullptr || master == nullptr || scratch == nullptr) {
        free(io);
        free(master);
        free(scratch);
        free(G);
        free(G2);
        if (legacyK != nullptr) {
            fft_jit_kernel_destroy(legacyK);
        }
        return nullptr;
    }
    {
        uint32_t s = 0x5eed9abcu ^ (uint32_t)N ^ (uint32_t)B;
        for (int i = 0; i < 2 * Btest * N; ++i) {
            s = s * 1664525u + 1013904223u;
            master[i] = (float)(s >> 11) / 2097152.0f - 0.5f;
        }
    }
    memset(io, 0, (size_t)Btest * (size_t)N * 8);
    memset(scratch, 0, (size_t)N * 8);

    size_t arenaSz = 0;
    for (int e = 0; e < nCls; ++e) {
        for (const auto &c : pool[e]) {
            const size_t need = joint_lut_bytes(
                N, c.rad, c.k, cls[e].place == FFT_PLACE_LOAD, false);
            if (need > arenaSz) {
                arenaSz = need;
            }
        }
    }
    if (haveBare) {
        const size_t need =
            joint_lut_bytes(N, basePlan.radices, basePlan.kStages, false,
                            false);
        if (need > arenaSz) {
            arenaSz = need;
        }
    }
    float *arena =
        static_cast<float *>(aligned_alloc(64, arenaSz ? arenaSz : 64));
    if (arena == nullptr) {
        free(io);
        free(master);
        free(scratch);
        free(G);
        free(G2);
        if (legacyK != nullptr) {
            fft_jit_kernel_destroy(legacyK);
        }
        return nullptr;
    }
    memset(arena, 0, arenaSz);

    const long iters = N <= 512 ? 4 : N <= 4096 ? 2 : 1;
    const double budgetNs = 2.0 * (4.0e6 + 11.0e6 * ((double)N / 32768.0));

    auto timeCand = [&](int place, int form, const SchedCand &c, int rounds,
                        double *nsOut, bool *okOut) -> void {
        SchedFBuilt b;
        *okOut = true;
        const int batchBlocks =
            (form == FFT_BATCH_FORM_SINGLE || form == FFT_BATCH_FORM_SLIM)
                ? Btest
                : 0;
        const int batchSlim = (form == FFT_BATCH_FORM_SLIM) ? 1 : 0;
        try {
            b = schedFBuildKernel(N, n, c.rad, c.k, hasFma, scratch, c.tile,
                                  nt, pf,
                                  place == FFT_PLACE_LOAD ? G : nullptr,
                                  nullptr, arena, 0, batchBlocks, batchSlim);
        } catch (...) {
            *okOut = false;
            return;
        }
        if (b.em == nullptr) {
            *okOut = false;
            return;
        }
        b.em->readyRE();
        auto runOnce = [&]() {
            if (place == FFT_PLACE_STANDALONE) {
                for (int bi = 0; bi < Btest; ++bi) {
                    joint_window_sweep(G2, io + (size_t)bi * 2 * N, N);
                }
            } else if (place == FFT_PLACE_PRESWEEP) {
                /* one batch-level streaming pass (the search times the
                 * canonical stride == N layout) */
                joint_window_sweep_batch(G2, io, N, (long)Btest);
            }
            if (form == FFT_BATCH_FORM_SINGLE) {
                fft_jit_fn2_t f2 = reinterpret_cast<fft_jit_fn2_t>(
                    const_cast<uint8_t *>(b.em->getCode()));
                f2(io, (long)N);
            } else if (form == FFT_BATCH_FORM_SLIM) {
                /* canonical contiguous entry (stride == N): the exact
                 * code path the wrapper dispatches to */
                fft_jit_fn1c_t f2c = reinterpret_cast<fft_jit_fn1c_t>(
                    const_cast<uint8_t *>(b.em->getCode()) +
                    b.em->contigEntryOff());
                f2c(io);
            } else {
                fft_jit_fn_t f1 = reinterpret_cast<fft_jit_fn_t>(
                    const_cast<uint8_t *>(b.em->getCode()));
                for (int bi = 0; bi < Btest; ++bi) {
                    f1(io + (size_t)bi * 2 * N);
                }
            }
        };
        memcpy(io, master, (size_t)Btest * (size_t)N * 8);
        runOnce();
        runOnce(); /* warm */
        double bestC = 1e30;
        for (int rd = 0; rd < rounds; ++rd) {
            const double s0 = sched_now_ns();
            for (long i = 0; i < iters; ++i) {
                runOnce();
            }
            const double per =
                (sched_now_ns() - s0) / ((double)iters * (double)Btest);
            if (per < bestC) {
                bestC = per;
            }
        }
        *nsOut = bestC;
        delete b.em; /* lut + scratch borrowed */
    };

    struct TEntry {
        double ns;
        int e;
        int ci;
    };
    std::vector<TEntry> timedRes;
    double bestPerPlace[4] = {0.0, 1e30, 1e30, 1e30}; /* [LOAD], [SA],
                                                       * [PRESWEEP] */
    double bestNs = 1e30;
    int bestE = -1;
    int bestCi = -1;
    int timed = 0;
    size_t cursor[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int rr = 0;
    const double t0 = searchT0; /* includes the bare anchor's base search */
    /* the legacy straight-line candidate (slot SLOT_LEGACY): time it
     * first so the budget gate sees it, and let it compete on the same
     * per-block amortized number. */
    if (legacyOk) {
        fft_jit_fn_t fn = fft_jit_kernel_fn(legacyK);
        memcpy(io, master, (size_t)Btest * (size_t)N * 8);
        for (int bi = 0; bi < Btest; ++bi) {
            fn(io + (size_t)bi * 2 * N);
        }
        for (int bi = 0; bi < Btest; ++bi) {
            fn(io + (size_t)bi * 2 * N); /* warm */
        }
        double bestC = 1e30;
        for (int rd = 0; rd < 3; ++rd) {
            const double s0 = sched_now_ns();
            for (long i = 0; i < iters; ++i) {
                for (int bi = 0; bi < Btest; ++bi) {
                    fn(io + (size_t)bi * 2 * N);
                }
            }
            const double per = (sched_now_ns() - s0) /
                               ((double)iters * (double)Btest);
            if (per < bestC) {
                bestC = per;
            }
        }
        timedRes.push_back({bestC, SLOT_LEGACY, 0});
        if (bestC < bestPerPlace[FFT_PLACE_LOAD]) {
            bestPerPlace[FFT_PLACE_LOAD] = bestC;
        }
        if (bestC < bestNs) {
            bestNs = bestC;
            bestE = SLOT_LEGACY;
            bestCi = 0;
        }
        ++timed;
    }

    /* the base bare-shape anchor (slot SLOT_BARE, R2 F6/L-H): presweep
     * sweep + the base-search plain kernel per block — timed right after
     * the legacy slot, before the round-robin, so the budget gate can
     * never starve it. */
    if (haveBare) {
        SchedFBuilt bb = {nullptr, nullptr, nullptr, false, false, 0};
        bool bok = true;
        try {
            bb = schedFBuildKernel(N, n, basePlan.radices, basePlan.kStages,
                                   hasFma, scratch, basePlan.tile, nt, pf,
                                   nullptr, nullptr, arena, 0, 0);
        } catch (...) {
            bok = false;
        }
        if (bok && bb.em != nullptr) {
            bb.em->readyRE();
            fft_jit_fn_t fn = reinterpret_cast<fft_jit_fn_t>(
                const_cast<uint8_t *>(bb.em->getCode()));
            auto runOnce = [&]() {
                joint_window_sweep_batch(G2, io, N, (long)Btest);
                for (int bi = 0; bi < Btest; ++bi) {
                    fn(io + (size_t)bi * 2 * N);
                }
            };
            memcpy(io, master, (size_t)Btest * (size_t)N * 8);
            runOnce();
            runOnce(); /* warm */
            double bestC = 1e30;
            for (int rd = 0; rd < 3; ++rd) {
                const double s0 = sched_now_ns();
                for (long i = 0; i < iters; ++i) {
                    runOnce();
                }
                const double per = (sched_now_ns() - s0) /
                                   ((double)iters * (double)Btest);
                if (per < bestC) {
                    bestC = per;
                }
            }
            timedRes.push_back({bestC, SLOT_BARE, 0});
            if (bestC < bestPerPlace[FFT_PLACE_PRESWEEP]) {
                bestPerPlace[FFT_PLACE_PRESWEEP] = bestC;
            }
            if (bestC < bestNs) {
                bestNs = bestC;
                bestE = SLOT_BARE;
                bestCi = 0;
            }
            ++timed;
        }
        delete bb.em; /* lut + scratch borrowed */
    }

    for (;;) {
        int pick = -1;
        for (int q = 0; q < nCls; ++q) {
            const int e = (rr + q) % nCls;
            if (cursor[e] < pool[e].size()) {
                pick = e;
                break;
            }
        }
        if (pick < 0) {
            break;
        }
        ++rr;
        if (bestE >= 0 && timed >= 3 && (sched_now_ns() - t0) > budgetNs) {
            break;
        }
        const int ci = (int)cursor[pick];
        ++cursor[pick];
        double ns = 1e30;
        bool ok = false;
        timeCand(cls[pick].place, cls[pick].form, pool[pick][ci], 3, &ns,
                 &ok);
        if (!ok) {
            continue;
        }
        timedRes.push_back({ns, pick, ci});
        if (ns < bestPerPlace[cls[pick].place]) {
            bestPerPlace[cls[pick].place] = ns;
        }
        if (ns < bestNs) {
            bestNs = ns;
            bestE = pick;
            bestCi = ci;
        }
        ++timed;
    }

    /* top-5 refinement (short-test resolution guard, base pattern). */
    st.refined = 0;
    std::sort(timedRes.begin(), timedRes.end(),
              [](const TEntry &a, const TEntry &b) { return a.ns < b.ns; });
    if (timedRes.size() >= 5) {
        const double topGap = timedRes[4].ns / timedRes[0].ns - 1.0;
        if (topGap < 0.05) {
            double bestR = 1e30;
            int bestRE = -1;
            int bestRCi = -1;
            for (int i = 0; i < 5; ++i) {
                const TEntry &te = timedRes[i];
                if (te.e >= SLOT_LEGACY) {
                    continue; /* legacy / bare slots: single-shot
                               * candidates, no retime */
                }
                double ns = 1e30;
                bool ok = false;
                timeCand(cls[te.e].place, cls[te.e].form, pool[te.e][te.ci],
                         9, &ns, &ok);
                if (!ok) {
                    continue;
                }
                if (ns < bestR) {
                    bestR = ns;
                    bestRE = te.e;
                    bestRCi = te.ci;
                }
            }
            if (bestRE >= 0 && bestR <= bestNs) {
                bestE = bestRE;
                bestCi = bestRCi;
                bestNs = bestR;
                st.refined = 1;
            }
        }
    }

    st.search_ns = sched_now_ns() - t0;
    st.timed = timed;
    st.pool = poolTotal;
    st.forced = (forcedPlace >= 0 || forcedForm >= 0) ? 1 : 0;
    free(arena);
    free(io);
    free(master);
    free(scratch);
    if (bestE < 0) {
        free(G);
        free(G2);
        if (legacyK != nullptr) {
            fft_jit_kernel_destroy(legacyK);
        }
        return nullptr;
    }
    if (bestE == SLOT_LEGACY) {
        /* the legacy straight-line kernel wins: (load, per-block loop) */
        st.place = FFT_PLACE_LOAD;
        st.form = FFT_BATCH_FORM_LOOP;
        st.k = 0;
        st.tile = 0;
        st.radices[0] = 0;
        st.best_place[FFT_PLACE_LOAD] = bestPerPlace[FFT_PLACE_LOAD];
        st.best_place[FFT_PLACE_STANDALONE] =
            bestPerPlace[FFT_PLACE_STANDALONE];
        st.best_place[FFT_PLACE_PRESWEEP] = bestPerPlace[FFT_PLACE_PRESWEEP];
        free(G);
        free(G2);
        if (getenv("FFT_SCHED_LOG") != nullptr) {
            fprintf(stderr,
                    "[joint-batch] N=%d B=%d place=load form=loop "
                    "plan=legacy pool=%d timed=%d search_ns=%.0f\n",
                    N, B, st.pool, st.timed, st.search_ns);
        }
        return legacyK;
    }
    if (legacyK != nullptr) {
        fft_jit_kernel_destroy(legacyK);
    }
    if (bestE == SLOT_BARE) {
        /* the base bare-shape anchor wins (R2 F6/L-H fallback): presweep
         * sweep + the base-search plain kernel per block. */
        st.place = FFT_PLACE_PRESWEEP;
        st.form = FFT_BATCH_FORM_LOOP;
        st.k = basePlan.kStages;
        st.tile = basePlan.tile;
        for (int i = 0; i < basePlan.kStages && i < 8; ++i) {
            st.radices[i] = basePlan.radices[i];
        }
    } else {
        const SchedCand &win = pool[bestE][bestCi];
        for (int i = 0; i < win.k; ++i) {
            st.radices[i] = win.rad[i];
        }
        st.k = win.k;
        st.tile = win.tile;
        st.place = cls[bestE].place;
        st.form = cls[bestE].form;
    }
    st.best_place[FFT_PLACE_LOAD] = bestPerPlace[FFT_PLACE_LOAD];
    st.best_place[FFT_PLACE_STANDALONE] = bestPerPlace[FFT_PLACE_STANDALONE];
    st.best_place[FFT_PLACE_PRESWEEP] = bestPerPlace[FFT_PLACE_PRESWEEP];

    if (getenv("FFT_SCHED_LOG") != nullptr) {
        const char *pl = st.place == FFT_PLACE_LOAD      ? "load"
                         : st.place == FFT_PLACE_PRESWEEP ? "presweep"
                                                          : "standalone";
        const char *fm = st.form == FFT_BATCH_FORM_SINGLE ? "single"
                         : st.form == FFT_BATCH_FORM_SLIM  ? "slim"
                                                           : "loop";
        fprintf(stderr,
                "[joint-batch] N=%d B=%d (Btest=%d) place=%s form=%s plan=",
                N, B, Btest, pl, fm);
        for (int t = 0; t < st.k; ++t) {
            fprintf(stderr, "%s%d", t ? "x" : "", st.radices[t]);
        }
        fprintf(stderr,
                "/t%d pool=%d timed=%d search_ns=%.0f refined=%d "
                "best[load=%.0f sa=%.0f pre=%.0f]%s\n",
                st.tile, st.pool, st.timed, st.search_ns, st.refined,
                st.best_place[FFT_PLACE_LOAD],
                st.best_place[FFT_PLACE_STANDALONE],
                st.best_place[FFT_PLACE_PRESWEEP],
                st.forced ? " (forced)" : "");
    }

    /* final build: the real B is baked for the single-core forms (the
     * slim form emits the dual generic+contiguous entries); the windowed
     * G rides the plan-owned lut gt rows (load placement);
     * standalone/presweep return the PLAIN kernel (the host sweeps the
     * window before executing it).  G/G2 stay alive until the tables
     * are materialized. */
    const int winBatch =
        (st.form == FFT_BATCH_FORM_SINGLE || st.form == FFT_BATCH_FORM_SLIM)
            ? B
            : 0;
    SchedFBuilt b;
    try {
        b = schedFBuildKernel(N, n, st.radices, st.k, hasFma, nullptr,
                              st.tile, nt, pf,
                              st.place == FFT_PLACE_LOAD ? G : nullptr,
                              nullptr, nullptr, 0, winBatch,
                              st.form == FFT_BATCH_FORM_SLIM ? 1 : 0);
    } catch (...) {
        free(G);
        free(G2);
        return nullptr;
    }
    free(G);
    free(G2);
    return joint_wrap_kernel(b, st.place == FFT_PLACE_LOAD ? 1 : 0,
                             winBatch > 0 ? 1 : 0);
}

extern "C" fft_jit_kernel *fft_jit_kernel_create_conv_searched(
    int N, const float *H_interleaved, int forcedPlace) {
    if (N < 32 || H_interleaved == nullptr) {
        return nullptr;
    }
    if (!is_pow2(N) || N > 32768) {
        return nullptr;
    }
    if (!cpu_has_avx()) {
        return nullptr;
    }
    int fp = forcedPlace;
    if (fp < 0) {
        const char *v = getenv("FFT_CONV_PLACE");
        if (v != nullptr) {
            if (strcmp(v, "store") == 0) {
                fp = FFT_PLACE_STORE;
            } else if (strcmp(v, "load") == 0) {
                fp = FFT_PLACE_LOAD;
            } else if (strcmp(v, "standalone") == 0 || strcmp(v, "sa") == 0) {
                fp = FFT_PLACE_STANDALONE;
            }
        }
    }
    if (fp >= 0 && fp != FFT_PLACE_STORE && fp != FFT_PLACE_LOAD &&
        fp != FFT_PLACE_STANDALONE) {
        return nullptr;
    }
    return joint_conv_create(N, H_interleaved, fp);
}

extern "C" fft_jit_kernel *fft_jit_kernel_create_windowed_batch(int N, int B) {
    if (N < 1 || !is_pow2(N) || N > 32768) {
        return nullptr;
    }
    if (B < 1 || B > (1 << 20)) {
        return nullptr;
    }
    if (!cpu_has_avx()) {
        return nullptr;
    }
    return joint_batch_create(N, B);
}

extern "C" int fft_jit_kernel_is_batch_single(const fft_jit_kernel *k) {
    return (k != nullptr && k->batch_single) ? 1 : 0;
}

extern "C" fft_jit_fn2_t fft_jit_kernel_fn2(const fft_jit_kernel *k) {
    return (k != nullptr) ? k->fn2 : nullptr;
}

extern "C" fft_jit_fn1c_t fft_jit_kernel_fn2c(const fft_jit_kernel *k) {
    return (k != nullptr) ? k->fn2c : nullptr;
}

/* topic jit-ps-sched-fuse: ps SchedEmitterF kernel (plain forward N >= 32,
 * windowed gauss N >= 64, fused conv H != nullptr N >= 32).  NULL on
 * illegal N / emit / OOM — fail-fast, no silent fallback. */
static fft_jit_kernel *schedF_create_kernel(int N, int windowed,
                                            const float *H) {
    if (N < 1 || !is_pow2(N) || N > 32768) {
        return nullptr;
    }
    if (N < 32) {
        return nullptr;
    }
    if (windowed && N < 64) {
        return nullptr; /* gauss N=32 stays on the legacy tier */
    }
    if (windowed && H != nullptr) {
        return nullptr;
    }
    if (!cpu_has_avx()) {
        return nullptr;
    }
    const bool hasFma =
        cpu_has_fma() && getenv("FFT_SCHED_NOFMA") == nullptr;
    const int n = ilog2(N);

    bool ntOn = false;
    bool pfOn = false;
    {
        const char *nts = getenv("FFT_NT_STORE");
        if (nts != nullptr && (strcmp(nts, "1") == 0 ||
                               strcmp(nts, "on") == 0)) {
            ntOn = true;
        }
        const char *pfs = getenv("FFT_PREFETCH");
        if (pfs != nullptr && (strcmp(pfs, "1") == 0 ||
                               strcmp(pfs, "on") == 0 ||
                               strcmp(pfs, "tile") == 0)) {
            pfOn = true;
        }
    }

    fft_jit_kernel *k = new (std::nothrow) fft_jit_kernel();
    if (k == nullptr) {
        return nullptr;
    }
    k->emitter = nullptr;
    k->lut = nullptr;
    k->lut_d = nullptr;
    k->win = nullptr;
    k->win_dup = nullptr;
    k->scratch_d = nullptr;
    k->scratch_f = nullptr;
    k->fn = nullptr;
    k->fn2 = nullptr;
    k->fn2c = nullptr;
    k->batch_single = 0;
    k->code_size = 0;
    k->windowed = windowed ? 1 : 0;
    k->precision = 0;

    if (windowed) {
        const size_t winBytes = ((size_t)N * sizeof(float) + 63u) & ~63u;
        k->win = static_cast<float *>(aligned_alloc(64, winBytes));
        if (k->win == nullptr || fill_gauss_window(N, k->win) != 0) {
            fft_jit_kernel_destroy(k);
            return nullptr;
        }
    }

    SchedFPlan plan;
    if (!schedF_pick(N, n, hasFma, ntOn, pfOn ? 1 : 0, windowed, plan)) {
        fft_jit_kernel_destroy(k);
        return nullptr;
    }

    const float *Hf = nullptr;
    if (H != nullptr) {
        Hf = reinterpret_cast<const float *>(H);
    }

    SchedFBuilt b;
    try {
        b = schedFBuildKernel(N, n, plan.radices, plan.kStages, hasFma,
                              nullptr, plan.tile, ntOn, pfOn ? 1 : 0,
                              k->win, Hf);
    } catch (...) {
        fft_jit_kernel_destroy(k);
        return nullptr;
    }
    k->emitter = b.em;
    k->lut = b.lut;
    k->scratch_f = (plan.kStages > 1) ? b.scratch : nullptr;
    if (plan.kStages == 1) {
        /* single-codelet N=32 owns no scratch; free if allocated */
        if (b.ownsScratch) {
            free(b.scratch);
        }
    }
    try {
        k->emitter->readyRE();
    } catch (...) {
        fft_jit_kernel_destroy(k);
        return nullptr;
    }
    k->fn = reinterpret_cast<fft_jit_fn_t>(
        const_cast<uint8_t *>(k->emitter->getCode()));
    k->code_size = static_cast<int>(k->emitter->getSize());

    if (getenv("FFT_SCHED_LOG") != nullptr) {
        fprintf(stderr, "[schedf] N=%d plan=", N);
        for (int t = 0; t < plan.kStages; ++t) {
            fprintf(stderr, "%s%d", t ? "x" : "", plan.radices[t]);
        }
        fprintf(stderr,
                "/t%d gen=%d pool=%d timed=%d search_ns=%.0f nt=%d pf=%d "
                "rf=%d %s\n",
                plan.tile, plan.genSz, plan.poolSz, plan.timed,
                plan.searchNs, ntOn ? 1 : 0, pfOn ? 1 : 0, plan.refined,
                plan.forced ? " (forced)" : "");
    }
    return k;
}


static int fill_gauss_window(int N, float *out) {
    if (N < 1 || out == nullptr) {
        return -1;
    }
    const double sigma = (double)N / 8.0;
    const double two_s2 = 2.0 * sigma * sigma;
    const double mid = (double)N * 0.5;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        const double d = (double)i - mid;
        const double g = std::exp(-(d * d) / two_s2);
        out[i] = (float)g;
        sum += (double)out[i];
    }
    if (!(sum > 0.0)) {
        return -1;
    }
    const double inv = 1.0 / sum;
    for (int i = 0; i < N; ++i) {
        out[i] = (float)((double)out[i] * inv);
    }
    return 0;
}

template <typename R>
static fft_jit_kernel *fft_jit_kernel_create_ex(int N, int windowed) {
    if (N < 1 || !is_pow2(N) || N > 32768) {
        return nullptr; /* family window is N=2^n n=0..15; anything else is
                           the host layer's FFTW-derived adapt path */
    }
    if (!cpu_has_avx()) {
        return nullptr; /* fail-fast, no silent fallback (DESIGN) */
    }

    /* FMA3 gate for the SchedEmitter pd complex multiplies (fp32 ps path
     * ignores this flag and MUST NOT change).  Without FMA3 the emitter
     * falls back to vmulpd x2 + vhsubpd with the conjugate twiddle
     * derived in registers from the single W table. */
    const bool hasFma = cpu_has_fma() &&
                        getenv("FFT_SCHED_NOFMA") ==
                            nullptr; /* debug: force the non-FMA path
                                        (never raises the feature) */

    const int n = ilog2(N);

    fft_jit_kernel *k = new (std::nothrow) fft_jit_kernel();
    if (k == nullptr) {
        return nullptr;
    }
    k->emitter = nullptr;
    k->lut = nullptr;
    k->lut_d = nullptr;
    k->win = nullptr;
    k->win_dup = nullptr;
    k->scratch_d = nullptr;
    k->scratch_f = nullptr;
    k->fn = nullptr;
    k->fn2 = nullptr;
    k->fn2c = nullptr;
    k->batch_single = 0;
    k->code_size = 0;
    k->windowed = windowed ? 1 : 0;
    k->precision = std::is_same_v<R, double> ? 1 : 0;

    if constexpr (std::is_same_v<R, float>) {
        if (windowed) {
            const size_t winBytes = ((size_t)N * sizeof(float) + 63u) & ~63u;
            const size_t dupBytes =
                ((size_t)(2 * N) * sizeof(float) + 63u) & ~63u;
            k->win = static_cast<float *>(aligned_alloc(64, winBytes));
            k->win_dup = static_cast<float *>(aligned_alloc(64, dupBytes));
            if (k->win == nullptr || k->win_dup == nullptr ||
                fill_gauss_window(N, k->win) != 0) {
                fft_jit_kernel_destroy(k);
                return nullptr;
            }
            for (int i = 0; i < N; ++i) {
                const float g = k->win[bitrev(static_cast<uint32_t>(i), n)];
                k->win_dup[2 * i] = g;
                k->win_dup[2 * i + 1] = g;
            }
        }
    }

    /* --- bit-reversal swap pairs (i < rev(i)) --- */
    std::vector<uint16_t> brVec;
    for (int i = 0; i < N; ++i) {
        const uint32_t r = bitrev(static_cast<uint32_t>(i), n);
        if (static_cast<uint32_t>(i) < r) {
            brVec.push_back(static_cast<uint16_t>(i));
            brVec.push_back(static_cast<uint16_t>(r));
        }
    }
    const int nPairs = static_cast<int>(brVec.size() / 2);

    /* --- twiddle tables W=(wr,wi) and Wn=(wi,-wr) for stages 3..n --- */
    std::vector<std::vector<R>> stageW, stageWn;
    for (int s = 3; s <= n; ++s) {
        const int h = 1 << (s - 1);
        const int m = 1 << s;
        std::vector<R> W(2 * h), Wn(2 * h);
        for (int kk = 0; kk < h; ++kk) {
            const double ang =
                -2.0 * kPi * (double)kk / (double)m;
            const R wr = (R)std::cos(ang);
            const R wi = (R)std::sin(ang);
            W[2 * kk] = wr;
            W[2 * kk + 1] = wi;
            Wn[2 * kk] = wi;
            Wn[2 * kk + 1] = -wr;
        }
        stageW.push_back(std::move(W));
        stageWn.push_back(std::move(Wn));
    }

    try {
        if constexpr (std::is_same_v<R, double>) {
            if (N >= 32) {
                /* ---- scheduling-refactored pd path (topic R0 + R1 + R2) ----
                 * R1: the decomposition is chosen at plan time —
                 * FFT_SCHED_ORDER forces a schedule (debug/verification:
                 * 32first | smallfirst | explicit radix list), otherwise
                 * a light enumerate+time search (DELTA F5) picks the
                 * fastest candidate; FFT_SCHED_SEARCH=off falls back to
                 * the greedy 32-first default.
                 * R2: the search space is (decomposition x tile block
                 * strategy) (DELTA F6); FFT_SCHED_BLOCK pins the block
                 * dimension (off | auto(=2048 elements) | explicit tile
                 * budget), FFT_NT_STORE / FFT_PREFETCH pin the R2 F7
                 * emission flags (unset/auto = measured plan defaults).
                 * All three are topic-local debug knobs, no public API. */
                int radices[8] = {0};
                int kStages = 0;
                bool forced = false;
                const char *ord = getenv("FFT_SCHED_ORDER");
                if (ord != nullptr && *ord != '\0') {
                    if (strcmp(ord, "smallfirst") == 0) {
                        sched_greedy32(n, radices, &kStages);
                        for (int a = 0; a < kStages / 2; ++a) {
                            const int tmp = radices[a];
                            radices[a] = radices[kStages - 1 - a];
                            radices[kStages - 1 - a] = tmp;
                        }
                        forced = true;
                    } else if (strcmp(ord, "32first") == 0) {
                        sched_greedy32(n, radices, &kStages);
                        forced = true;
                    } else {
                        /* explicit radix list, e.g. "16,8,8" (product
                         * must equal N) */
                        int rr[8];
                        int nr = 0;
                        long prod = 1;
                        bool ok = true;
                        const char *s = ord;
                        while (*s != '\0') {
                            char *end = nullptr;
                            const long v = strtol(s, &end, 10);
                            if (end == s || nr >= 8 ||
                                (v != 2 && v != 4 && v != 8 && v != 16 &&
                                 v != 32)) {
                                ok = false;
                                break;
                            }
                            rr[nr++] = (int)v;
                            prod *= v;
                            s = end;
                            if (*s == ',') {
                                ++s;
                            } else if (*s != '\0') {
                                ok = false;
                                break;
                            }
                        }
                        if (ok && prod == N && nr >= 1) {
                            for (int i = 0; i < nr; ++i) {
                                radices[i] = rr[i];
                            }
                            kStages = nr;
                            forced = true;
                        }
                    }
                }
                /* R2 F6 block-strategy pin (invalid values ignored =
                 * auto/search). */
                int forcedTile = -1;
                const char *blk = getenv("FFT_SCHED_BLOCK");
                if (blk != nullptr && *blk != '\0') {
                    if (strcmp(blk, "off") == 0 || strcmp(blk, "0") == 0) {
                        forcedTile = 0;
                    } else if (strcmp(blk, "auto") == 0) {
                        forcedTile = 2048;
                    } else {
                        char *end = nullptr;
                        const long v = strtol(blk, &end, 10);
                        if (end != blk && *end == '\0' && v >= 64 &&
                            v <= 32768) {
                            forcedTile = (int)v;
                        }
                    }
                }
                /* R2 F7 emission flags (unset/auto = measured defaults;
                 * the defaults below are the R2 measured winners — see
                 * evidence/measure-r2.log F7 matrix). */
                bool ntOn = kNtStoreDefault;
                bool pfOn = kPrefetchDefault;
                const char *nts = getenv("FFT_NT_STORE");
                if (nts != nullptr &&
                    (strcmp(nts, "1") == 0 || strcmp(nts, "on") == 0)) {
                    ntOn = true;
                } else if (nts != nullptr &&
                           (strcmp(nts, "0") == 0 ||
                            strcmp(nts, "off") == 0)) {
                    ntOn = false;
                }
                const char *pfs = getenv("FFT_PREFETCH");
                if (pfs != nullptr &&
                    (strcmp(pfs, "1") == 0 || strcmp(pfs, "on") == 0 ||
                     strcmp(pfs, "tile") == 0)) {
                    pfOn = true;
                } else if (pfs != nullptr &&
                           (strcmp(pfs, "0") == 0 ||
                            strcmp(pfs, "off") == 0)) {
                    pfOn = false;
                }
                double searchNs = 0.0;
                int timedCands = 0;
                int poolSz = 0;
                int genSz = 0;
                int planTile = -1;
                if (!forced && n >= 6) {
                    const char *srch = getenv("FFT_SCHED_SEARCH");
                    const bool off =
                        srch != nullptr && (strcmp(srch, "0") == 0 ||
                                            strcmp(srch, "off") == 0);
                    if (!off) {
                        sched_search(N, n, hasFma, ntOn, pfOn, forcedTile,
                                     radices, &kStages, &planTile,
                                     &searchNs, &timedCands, &poolSz,
                                     &genSz);
                    }
                }
                if (kStages == 0) {
                    sched_greedy32(n, radices, &kStages);
                }
                if (planTile < 0) {
                    planTile = forcedTile >= 0 ? forcedTile : 0;
                }
                SchedBuilt b =
                    schedBuildKernel(N, n, radices, kStages, hasFma,
                                     nullptr, planTile, ntOn,
                                     pfOn ? 1 : 0);
                k->emitter = b.em;
                k->lut_d = b.lut;
                k->scratch_d = b.scratch; /* nullptr when kStages == 1 */
                if (getenv("FFT_SCHED_LOG") != nullptr) {
                    fprintf(stderr, "[sched] N=%d plan=", N);
                    for (int t = 0; t < kStages; ++t) {
                        fprintf(stderr, "%s%d", t ? "x" : "",
                                radices[t]);
                    }
                    fprintf(stderr,
                            "/t%d gen=%d pool=%d timed=%d search_ns=%.0f "
                            "nt=%d pf=%d%s\n",
                            planTile, genSz, poolSz, timedCands, searchNs,
                            ntOn ? 1 : 0, pfOn ? 1 : 0,
                            forced ? " (forced)" : "");
                }
                /* stageW/stageWn vectors above are unused on this path
                 * (leaf-internal twiddles are rip-pool constants); the
                 * single TW_t LUT is the only twiddle materialization. */
            } else if (N == 1) {
                class IdentityEmitter final : public Xbyak::CodeGenerator {
                public:
                    IdentityEmitter() : Xbyak::CodeGenerator(64) { ret(); }
                };
                k->emitter = new IdentityEmitter();
            } else {
                k->emitter =
                    new SmallEmitter<R>(n, stageW, stageWn, k->win);
            }
        } else if (N == 1) {
            class IdentityEmitter final : public Xbyak::CodeGenerator {
            public:
                IdentityEmitter() : Xbyak::CodeGenerator(64) { ret(); }
            };
            k->emitter = new IdentityEmitter();
        } else if (N <= 16) {
            k->emitter = new SmallEmitter<R>(n, stageW, stageWn, k->win);
        } else if constexpr (std::is_same_v<R, float>) {
            /* topic jit-ps-sched-fuse (L-A): the ps path above N=16 moves
             * to the fp32 SchedEmitterF (mixed radix + codelet leaves +
             * single TW table + plan-time search).  FFT_PS_SCHED=off keeps
             * the legacy radix-2 three-tier dispatch as the topic-local
             * comparison build; windowed N=32 also stays legacy (the
             * sched window rides the stage-0 gather, N >= 64 only).  The
             * fp64 pd path above is untouched. */
            const char *psk = getenv("FFT_PS_SCHED");
            const bool legacy =
                psk != nullptr &&
                (strcmp(psk, "0") == 0 || strcmp(psk, "off") == 0);
            const bool schedable = !legacy && !(windowed && N < 64);
            if (schedable) {
                fft_jit_kernel_destroy(k);
                return schedF_create_kernel(N, windowed, nullptr);
            } else if (N <= 256) {
                k->emitter = new PoolEmitter<R>(brVec, nPairs, stageW,
                                                stageWn, n, k->win_dup);
            } else {
                const R *stageWptr[16] = {nullptr};
                const R *stageWnptr[16] = {nullptr};

                size_t bitrevBytes = (size_t)nPairs * 4;
                size_t sign2Bytes = 16;
                size_t twBytes = 0;
                for (int s = 3; s <= n; ++s) {
                    twBytes += 32;
                    twBytes += stageW[s - 3].size() * sizeof(R);
                    twBytes += 32;
                    twBytes += stageWn[s - 3].size() * sizeof(R);
                }
                const size_t raw = bitrevBytes + sign2Bytes + 32 + twBytes;
                const size_t total = (raw + 63u) & ~63u;

                R *lut = static_cast<R *>(aligned_alloc(64, total));
                if (lut == nullptr) {
                    fft_jit_kernel_destroy(k);
                    return nullptr;
                }
                k->lut = lut;

                char *base = reinterpret_cast<char *>(lut);
                uint16_t *brTab = reinterpret_cast<uint16_t *>(base);
                for (int i = 0; i < (int)brVec.size(); ++i) {
                    brTab[i] = brVec[i];
                }
                R *sign2Tab = reinterpret_cast<R *>(base + bitrevBytes);
                const float s2[4] = {0.0f, 0.0f, 0.0f, -0.0f};
                for (int i = 0; i < 4; ++i) {
                    sign2Tab[i] = s2[i];
                }
                char *twPtr = base + bitrevBytes + sign2Bytes;
                twPtr += (32 - (reinterpret_cast<uintptr_t>(twPtr) % 32)) % 32;
                for (int s = 3; s <= n; ++s) {
                    stageWptr[s] = reinterpret_cast<const R *>(twPtr);
                    size_t wBytes = stageW[s - 3].size() * sizeof(R);
                    __builtin_memcpy(twPtr, stageW[s - 3].data(), wBytes);
                    twPtr += wBytes;
                    twPtr +=
                        (32 - (reinterpret_cast<uintptr_t>(twPtr) % 32)) % 32;
                    stageWnptr[s] = reinterpret_cast<const R *>(twPtr);
                    size_t wnBytes = stageWn[s - 3].size() * sizeof(R);
                    __builtin_memcpy(twPtr, stageWn[s - 3].data(), wnBytes);
                    twPtr += wnBytes;
                    twPtr +=
                        (32 - (reinterpret_cast<uintptr_t>(twPtr) % 32)) % 32;
                }

                k->emitter = new LutEmitter<R>(brTab, nPairs, n, sign2Tab,
                                               stageWptr, stageWnptr,
                                               k->win_dup);
            }
        } else if (N <= 256) {
            k->emitter = new PoolEmitter<R>(brVec, nPairs, stageW, stageWn,
                                            n, k->win_dup);
        } else {
            const R *stageWptr[16] = {nullptr};
            const R *stageWnptr[16] = {nullptr};

            size_t bitrevBytes = (size_t)nPairs * 4;
            size_t sign2Bytes = 16;
            size_t twBytes = 0;
            for (int s = 3; s <= n; ++s) {
                twBytes += 32;
                twBytes += stageW[s - 3].size() * sizeof(R);
                twBytes += 32;
                twBytes += stageWn[s - 3].size() * sizeof(R);
            }
            const size_t raw = bitrevBytes + sign2Bytes + 32 + twBytes;
            const size_t total = (raw + 63u) & ~63u;

            R *lut = static_cast<R *>(aligned_alloc(64, total));
            if (lut == nullptr) {
                fft_jit_kernel_destroy(k);
                return nullptr;
            }
            if constexpr (std::is_same_v<R, double>) {
                k->lut_d = lut;
            } else {
                k->lut = lut;
            }

            char *base = reinterpret_cast<char *>(lut);
            uint16_t *brTab = reinterpret_cast<uint16_t *>(base);
            for (int i = 0; i < (int)brVec.size(); ++i) {
                brTab[i] = brVec[i];
            }
            R *sign2Tab =
                reinterpret_cast<R *>(base + bitrevBytes);
            if constexpr (std::is_same_v<R, float>) {
                const float s2[4] = {0.0f, 0.0f, 0.0f, -0.0f};
                for (int i = 0; i < 4; ++i) {
                    sign2Tab[i] = s2[i];
                }
            } else {
                const double s2[2] = {0.0, -0.0};
                for (int i = 0; i < 2; ++i) {
                    sign2Tab[i] = s2[i];
                }
            }
            char *twPtr = base + bitrevBytes + sign2Bytes;
            twPtr += (32 - (reinterpret_cast<uintptr_t>(twPtr) % 32)) % 32;
            for (int s = 3; s <= n; ++s) {
                stageWptr[s] =
                    reinterpret_cast<const R *>(twPtr);
                size_t wBytes = stageW[s - 3].size() * sizeof(R);
                __builtin_memcpy(twPtr, stageW[s - 3].data(), wBytes);
                twPtr += wBytes;
                twPtr += (32 - (reinterpret_cast<uintptr_t>(twPtr) % 32)) % 32;
                stageWnptr[s] =
                    reinterpret_cast<const R *>(twPtr);
                size_t wnBytes = stageWn[s - 3].size() * sizeof(R);
                __builtin_memcpy(twPtr, stageWn[s - 3].data(), wnBytes);
                twPtr += wnBytes;
                twPtr += (32 - (reinterpret_cast<uintptr_t>(twPtr) % 32)) % 32;
            }

            k->emitter = new LutEmitter<R>(brTab, nPairs, n, sign2Tab,
                                           stageWptr, stageWnptr, k->win_dup);
        }

        k->emitter->readyRE(); /* R+X, flush icache */
        k->fn = reinterpret_cast<fft_jit_fn_t>(
            const_cast<uint8_t *>(k->emitter->getCode()));
        k->code_size = static_cast<int>(k->emitter->getSize());
    } catch (const std::exception &) {
        fft_jit_kernel_destroy(k);
        return nullptr;
    } catch (...) {
        fft_jit_kernel_destroy(k);
        return nullptr;
    }

    if constexpr (std::is_same_v<R, float>) {
        if (windowed && k->win == nullptr) {
            fft_jit_kernel_destroy(k);
            return nullptr;
        }
    }
    return k;
}

extern "C" fft_jit_kernel *fft_jit_kernel_create(int N) {
    return fft_jit_kernel_create_ex<float>(N, 0);
}

extern "C" fft_jit_kernel *fft_jit_kernel_create_fp64(int N) {
    return fft_jit_kernel_create_ex<double>(N, 0);
}

/* topic jit-fp64-conv-fuse (L-I / draft {#API-FFT-011}): opt-in FUSED
 * fp64 cyclic-convolution weld at the padded transform size Np = 2^n
 * (4 <= Np <= 32768).  h_pad = Np zero-padded complex doubles (natural
 * order); H[k] = F(h_pad) is materialized here by reusing the promoted
 * forward SchedEmitter machinery on a scratch copy (h_pad is NOT
 * clobbered).  Decomposition: n <= 5 forced k>=2 splits (the weld needs
 * >= 2 stages — the [32] single-codelet shape is not weldable); n >= 6
 * reuses the promoted plan-time (decomposition x tile) search with the
 * same FFT_SCHED_* knob semantics as the forward create.  The fp64
 * forward path (fft_jit_kernel_create_ex) above is untouched.  Returns
 * NULL on illegal Np / NULL h_pad / missing AVX / emit / OOM. */
extern "C" fft_jit_kernel *fft_jit_kernel_create_conv_fused_fp64(
    int Np, const double *h_pad) {
    if (Np < 4 || !is_pow2(Np) || Np > 32768 || h_pad == nullptr) {
        return nullptr;
    }
    if (!cpu_has_avx()) {
        return nullptr; /* fail-fast, no silent fallback */
    }
    const bool hasFma = cpu_has_fma() &&
                        getenv("FFT_SCHED_NOFMA") == nullptr;
    const int n = ilog2(Np);

    /* --- decomposition choice (searched for n >= 6, forced below) --- */
    int radices[8] = {0};
    int kStages = 0;
    int planTile = -1;
    bool ntOn = kNtStoreDefault;
    bool pfOn = kPrefetchDefault;
    const char *nts = getenv("FFT_NT_STORE");
    if (nts != nullptr &&
        (strcmp(nts, "1") == 0 || strcmp(nts, "on") == 0)) {
        ntOn = true;
    } else if (nts != nullptr &&
               (strcmp(nts, "0") == 0 || strcmp(nts, "off") == 0)) {
        ntOn = false;
    }
    const char *pfs = getenv("FFT_PREFETCH");
    if (pfs != nullptr &&
        (strcmp(pfs, "1") == 0 || strcmp(pfs, "on") == 0 ||
         strcmp(pfs, "tile") == 0)) {
        pfOn = true;
    } else if (pfs != nullptr &&
               (strcmp(pfs, "0") == 0 || strcmp(pfs, "off") == 0)) {
        pfOn = false;
    }
    if (n <= 5) {
        /* forced k >= 2 splits: n=2 [2,2], n=3 [4,2], n=4 [4,4],
         * n=5 [4,4,2] (excludes the single-codelet [32]) */
        static const int smallRad[4][3] = {
            {2, 2, 0}, {4, 2, 0}, {4, 4, 0}, {4, 4, 2}};
        const int *r = smallRad[n - 2];
        kStages = (n <= 4) ? 2 : 3;
        for (int i = 0; i < kStages; ++i) {
            radices[i] = r[i];
        }
        planTile = 0;
    } else {
        int forcedTile = -1;
        const char *blk = getenv("FFT_SCHED_BLOCK");
        if (blk != nullptr && *blk != '\0') {
            if (strcmp(blk, "off") == 0 || strcmp(blk, "0") == 0) {
                forcedTile = 0;
            } else if (strcmp(blk, "auto") == 0) {
                forcedTile = 2048;
            } else {
                char *end = nullptr;
                const long v = strtol(blk, &end, 10);
                if (end != blk && *end == '\0' && v >= 64 && v <= 32768) {
                    forcedTile = (int)v;
                }
            }
        }
        const char *ord = getenv("FFT_SCHED_ORDER");
        bool forced = false;
        if (ord != nullptr && *ord != '\0') {
            if (strcmp(ord, "smallfirst") == 0) {
                sched_greedy32(n, radices, &kStages);
                for (int a = 0; a < kStages / 2; ++a) {
                    const int tmp = radices[a];
                    radices[a] = radices[kStages - 1 - a];
                    radices[kStages - 1 - a] = tmp;
                }
                forced = true;
            } else if (strcmp(ord, "32first") == 0) {
                sched_greedy32(n, radices, &kStages);
                forced = true;
            } else {
                /* explicit radix list, e.g. "16,8,8" (product must equal
                 * Np) — same parse as the forward path */
                int rr[8];
                int nr = 0;
                long prod = 1;
                bool ok = true;
                const char *s = ord;
                while (*s != '\0') {
                    char *end = nullptr;
                    const long v = strtol(s, &end, 10);
                    if (end == s || nr >= 8 ||
                        (v != 2 && v != 4 && v != 8 && v != 16 && v != 32)) {
                        ok = false;
                        break;
                    }
                    rr[nr++] = (int)v;
                    prod *= v;
                    s = end;
                    if (*s == ',') {
                        ++s;
                    } else if (*s != '\0') {
                        ok = false;
                        break;
                    }
                }
                if (ok && prod == Np && nr >= 1) {
                    for (int i = 0; i < nr; ++i) {
                        radices[i] = rr[i];
                    }
                    kStages = nr;
                    forced = true;
                }
            }
        }
        if (!forced) {
            const char *srch = getenv("FFT_SCHED_SEARCH");
            const bool off =
                srch != nullptr && (strcmp(srch, "0") == 0 ||
                                    strcmp(srch, "off") == 0);
            if (!off) {
                double searchNs = 0.0;
                int timedCands = 0, poolSz = 0, genSz = 0;
                sched_search(Np, n, hasFma, ntOn, pfOn, forcedTile,
                             radices, &kStages, &planTile, &searchNs,
                             &timedCands, &poolSz, &genSz);
            }
        }
        if (kStages == 0) {
            sched_greedy32(n, radices, &kStages);
        }
        if (kStages < 2) {
            /* defensive: split a lone [32] (only reachable via forced
             * order lists) into [16, 2] so the weld has 2 stages */
            radices[1] = radices[0] == 32 ? 2 : radices[0];
            radices[0] = radices[0] == 32 ? 16 : 2;
            if (radices[0] * radices[1] != Np) {
                return nullptr; /* unreachable: n >= 6 has k >= 2 */
            }
            kStages = 2;
        }
        if (planTile < 0) {
            planTile = forcedTile >= 0 ? forcedTile : 0;
        }
    }

    /* fix conv-weld-r32-final: the weld's radix-32 FINAL stage is not a
     * valid shape (the conv weld's final-stage leaf-32 path is only
     * correct with 32 in a NON-final position — a latent forced-order
     * limitation of the base weld, never hit by the plan-time search;
     * found during jit-fp64-linear-conv R0, evidence in
     * poc/jit-fp64-linear-conv/ndf/evidence).  The search never picks a
     * 32-final, but an FFT_SCHED_ORDER env pin (or the greedy32 fallback
     * at n % 5 == 0 under FFT_SCHED_SEARCH=off / 32first / smallfirst)
     * could: split any trailing 32 into [16, 2] so the conv kernel is
     * ALWAYS a valid shape — the same defense the fp64 inverse create
     * already carries.  Pure no-op for every non-32-final decomposition
     * (default search selections unchanged).  A forced 8-stage 32-final
     * list cannot fit the [16, 2] split in the radices[8] array — fail
     * closed (NULL) rather than emit the invalid shape. */
    if (kStages >= 1 && radices[kStages - 1] == 32) {
        if (kStages >= 8) {
            return nullptr; /* unreachable via search; forced pins only */
        }
        radices[kStages - 1] = 16;
        radices[kStages] = 2;
        kStages += 1;
    }

    fft_jit_kernel *k = new (std::nothrow) fft_jit_kernel();
    if (k == nullptr) {
        return nullptr;
    }
    k->emitter = nullptr;
    k->lut = nullptr;
    k->lut_d = nullptr;
    k->win = nullptr;
    k->win_dup = nullptr;
    k->scratch_d = nullptr;
    k->h_mul_d = nullptr;
    k->h_swap_d = nullptr;
    k->fn = nullptr;
    k->code_size = 0;
    k->windowed = 0;
    k->precision = 1;

    /* --- plan-time H materialization: plain forward build with the SAME
     * decomposition (no second search), run on a scratch copy of h_pad —
     * the caller's h_pad is NOT clobbered; no exp at plan time.
     * Ownership discipline: locals are nulled out the moment their
     * content transfers to k (or is freed), so the catch blocks below
     * never double-free what fft_jit_kernel_destroy(k) already owns. --- */
    SchedBuilt fwd = {nullptr, nullptr, nullptr, false, false};
    double *hwork = nullptr;
    double *hMul = nullptr;
    double *hSwap = nullptr;
    SchedConvBuilt b = {nullptr, nullptr, nullptr};
    try {
        fwd = schedBuildKernel(Np, n, radices, kStages, hasFma, nullptr,
                               planTile, ntOn, pfOn ? 1 : 0);
        fwd.em->readyRE();
        fft_jit_fn_t ffn = reinterpret_cast<fft_jit_fn_t>(
            const_cast<uint8_t *>(fwd.em->getCode()));
        hwork = static_cast<double *>(aligned_alloc(64, (size_t)Np * 16));
        if (hwork == nullptr) {
            throw std::bad_alloc();
        }
        __builtin_memcpy(hwork, h_pad, (size_t)Np * 16);
        ffn(hwork); /* hwork = H (natural order) */

        /* H tables: separate aligned allocations (NOT part of the twiddle
         * LUT block — ARCH-FFT-016 single-twiddle discipline).  hSwap =
         * pre-swapped twin (hi, +hr|FMA / hi, -hr|no-FMA), same sign
         * discipline as the TW twins. */
        hMul = static_cast<double *>(aligned_alloc(64, (size_t)Np * 16));
        hSwap = static_cast<double *>(aligned_alloc(64, (size_t)Np * 16));
        if (hMul == nullptr || hSwap == nullptr) {
            throw std::bad_alloc();
        }
        for (int i = 0; i < Np; ++i) {
            const double hr = hwork[2 * i];
            const double hi = hwork[2 * i + 1];
            hMul[2 * i] = hr;
            hMul[2 * i + 1] = hi;
            hSwap[2 * i] = hi;
            hSwap[2 * i + 1] = hasFma ? hr : -hr;
        }
        free(hwork);
        hwork = nullptr;
        delete fwd.em;
        free(fwd.lut);
        free(fwd.scratch);
        fwd.em = nullptr;
        fwd.lut = nullptr;
        fwd.scratch = nullptr;

        b = schedConvBuildKernel(Np, n, radices, kStages, hasFma, hMul,
                                 hSwap, planTile, ntOn, pfOn ? 1 : 0);
        k->emitter = b.em;
        k->lut_d = b.lut;
        k->scratch_d = b.scratch;
        k->h_mul_d = hMul;
        k->h_swap_d = hSwap;
        b.em = nullptr;
        b.lut = nullptr;
        b.scratch = nullptr;
        hMul = nullptr;
        hSwap = nullptr;
        k->emitter->readyRE(); /* R+X, flush icache */
        k->fn = reinterpret_cast<fft_jit_fn_t>(
            const_cast<uint8_t *>(k->emitter->getCode()));
        k->code_size = static_cast<int>(k->emitter->getSize());
    } catch (...) {
        fft_jit_kernel_destroy(k);
        delete b.em;
        free(b.lut);
        free(b.scratch);
        delete fwd.em;
        free(fwd.lut);
        free(fwd.scratch);
        free(hMul);
        free(hSwap);
        free(hwork);
        return nullptr;
    }

    if (getenv("FFT_SCHED_LOG") != nullptr) {
        fprintf(stderr, "[conv] Np=%d plan=", Np);
        for (int t = 0; t < kStages; ++t) {
            fprintf(stderr, "%s%d", t ? "x" : "", radices[t]);
        }
        fprintf(stderr, "/t%d code=%d fma=%d nt=%d pf=%d\n", planTile,
                k->code_size, hasFma ? 1 : 0, ntOn ? 1 : 0, pfOn ? 1 : 0);
    }
    return k;
}

/* topic jit-fp64-linear-conv (DELTA F1/F2; draft {#API-FFT-012} /
 * {#BEH-FFT-017}): opt-in fp64 FUSED overlap-save LINEAR-convolution
 * weld at the pow2 block size N = 2^n (4 <= N <= 32768), kernel length
 * 0 < nh <= N.  h_pad = N zero-padded complex doubles (natural order);
 * H[k] = F(h_pad) materialized here on a scratch copy exactly as in the
 * conv weld (promoted forward SchedEmitter machinery, h_pad NOT
 * clobbered, hMul/hSwap separate from the twiddle LUT — no fourth
 * twiddle subsystem).  The weld body = SchedEmitterO (rotated half-A
 * gather + valid-prefix half-B final stores; see the class comment).
 * Decomposition = the conv create framework verbatim (n <= 5 forced
 * k >= 2 splits; n >= 6 the promoted plan-time (decomposition x tile)
 * search; FFT_SCHED_* knob semantics inherited — the conv path above is
 * untouched).  Topic-local knob FFT_OLS_DISCARD = skip|write pins the
 * discard-zone handling (default skip).  Returns NULL on illegal
 * (N, nh) / NULL h_pad / missing AVX / emit / OOM. */
extern "C" fft_jit_kernel *fft_jit_kernel_create_ols_fused_fp64(
    int N, const double *h_pad, int nh) {
    if (N < 4 || !is_pow2(N) || N > 32768 || h_pad == nullptr) {
        return nullptr;
    }
    if (nh <= 0 || nh > N) {
        return nullptr;
    }
    if (!cpu_has_avx()) {
        return nullptr; /* fail-fast, no silent fallback */
    }
    const bool hasFma = cpu_has_fma() &&
                        getenv("FFT_SCHED_NOFMA") == nullptr;
    const int n = ilog2(N);

    /* topic-local adjudication knob: discard-zone handling (skip = the
     * half-B final stores predicate the valid prefix; write = full-N
     * stores, discard written-then-abandoned — semantics unpromised
     * either way, {#API-FFT-012}). */
    bool writeDiscard = false;
    const char *disc = getenv("FFT_OLS_DISCARD");
    if (disc != nullptr && strcmp(disc, "write") == 0) {
        writeDiscard = true;
    }

    /* --- decomposition choice (conv framework, verbatim) --- */
    int radices[8] = {0};
    int kStages = 0;
    int planTile = -1;
    bool ntOn = kNtStoreDefault;
    bool pfOn = kPrefetchDefault;
    const char *nts = getenv("FFT_NT_STORE");
    if (nts != nullptr &&
        (strcmp(nts, "1") == 0 || strcmp(nts, "on") == 0)) {
        ntOn = true;
    } else if (nts != nullptr &&
               (strcmp(nts, "0") == 0 || strcmp(nts, "off") == 0)) {
        ntOn = false;
    }
    const char *pfs = getenv("FFT_PREFETCH");
    if (pfs != nullptr &&
        (strcmp(pfs, "1") == 0 || strcmp(pfs, "on") == 0 ||
         strcmp(pfs, "tile") == 0)) {
        pfOn = true;
    } else if (pfs != nullptr &&
               (strcmp(pfs, "0") == 0 || strcmp(pfs, "off") == 0)) {
        pfOn = false;
    }
    if (n <= 5) {
        /* forced k >= 2 splits: n=2 [2,2], n=3 [4,2], n=4 [4,4],
         * n=5 [4,4,2] (excludes the single-codelet [32]) */
        static const int smallRad[4][3] = {
            {2, 2, 0}, {4, 2, 0}, {4, 4, 0}, {4, 4, 2}};
        const int *r = smallRad[n - 2];
        kStages = (n <= 4) ? 2 : 3;
        for (int i = 0; i < kStages; ++i) {
            radices[i] = r[i];
        }
        planTile = 0;
    } else {
        int forcedTile = -1;
        const char *blk = getenv("FFT_SCHED_BLOCK");
        if (blk != nullptr && *blk != '\0') {
            if (strcmp(blk, "off") == 0 || strcmp(blk, "0") == 0) {
                forcedTile = 0;
            } else if (strcmp(blk, "auto") == 0) {
                forcedTile = 2048;
            } else {
                char *end = nullptr;
                const long v = strtol(blk, &end, 10);
                if (end != blk && *end == '\0' && v >= 64 && v <= 32768) {
                    forcedTile = (int)v;
                }
            }
        }
        const char *ord = getenv("FFT_SCHED_ORDER");
        bool forced = false;
        if (ord != nullptr && *ord != '\0') {
            if (strcmp(ord, "smallfirst") == 0) {
                sched_greedy32(n, radices, &kStages);
                for (int a = 0; a < kStages / 2; ++a) {
                    const int tmp = radices[a];
                    radices[a] = radices[kStages - 1 - a];
                    radices[kStages - 1 - a] = tmp;
                }
                forced = true;
            } else if (strcmp(ord, "32first") == 0) {
                sched_greedy32(n, radices, &kStages);
                forced = true;
            } else {
                int rr[8];
                int nr = 0;
                long prod = 1;
                bool ok = true;
                const char *s = ord;
                while (*s != '\0') {
                    char *end = nullptr;
                    const long v = strtol(s, &end, 10);
                    if (end == s || nr >= 8 ||
                        (v != 2 && v != 4 && v != 8 && v != 16 && v != 32)) {
                        ok = false;
                        break;
                    }
                    rr[nr++] = (int)v;
                    prod *= v;
                    s = end;
                    if (*s == ',') {
                        ++s;
                    } else if (*s != '\0') {
                        ok = false;
                        break;
                    }
                }
                if (ok && prod == N && nr >= 1) {
                    for (int i = 0; i < nr; ++i) {
                        radices[i] = rr[i];
                    }
                    kStages = nr;
                    forced = true;
                }
            }
        }
        if (!forced) {
            const char *srch = getenv("FFT_SCHED_SEARCH");
            const bool off =
                srch != nullptr && (strcmp(srch, "0") == 0 ||
                                    strcmp(srch, "off") == 0);
            if (!off) {
                double searchNs = 0.0;
                int timedCands = 0, poolSz = 0, genSz = 0;
                sched_search(N, n, hasFma, ntOn, pfOn, forcedTile,
                             radices, &kStages, &planTile, &searchNs,
                             &timedCands, &poolSz, &genSz);
            }
        }
        if (kStages == 0) {
            sched_greedy32(n, radices, &kStages);
        }
        if (kStages < 2) {
            /* defensive: split a lone [32] into [16, 2] so the weld has
             * 2 stages */
            radices[1] = radices[0] == 32 ? 2 : radices[0];
            radices[0] = radices[0] == 32 ? 16 : 2;
            if (radices[0] * radices[1] != N) {
                return nullptr; /* unreachable: n >= 6 has k >= 2 */
            }
            kStages = 2;
        }
        if (planTile < 0) {
            planTile = forcedTile >= 0 ? forcedTile : 0;
        }
    }

    /* fix conv-weld-r32-final (the R0 evidence note claimed this guard
     * lived here; it actually landed in the fp64 inverse create — the
     * OLS weld copied the conv final-stage leaf-32 path and inherits
     * the same 32-final invalid shape): split any trailing 32 into
     * [16, 2] so the OLS kernel is ALWAYS a valid shape — identical
     * defense to the conv weld create and the fp64 inverse create.  No
     * search-selected decomposition ever ends in 32 (no-op there); an
     * 8-stage forced 32-final fails closed (NULL). */
    if (kStages >= 1 && radices[kStages - 1] == 32) {
        if (kStages >= 8) {
            return nullptr; /* unreachable via search; forced pins only */
        }
        radices[kStages - 1] = 16;
        radices[kStages] = 2;
        kStages += 1;
    }

    fft_jit_kernel *k = new (std::nothrow) fft_jit_kernel();
    if (k == nullptr) {
        return nullptr;
    }
    k->emitter = nullptr;
    k->lut = nullptr;
    k->lut_d = nullptr;
    k->win = nullptr;
    k->win_dup = nullptr;
    k->scratch_d = nullptr;
    k->h_mul_d = nullptr;
    k->h_swap_d = nullptr;
    k->spec_d = nullptr;
    k->fn = nullptr;
    k->fn_ols = nullptr;
    k->code_size = 0;
    k->windowed = 0;
    k->precision = 1;

    /* --- plan-time H materialization: plain forward build with the SAME
     * decomposition (no second search), run on a scratch copy of h_pad —
     * identical discipline to the conv weld (h_pad NOT clobbered, no exp
     * at plan time; ownership transfer discipline likewise). --- */
    SchedBuilt fwd = {nullptr, nullptr, nullptr, false, false};
    double *hwork = nullptr;
    double *hMul = nullptr;
    double *hSwap = nullptr;
    SchedOlsBuilt b = {nullptr, nullptr, nullptr, nullptr};
    try {
        fwd = schedBuildKernel(N, n, radices, kStages, hasFma, nullptr,
                               planTile, ntOn, pfOn ? 1 : 0);
        fwd.em->readyRE();
        fft_jit_fn_t ffn = reinterpret_cast<fft_jit_fn_t>(
            const_cast<uint8_t *>(fwd.em->getCode()));
        hwork = static_cast<double *>(aligned_alloc(64, (size_t)N * 16));
        if (hwork == nullptr) {
            throw std::bad_alloc();
        }
        __builtin_memcpy(hwork, h_pad, (size_t)N * 16);
        ffn(hwork); /* hwork = H (natural order) */

        hMul = static_cast<double *>(aligned_alloc(64, (size_t)N * 16));
        hSwap = static_cast<double *>(aligned_alloc(64, (size_t)N * 16));
        if (hMul == nullptr || hSwap == nullptr) {
            throw std::bad_alloc();
        }
        for (int i = 0; i < N; ++i) {
            const double hr = hwork[2 * i];
            const double hi = hwork[2 * i + 1];
            hMul[2 * i] = hr;
            hMul[2 * i + 1] = hi;
            hSwap[2 * i] = hi;
            hSwap[2 * i + 1] = hasFma ? hr : -hr;
        }
        free(hwork);
        hwork = nullptr;
        delete fwd.em;
        free(fwd.lut);
        free(fwd.scratch);
        fwd.em = nullptr;
        fwd.lut = nullptr;
        fwd.scratch = nullptr;

        b = schedOlsBuildKernel(N, n, radices, kStages, hasFma, hMul,
                                hSwap, nh, writeDiscard, planTile, ntOn,
                                pfOn ? 1 : 0);
        k->emitter = b.em;
        k->lut_d = b.lut;
        k->scratch_d = b.scratch;
        k->spec_d = b.spec;
        k->h_mul_d = hMul;
        k->h_swap_d = hSwap;
        b.em = nullptr;
        b.lut = nullptr;
        b.scratch = nullptr;
        b.spec = nullptr;
        hMul = nullptr;
        hSwap = nullptr;
        k->emitter->readyRE(); /* R+X, flush icache */
        k->fn = static_cast<SchedEmitterO *>(
                    k->emitter)->fnEntry();
        k->fn_ols = static_cast<SchedEmitterO *>(
                        k->emitter)->streamEntry();
        k->code_size = static_cast<int>(k->emitter->getSize());
    } catch (...) {
        fft_jit_kernel_destroy(k);
        delete b.em;
        free(b.lut);
        free(b.scratch);
        free(b.spec);
        delete fwd.em;
        free(fwd.lut);
        free(fwd.scratch);
        free(hMul);
        free(hSwap);
        free(hwork);
        return nullptr;
    }

    if (getenv("FFT_SCHED_LOG") != nullptr) {
        fprintf(stderr, "[ols64] N=%d nh=%d plan=", N, nh);
        for (int t = 0; t < kStages; ++t) {
            fprintf(stderr, "%s%d", t ? "x" : "", radices[t]);
        }
        fprintf(stderr, "/t%d code=%d fma=%d nt=%d pf=%d discard=%s\n",
                planTile, k->code_size, hasFma ? 1 : 0, ntOn ? 1 : 0,
                pfOn ? 1 : 0, writeDiscard ? "write" : "skip");
    }
    return k;
}

extern "C" fft_jit_ols_fn_t fft_jit_kernel_ols_stream_fn(
    const fft_jit_kernel *k) {
    return (k != nullptr) ? k->fn_ols : nullptr;
}

/* topic jit-fp64-inverse (amend resume 2026-09-17, DELTA F3; draft
 * {#API-FFT-009} / {#BEH-FFT-010}): opt-in FUSED fp64 INVERSE kernel at
 * N = 2^n (32 <= N <= 32768) — IFFT(x) = conj(F(conj(x))) / N as ONE
 * welded kernel: the conjugate is sunk into the stage-0 gather LOAD
 * boundary (per gathered pair vector one vxorpd against the rip sign
 * mask), the middle stages run the promoted forward sweeps / twiddles
 * unchanged (conjugate trick — no inverse twiddle table, no second
 * emitter, no new ps/pd branch), and the final stage stores
 * conj(.) * (1/N) via one vmulpd against the (invN, -invN) rip constant
 * (the conv weld half-B finalMode-2 idiom).  Decomposition: SAME
 * framework as the same-N fp64 forward — n <= 5 forced k>=2 splits
 * ({4,4,2} at N=32; the [32] single-codelet shape has no separate final
 * store stage to sink the boundary ops into), n >= 6 the promoted
 * plan-time (decomposition x tile) search with the same FFT_SCHED_*
 * knob semantics as the forward / conv creates.  The fp64 forward path
 * (fft_jit_kernel_create_ex<double>) and the conv weld above are
 * untouched.  Returns NULL on illegal N / missing AVX / emit / OOM. */
extern "C" fft_jit_kernel *fft_jit_kernel_create_inverse_fused_fp64(int N) {
    if (N < 32 || !is_pow2(N) || N > 32768) {
        return nullptr;
    }
    if (!cpu_has_avx()) {
        return nullptr; /* fail-fast, no silent fallback */
    }
    const bool hasFma = cpu_has_fma() &&
                        getenv("FFT_SCHED_NOFMA") == nullptr;
    const int n = ilog2(N);

    /* --- decomposition choice (same framework as the same-N forward) --- */
    int radices[8] = {0};
    int kStages = 0;
    int planTile = -1;
    bool ntOn = kNtStoreDefault;
    bool pfOn = kPrefetchDefault;
    const char *nts = getenv("FFT_NT_STORE");
    if (nts != nullptr &&
        (strcmp(nts, "1") == 0 || strcmp(nts, "on") == 0)) {
        ntOn = true;
    } else if (nts != nullptr &&
               (strcmp(nts, "0") == 0 || strcmp(nts, "off") == 0)) {
        ntOn = false;
    }
    const char *pfs = getenv("FFT_PREFETCH");
    if (pfs != nullptr &&
        (strcmp(pfs, "1") == 0 || strcmp(pfs, "on") == 0 ||
         strcmp(pfs, "tile") == 0)) {
        pfOn = true;
    } else if (pfs != nullptr &&
               (strcmp(pfs, "0") == 0 || strcmp(pfs, "off") == 0)) {
        pfOn = false;
    }
    if (n <= 5) {
        /* forced k >= 2 splits: only n = 5 (N = 32) is reachable here
         * ({4,4,2}; excludes the single-codelet [32]) */
        static const int smallRad[4][3] = {
            {2, 2, 0}, {4, 2, 0}, {4, 4, 0}, {4, 4, 2}};
        const int *r = smallRad[n - 2];
        kStages = (n <= 4) ? 2 : 3;
        for (int i = 0; i < kStages; ++i) {
            radices[i] = r[i];
        }
        planTile = 0;
    } else {
        int forcedTile = -1;
        const char *blk = getenv("FFT_SCHED_BLOCK");
        if (blk != nullptr && *blk != '\0') {
            if (strcmp(blk, "off") == 0 || strcmp(blk, "0") == 0) {
                forcedTile = 0;
            } else if (strcmp(blk, "auto") == 0) {
                forcedTile = 2048;
            } else {
                char *end = nullptr;
                const long v = strtol(blk, &end, 10);
                if (end != blk && *end == '\0' && v >= 64 && v <= 32768) {
                    forcedTile = (int)v;
                }
            }
        }
        const char *ord = getenv("FFT_SCHED_ORDER");
        bool forced = false;
        if (ord != nullptr && *ord != '\0') {
            if (strcmp(ord, "smallfirst") == 0) {
                sched_greedy32(n, radices, &kStages);
                for (int a = 0; a < kStages / 2; ++a) {
                    const int tmp = radices[a];
                    radices[a] = radices[kStages - 1 - a];
                    radices[kStages - 1 - a] = tmp;
                }
                forced = true;
            } else if (strcmp(ord, "32first") == 0) {
                sched_greedy32(n, radices, &kStages);
                forced = true;
            } else {
                /* explicit radix list, e.g. "16,8,8" (product must equal
                 * N) — same parse as the forward path */
                int rr[8];
                int nr = 0;
                long prod = 1;
                bool ok = true;
                const char *s = ord;
                while (*s != '\0') {
                    char *end = nullptr;
                    const long v = strtol(s, &end, 10);
                    if (end == s || nr >= 8 ||
                        (v != 2 && v != 4 && v != 8 && v != 16 && v != 32)) {
                        ok = false;
                        break;
                    }
                    rr[nr++] = (int)v;
                    prod *= v;
                    s = end;
                    if (*s == ',') {
                        ++s;
                    } else if (*s != '\0') {
                        ok = false;
                        break;
                    }
                }
                if (ok && prod == N && nr >= 1) {
                    for (int i = 0; i < nr; ++i) {
                        radices[i] = rr[i];
                    }
                    kStages = nr;
                    forced = true;
                }
            }
        }
        if (!forced) {
            const char *srch = getenv("FFT_SCHED_SEARCH");
            const bool off =
                srch != nullptr && (strcmp(srch, "0") == 0 ||
                                    strcmp(srch, "off") == 0);
            if (!off) {
                double searchNs = 0.0;
                int timedCands = 0, poolSz = 0, genSz = 0;
                sched_search(N, n, hasFma, ntOn, pfOn, forcedTile,
                             radices, &kStages, &planTile, &searchNs,
                             &timedCands, &poolSz, &genSz);
            }
        }
        if (kStages == 0) {
            sched_greedy32(n, radices, &kStages);
        }
        if (kStages < 2) {
            /* defensive: split a lone [32] (only reachable via forced
             * order lists) into [16, 2] so the fused final exists */
            radices[1] = radices[0] == 32 ? 2 : radices[0];
            radices[0] = radices[0] == 32 ? 16 : 2;
            if (radices[0] * radices[1] != N) {
                return nullptr; /* unreachable: n >= 6 has k >= 2 */
            }
            kStages = 2;
        }
        if (planTile < 0) {
            planTile = forcedTile >= 0 ? forcedTile : 0;
        }
    }

    /* topic OLS defensive guard: the weld's single-k radix-32 FINAL
     * stage is not a valid shape (the promoted conv weld's final-stage
     * leaf-32 path is only correct with 32 in a NON-final position — a
     * latent forced-order limitation of the base weld, never hit by the
     * plan-time search; see ndf/evidence).  The search never picks a
     * 32-final, but an FFT_SCHED_ORDER env pin could: split any trailing
     * 32 into [16, 2] so the OLS kernel is ALWAYS a valid shape. */
    if (kStages >= 1 && radices[kStages - 1] == 32) {
        if (kStages >= 8) {
            return nullptr; /* unreachable: n <= 15 caps k at 8 */
        }
        radices[kStages - 1] = 16;
        radices[kStages] = 2;
        kStages += 1;
    }

    fft_jit_kernel *k = new (std::nothrow) fft_jit_kernel();
    if (k == nullptr) {
        return nullptr;
    }
    k->emitter = nullptr;
    k->lut = nullptr;
    k->lut_d = nullptr;
    k->win = nullptr;
    k->win_dup = nullptr;
    k->scratch_d = nullptr;
    k->h_mul_d = nullptr;
    k->h_swap_d = nullptr;
    k->fn = nullptr;
    k->code_size = 0;
    k->windowed = 0;
    k->precision = 1;

    SchedInvBuilt b = {nullptr, nullptr, nullptr};
    try {
        b = schedInvBuildKernel(N, n, radices, kStages, hasFma, planTile,
                                ntOn, pfOn ? 1 : 0);
        k->emitter = b.em;
        k->lut_d = b.lut;
        k->scratch_d = b.scratch;
        b.em = nullptr;
        b.lut = nullptr;
        b.scratch = nullptr;
        k->emitter->readyRE(); /* R+X, flush icache */
        k->fn = reinterpret_cast<fft_jit_fn_t>(
            const_cast<uint8_t *>(k->emitter->getCode()));
        k->code_size = static_cast<int>(k->emitter->getSize());
    } catch (...) {
        fft_jit_kernel_destroy(k);
        delete b.em;
        free(b.lut);
        free(b.scratch);
        return nullptr;
    }

    if (getenv("FFT_SCHED_LOG") != nullptr) {
        fprintf(stderr, "[inv64] N=%d plan=", N);
        for (int t = 0; t < kStages; ++t) {
            fprintf(stderr, "%s%d", t ? "x" : "", radices[t]);
        }
        fprintf(stderr, "/t%d code=%d fma=%d nt=%d pf=%d\n", planTile,
                k->code_size, hasFma ? 1 : 0, ntOn ? 1 : 0, pfOn ? 1 : 0);
    }
    return k;
}

/*
 * topic jit-codelet-weld (DELTA F3; draft {#API-FFT-011} coverage
 * upgrade): opt-in FUSED fp64 cyclic convolution at ORIGINAL SIZE N
 * (arbitrary factorable N, no zero-padding inflation).  N = r1*..*rk over
 * the weld domain {2,3,4,5,6,7,8,9,10,12,13,15,16,20,25,32} with k >= 2;
 * pow2 N stays on the promoted SchedEmitterC path (fft_jit_kernel_
 * create_conv_fused_fp64 — regression-guarded, untouched).  Leaves:
 * pow2 factors -> own vector leaves; 3/5-family factors -> FFTW codelet
 * leaves (L-K subset library, numeric-probe-locked convention); the
 * decomposition is plan-time searched (FFT_WELD_ORDER forces one for
 * mixed-dispatch verification, FFT_WELD_SEARCH=off takes the greedy
 * anchor).  Fused boundary: own final stores conj(X*H) / conj(.)/N;
 * codelet final moves the pointwise op to half B's stage-0 gather loads
 * (conj(X)*conjH slots) + an INVN sweep — the intermediate spectrum
 * transits io exactly once either way.
 *
 * Returns NULL on illegal N / NULL h / missing AVX2+FMA (the vendored
 * simd-avx2.h requires FMA3 in VZMUL/VZMULJ — without it the family
 * contract keeps the pad route) / unfactorable N / emit / OOM.  h is NOT
 * clobbered (H materialized on a scratch copy by the forward weld).
 */
extern "C" fft_jit_kernel *fft_jit_kernel_create_conv_weld_fp64(
    int N, const double *h) {
    if (N < 6 || N > 32768 || is_pow2(N) || h == nullptr) {
        return nullptr;
    }
    if (!cpu_has_avx() || !cpu_has_fma()) {
        return nullptr; /* fail-fast, no silent fallback */
    }
    weld_codelets_init();
    if (!weld_factorable(N)) {
        return nullptr;
    }
    const bool hasFma = cpu_has_fma() &&
                        getenv("FFT_SCHED_NOFMA") == nullptr;

    int radices[8] = {0};
    int kStages = 0;
    bool forced = false;
    const char *ord = getenv("FFT_WELD_ORDER");
    if (ord != nullptr && *ord != '\0') {
        int rr[8];
        int nr = 0;
        long prod = 1;
        bool ok = true;
        const char *p = ord;
        while (*p != '\0') {
            char *end = nullptr;
            const long v = strtol(p, &end, 10);
            bool legal = false;
            for (int fi = 0; fi < weld_n_factors; ++fi) {
                if (v == weld_factors[fi]) {
                    legal = true;
                }
            }
            if (end == p || nr >= 8 || !legal) {
                ok = false;
                break;
            }
            rr[nr++] = (int)v;
            prod *= v;
            p = end;
            if (*p == ',') {
                ++p;
            } else if (*p != '\0') {
                ok = false;
                break;
            }
        }
        if (ok && prod == N && nr >= 2 && weld_viable(N, rr, nr)) {
            for (int i = 0; i < nr; ++i) {
                radices[i] = rr[i];
            }
            kStages = nr;
            forced = true;
        }
    }
    double searchNs = 0.0;
    int timedCands = 0, poolSz = 0, genSz = 0;
    if (!forced) {
        const char *srch = getenv("FFT_WELD_SEARCH");
        const bool off = srch != nullptr && (strcmp(srch, "0") == 0 ||
                                             strcmp(srch, "off") == 0);
        if (!off) {
            weld_search(N, hasFma, radices, &kStages, &searchNs,
                        &timedCands, &poolSz, &genSz);
        }
    }
    if (kStages == 0) {
        int rad[8];
        int k = 0;
        weld_anchor(N, true, rad, &k);
        if (k == 0) {
            return nullptr;
        }
        for (int i = 0; i < k; ++i) {
            radices[i] = rad[i];
        }
        kStages = k;
    }

    fft_jit_kernel *k = new (std::nothrow) fft_jit_kernel();
    if (k == nullptr) {
        return nullptr;
    }
    k->emitter = nullptr;
    k->lut = nullptr;
    k->lut_d = nullptr;
    k->win = nullptr;
    k->win_dup = nullptr;
    k->scratch_d = nullptr;
    k->h_mul_d = nullptr;
    k->h_swap_d = nullptr;
    k->weld_arena = nullptr;
    k->weld_arena_bytes = 0;
    k->weld_hc = nullptr;
    k->fn = nullptr;
    k->code_size = 0;
    k->windowed = 0;
    k->precision = 1;

    /* --- H materialization: forward weld on a scratch copy of h (h is
     * NOT clobbered; no exp/cos at execute).  Ownership: locals nulled
     * the moment their content transfers to k (or is freed). --- */
    WeldBuilt fwd = {nullptr, nullptr, 0, nullptr, nullptr, false, false};
    double *hwork = nullptr;
    double *hMul = nullptr;
    double *hSwap = nullptr;
    double *hcNat = nullptr;
    WeldBuilt b = {nullptr, nullptr, 0, nullptr, nullptr, false, false};
    try {
        fwd = weldBuildKernel(N, radices, kStages, hasFma, /*conv=*/false,
                              /*outIsCaller=*/true /* hwork has no slack
                                for odd-tail padding lanes */,
                              nullptr, nullptr, nullptr, nullptr, nullptr);
        fwd.em->readyRE();
        fft_jit_fn_t ffn = reinterpret_cast<fft_jit_fn_t>(
            const_cast<uint8_t *>(fwd.em->getCode()));
        hwork = static_cast<double *>(
            aligned_alloc(64, weld_pad64((size_t)N * 16)));
        if (hwork == nullptr) {
            throw std::bad_alloc();
        }
        __builtin_memcpy(hwork, h, (size_t)N * 16);
        ffn(hwork); /* hwork = H (natural order), original size */

        int kinds8[8];
        int B8[8];
        B8[0] = radices[0];
        for (int t = 1; t < kStages; ++t) {
            B8[t] = B8[t - 1] * radices[t];
        }
        for (int t = 0; t < kStages; ++t) {
            const int nblocks = (t == 0) ? (N / B8[0]) : (N / B8[t]);
            weld_stage_kind(t, kStages, radices[t], (t == 0) ? 1 : B8[t - 1],
                            nblocks, &kinds8[t]);
        }
        const bool finalCodelet = !(kinds8[kStages - 1] == WSK_OWN_V2 ||
                                    kinds8[kStages - 1] == WSK_OWN_32);
        const bool stage0Own = (kinds8[0] == WSK_OWN_V2 ||
                                kinds8[0] == WSK_OWN_32 ||
                                kinds8[0] == WSK_OWN_S2);
        const bool needStoreFusion = !finalCodelet || !stage0Own;
        const bool needLoadFusion = finalCodelet && stage0Own;
        if (needStoreFusion) {
            hMul = static_cast<double *>(
                aligned_alloc(64, weld_pad64((size_t)N * 16)));
            hSwap = static_cast<double *>(
                aligned_alloc(64, weld_pad64((size_t)N * 16)));
            if (hMul == nullptr || hSwap == nullptr) {
                throw std::bad_alloc();
            }
            for (int i = 0; i < N; ++i) {
                const double hr = hwork[2 * i];
                const double hi = hwork[2 * i + 1];
                hMul[2 * i] = hr;
                hMul[2 * i + 1] = hi;
                hSwap[2 * i] = hi;
                hSwap[2 * i + 1] = hasFma ? hr : -hr;
            }
        }
        if (needLoadFusion) {
            hcNat = static_cast<double *>(
                aligned_alloc(64, weld_pad64((size_t)N * 32)));
            if (hcNat == nullptr) {
                throw std::bad_alloc();
            }
            for (int i = 0; i < N; ++i) {
                const double hr = hwork[2 * i];
                const double hi = hwork[2 * i + 1];
                hcNat[4 * i + 0] = hr;  /* conj(H) slot: (hr, -hi, -hi, hr) */
                hcNat[4 * i + 1] = -hi;
                hcNat[4 * i + 2] = -hi;
                hcNat[4 * i + 3] = hr;
            }
        }
        free(hwork);
        hwork = nullptr;
        delete fwd.em;
        free(fwd.arena);
        free(fwd.scratch);
        free(fwd.scratch2);
        fwd.em = nullptr;
        fwd.arena = nullptr;
        fwd.scratch = nullptr;
        fwd.scratch2 = nullptr;

        b = weldBuildKernel(N, radices, kStages, hasFma, /*conv=*/true,
                            /*outIsCaller=*/true,
                            (hMul != nullptr) ? hMul : hwork,
                            (hSwap != nullptr) ? hSwap : hwork, hcNat,
                            nullptr, nullptr);
        k->emitter = b.em;
        k->scratch_d = b.scratch;
        k->h_mul_d = hMul;
        k->h_swap_d = hSwap;
        k->weld_arena = b.arena;
        k->weld_arena_bytes = b.arena_bytes;
        k->weld_hc = hcNat;
        b.em = nullptr;
        b.arena = nullptr;
        b.scratch = nullptr;
        b.scratch2 = nullptr;
        hMul = nullptr;
        hSwap = nullptr;
        hcNat = nullptr;
        k->emitter->readyRE();
        k->fn = reinterpret_cast<fft_jit_fn_t>(
            const_cast<uint8_t *>(k->emitter->getCode()));
        k->code_size = static_cast<int>(k->emitter->getSize());
    } catch (...) {
        fft_jit_kernel_destroy(k);
        delete b.em;
        free(b.arena);
        free(b.scratch);
        free(b.scratch2);
        delete fwd.em;
        free(fwd.arena);
        free(fwd.scratch);
        free(fwd.scratch2);
        free(hMul);
        free(hSwap);
        free(hcNat);
        free(hwork);
        return nullptr;
    }

    if (getenv("FFT_WELD_LOG") != nullptr) {
        fprintf(stderr, "[weld] N=%d plan=", N);
        for (int t = 0; t < kStages; ++t) {
            fprintf(stderr, "%s%d", t ? "x" : "", radices[t]);
        }
        fprintf(stderr,
                " code=%d fma=%d gen=%d pool=%d timed=%d search_ns=%.0f%s\n",
                k->code_size, hasFma ? 1 : 0, genSz, poolSz, timedCands,
                searchNs, forced ? " (forced)" : "");
    }
    return k;
}

/*
 * topic jit-anyN-fwd-inv (DELTA F1 / draft {#API-FFT-013}): SINGLE
 * forward / inverse transform kernels at ORIGINAL non-2^k SIZE N — the
 * WSK engine pulled out of the conv weld two-half context (DESIGN §L-Q).
 *
 *   forward: the conv=0 mode-0 direct form the search already times —
 *     stage-0 digitrev gather, per-stage searched leaves (own vector /
 *     FFTW codelet / F2 hand leaves), plain final store to the caller's
 *     io (no pointwise interface; mode-1 degenerates to a plain store).
 *     Zero H materialization (no operator to apply).
 *   inverse: the boundary-sunk conjugate trick on the same geometry
 *     (SchedEmitterI precedent): entry conj sunk into the stage-0 gather
 *     loads (mode-3 sign flip minus the conj(H) multiply) or the
 *     conjugating PERMUTE on a codelet stage-0; exit conj(.)·(1/N) at
 *     the final store (own = the mode-2 form itself; codelet = the
 *     WELD_RUN_INVN sweep).  Output order matches FFTW_BACKWARD / N.
 *
 * Decomposition = the conv framework verbatim (FFT_WELD_ORDER force /
 * FFT_WELD_SEARCH + budget-guarded short-timing / greedy anchor; factor
 * domain = weld_factors incl. the R1 vendored radix 11/14 n1fv leaves —
 * 11/14 stages walk the WSK_N1 n1fv+TW-sweep mode like radix 13).
 * Requires AVX2+FMA3
 * (fail-fast, [[CON-FFT-002]]).  Returns NULL on illegal N (pow2 stays
 * on the family path — the entries never see it; N < 6 / > 32768 /
 * unfactorable), missing features, emit failure, or OOM — never a
 * silent fallback.  Topic-local, NOT public API entries (the public face
 * is fft_plan_create_fp64 / _inverse in the host layer).
 */
static fft_jit_kernel *anyN_create_kernel(int N, bool inverse) {
    if (N < 6 || N > 32768 || is_pow2(N)) {
        return nullptr;
    }
    if (!cpu_has_avx() || !cpu_has_fma()) {
        return nullptr; /* fail-fast, no silent fallback */
    }
    weld_codelets_init();
    if (!weld_factorable(N)) {
        return nullptr;
    }
    const bool hasFma = cpu_has_fma() &&
                        getenv("FFT_SCHED_NOFMA") == nullptr;

    int radices[8] = {0};
    int kStages = 0;
    bool forced = false;
    const char *ord = getenv("FFT_WELD_ORDER");
    if (ord != nullptr && *ord != '\0') {
        int rr[8];
        int nr = 0;
        long prod = 1;
        bool ok = true;
        const char *q = ord;
        while (*q != '\0') {
            char *end = nullptr;
            const long v = strtol(q, &end, 10);
            bool legal = false;
            for (int fi = 0; fi < weld_n_factors; ++fi) {
                if (v == weld_factors[fi]) {
                    legal = true;
                }
            }
            if (end == q || nr >= 8 || !legal) {
                ok = false;
                break;
            }
            rr[nr++] = (int)v;
            prod *= v;
            q = end;
            if (*q == ',') {
                ++q;
            } else if (*q != '\0') {
                ok = false;
                break;
            }
        }
        if (ok && prod == N && nr >= 2 && weld_viable(N, rr, nr)) {
            for (int i = 0; i < nr; ++i) {
                radices[i] = rr[i];
            }
            kStages = nr;
            forced = true;
        }
    }
    double searchNs = 0.0;
    int timedCands = 0, poolSz = 0, genSz = 0;
    if (!forced) {
        const char *srch = getenv("FFT_WELD_SEARCH");
        const bool off = srch != nullptr && (strcmp(srch, "0") == 0 ||
                                             strcmp(srch, "off") == 0);
        if (!off) {
            weld_search(N, hasFma, radices, &kStages, &searchNs,
                        &timedCands, &poolSz, &genSz);
        }
    }
    if (kStages == 0) {
        int rad[8];
        int k = 0;
        weld_anchor(N, true, rad, &k);
        if (k == 0) {
            return nullptr;
        }
        for (int i = 0; i < k; ++i) {
            radices[i] = rad[i];
        }
        kStages = k;
    }

    fft_jit_kernel *k = new (std::nothrow) fft_jit_kernel();
    if (k == nullptr) {
        return nullptr;
    }
    k->emitter = nullptr;
    k->lut = nullptr;
    k->lut_d = nullptr;
    k->win = nullptr;
    k->win_dup = nullptr;
    k->scratch_d = nullptr;
    k->h_mul_d = nullptr;
    k->h_swap_d = nullptr;
    k->weld_arena = nullptr;
    k->weld_arena_bytes = 0;
    k->weld_hc = nullptr;
    k->fn = nullptr;
    k->code_size = 0;
    k->windowed = 0;
    k->precision = 1;

    WeldBuilt b = {nullptr, nullptr, 0, nullptr, nullptr, false, false};
    try {
        b = weldBuildKernel(N, radices, kStages, hasFma, /*conv=*/false,
                            /*outIsCaller=*/true,
                            nullptr, nullptr, nullptr, nullptr, nullptr,
                            /*invSingle=*/inverse);
        k->emitter = b.em;
        k->scratch_d = b.scratch;
        k->weld_arena = b.arena;
        k->weld_arena_bytes = b.arena_bytes;
        /* b.scratch2 (odd-tail N1 final routing buffer): the emitted code
         * bakes its address; ownership mirrors the Trunk conv create (the
         * allocation deliberately outlives the WeldBuilt, freed never on
         * this topic-local path — recorded in evidence). */
        b.em = nullptr;
        b.arena = nullptr;
        b.scratch = nullptr;
        b.scratch2 = nullptr;
        k->emitter->readyRE();
        k->fn = reinterpret_cast<fft_jit_fn_t>(
            const_cast<uint8_t *>(k->emitter->getCode()));
        k->code_size = static_cast<int>(k->emitter->getSize());
    } catch (...) {
        fft_jit_kernel_destroy(k);
        delete b.em;
        free(b.arena);
        free(b.scratch);
        free(b.scratch2);
        return nullptr;
    }

    if (getenv("FFT_WELD_LOG") != nullptr) {
        fprintf(stderr, "[anyN%s] N=%d plan=", inverse ? "-inv" : "", N);
        for (int t = 0; t < kStages; ++t) {
            fprintf(stderr, "%s%d", t ? "x" : "", radices[t]);
        }
        fprintf(stderr,
                " code=%d fma=%d gen=%d pool=%d timed=%d search_ns=%.0f%s\n",
                k->code_size, hasFma ? 1 : 0, genSz, poolSz, timedCands,
                searchNs, forced ? " (forced)" : "");
    }
    return k;
}

extern "C" fft_jit_kernel *fft_jit_kernel_create_anyN_forward_fp64(int N) {
    return anyN_create_kernel(N, /*inverse=*/false);
}

extern "C" fft_jit_kernel *fft_jit_kernel_create_anyN_inverse_fp64(int N) {
    return anyN_create_kernel(N, /*inverse=*/true);
}

/* topic jit-anyN-fwd-inv: host-layer feature gate for the non-pow2
 * coverage-set entries ([[CON-FFT-002]] fail-closed: no AVX2+FMA3 -> the
 * non-pow2档 rejects NULL; the pow2 family path is unaffected). */
extern "C" int fft_jit_has_avx2_fma(void) {
    return (cpu_has_avx() && cpu_has_fma()) ? 1 : 0;
}

extern "C" fft_jit_kernel *fft_jit_kernel_create_gauss_window(int N) {
    return fft_jit_kernel_create_ex<float>(N, 1);
}

/* topic jit-ps-sched-fuse (L-B / DELTA F2): opt-in FUSED cyclic-conv
 * kernel — ONE execute welds the forward half (final stage multiplies
 * X[k]·H[k] and conjugates at the store boundary) with the
 * conjugate-trick inverse half (stage-0 gathers the natural conj(X·H)
 * directly; final stage stores conj(·)/N).  H is plan-owned (copied
 * into the kernel LUT block; separate tables from the twiddles, ARCH-
 * FFT-016).  Internal entry used by the topic host wrapper; the public
 * ABI (fft_plan_create_conv / fft_execute) is unchanged.  NULL on
 * N < 32 / illegal N / emit / OOM (caller keeps the three-segment
 * path for those N). */
extern "C" fft_jit_kernel *fft_jit_kernel_create_conv_fused(
    int N, const float *H_interleaved) {
    if (N < 32 || H_interleaved == nullptr) {
        return nullptr;
    }
    return schedF_create_kernel(N, 0, H_interleaved);
}

extern "C" int fft_gauss_window_fill(int N, float *out) {
    return fill_gauss_window(N, out);
}

extern "C" const float *fft_jit_kernel_gauss_lut(const fft_jit_kernel *k) {
    return (k != nullptr) ? k->win : nullptr;
}

extern "C" fft_jit_fn_t fft_jit_kernel_fn(const fft_jit_kernel *k) {
    return (k != nullptr) ? k->fn : nullptr;
}

extern "C" void fft_jit_kernel_destroy(fft_jit_kernel *k) {
    if (k == nullptr) {
        return;
    }
    delete k->emitter;
    free(k->lut);
    free(k->lut_d);
    free(k->win);
    free(k->win_dup);
    free(k->scratch_d);
    free(k->scratch_f);
    /* topic jit-fp64-conv-fuse: conv weld H tables (nullptr on every
     * forward / fp32 kernel — value-init keeps those paths untouched) */
    free(k->h_mul_d);
    free(k->h_swap_d);
    /* topic jit-codelet-weld: weld arena + load-fusion conj(H) slots
     * (nullptr on every other kernel — value-init) */
    free(k->weld_arena);
    free(k->weld_hc);
    /* topic jit-fp64-linear-conv: OLS weld spectrum transit buffer
     * (nullptr on every other kernel — value-init) */
    free(k->spec_d);
    delete k;
}

extern "C" int fft_jit_kernel_code_size(const fft_jit_kernel *k) {
    return (k != nullptr) ? k->code_size : 0;
}
