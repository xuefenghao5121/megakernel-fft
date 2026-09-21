#ifndef MEGAKERNEL_FFT_JIT_FFT_JIT_H
#define MEGAKERNEL_FFT_JIT_FFT_JIT_H

/*
 * poc/jit-fp64-forward/jit/fft_jit.h — Xbyak JIT kernel surface (internal).
 *
 * Copy-then-edit of Trunk src/jit/fft_jit.h (family emitter, all ps-based)
 * to add the opt-in fp64 forward emitter (draft {#API-FFT-008} /
 * {#BEH-FFT-009}).  Clauses: {#ARCH-FFT-008} (JIT fixed-N execute path),
 * {#ARCH-FFT-009} (twiddle LUT), {#ARCH-FFT-011} (N=2^n family twiddle
 * specialization), {#ARCH-FFT-015} (gauss window LUT + first-stage fuse).
 * This header is internal; the public ABI is include/fft.h.
 *
 * Precision-parameterized emitter: the SAME butterfly / complex-mul /
 * load-store generation logic branches ps/pd by element type (fp32
 * preserved, fp64 added).  double complex = 16 bytes; xmm holds 1 double
 * complex (SIMD width halved: mulps→mulpd, addps→addpd, vhsubps→vhsubpd).
 * Twiddle table float→double on the already-promoted materialization path
 * (MUST NOT build a second twiddle subsystem).  fp32 kernels MUST NOT
 * regress.
 *
 * Plan-once / execute-many: fft_jit_kernel_create() / _create_fp64() emit
 * the machine code and (for n=9..15) the twiddle LUT once.  execute is a
 * single indirect call through fft_jit_kernel_fn() - it never re-enters
 * fftw execute and MUST NOT call exp.
 */

#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fft_jit_kernel fft_jit_kernel;

/* Executable kernel entry: X(k) = DFT of io[0..N-1], in-place, natural
 * order out (BEH-FFT-002 / BEH-FFT-003).  io is caller-owned.  Typed as a
 * raw pointer because the emitted machine code only sees the buffer address
 * (rdi) and never the C complex type ({#API-FFT-001}). */
typedef void (*fft_jit_fn_t)(void *io);

/*
 * Emit a family kernel for N = 2^n (n=0..15).  Returns NULL on unsupported N
 * (N<1, non-power-of-two, or N>32768), missing CPU features (AVX), or
 * emission/OOM failure (DESIGN failure modes).  The host layer dispatches
 * N=2^n n=0..15 here and every other N to the FFTW-derived adapt path.
 *
 *   n=0 (N=1):    degenerate DFT identity kernel.
 *   n=1..4:       specialized immediate kernels - register-resident,
 *                 fully unrolled, bit-reversal folded into the stage-2
 *                 load offsets, single load pass + single store pass,
 *                 no bitrev/stage loops; twiddles baked as a rip-relative
 *                 constant pool (no execute-resident LUT; no vhsubps for
 *                 N=8/16, no vzeroupper at any of these sizes).
 *   n=5..8:       loop immediate path - twiddles baked as a rip-relative
 *                 constant pool in the instruction stream (no
 *                 execute-resident LUT; R0 representative N=256).
 *   n=9..15:      LUT-address path - plan-time LUT, base address baked as a
 *                 movabs immediate; execute uses vmovups ([[ARCH-FFT-009]]).
 *                 n=11..15 reuse the n=9..10 path with a longer table.
 */
fft_jit_kernel *fft_jit_kernel_create(int N);

/*
 * Opt-in fp64 forward kernel (draft {#API-FFT-008} / {#BEH-FFT-009}).
 * Precision-parameterized emitter: same butterfly/complex-mul/load-store
 * generation logic branches ps/pd by element type (fp32 preserved, fp64
 * added).  double complex = 16 bytes; xmm holds 1 double complex (SIMD
 * width halved).  Twiddle table float→double on the same materialization
 * path.  Same family tiering as fp32: n=0 identity, n=1..4 immediate,
 * n=5..8 loop immediate, n=9..15 LUT-address.  Returns NULL on illegal N /
 * missing AVX / emit / OOM.  MUST NOT silently fall back to fp32.
 */
fft_jit_kernel *fft_jit_kernel_create_fp64(int N);

/*
 * Topic jit-fp64-conv-fuse (L-I fused conv weld, draft {#API-FFT-011}):
 * opt-in fp64 FUSED cyclic-convolution kernel at the padded transform
 * size Np = 2^n (4 <= Np <= 32768).  h_pad is the caller's ZERO-PADDED
 * kernel (Np complex doubles, natural order); the plan materializes
 * H[k] = F(h_pad) by reusing the promoted fp64 forward SchedEmitter on a
 * scratch copy (h_pad is NOT clobbered) and builds two separate plan-time
 * tables hMul (natural H) / hSwap (pre-swapped twin, sign following the
 * FMA3 gate) — separate tables from the twiddle LUT, no fourth twiddle
 * subsystem.
 *
 * ONE welded kernel completes forward -> pointwise -> inverse: the
 * forward half's final-stage stores multiply X[k]*H[k] and conjugate AT
 * THE STORE BOUNDARY (per-store-vector 32B H pair + twin, natural-order
 * indexed relative to the scratch k-cursor); the inverse half's stage-0
 * gather consumes the natural conj(X*H) directly (no wrapper sweep
 * between halves); the inverse half's final-stage stores apply
 * conj(.) * (1/Np) via one vmulpd against an (invN, -invN) rip constant.
 * The intermediate spectrum transits the caller buffer exactly once.
 * Output = Np-point cyclic convolution of the (padded) input with h_pad,
 * in-place, natural order.  Decomposition: n <= 5 forced k>=2 splits
 * (the weld needs >= 2 stages); n >= 6 reuses the promoted plan-time
 * (decomposition x tile) search.  Returns NULL on illegal Np / NULL
 * h_pad / missing AVX / emit / OOM.  MUST NOT silently fall back to the
 * fp64 forward path or fp32.
 */
fft_jit_kernel *fft_jit_kernel_create_conv_fused_fp64(int Np,
                                                      const double *h_pad);

/*
 * topic jit-codelet-weld (draft {#BEH-FFT-016}; reuses {#API-FFT-011}):
 * opt-in fp64 FUSED cyclic-convolution weld at ORIGINAL SIZE N (>= 6,
 * non-power-of-two, factorable over the weld domain
 * {2,3,4,5,6,7,8,9,10,12,13,15,16,20,25,32} into >= 2 stages).  Same
 * fused-boundary semantics as the pow2 weld above (the weld layer is
 * leaf-source agnostic); leaves are searched per stage: pow2 factors run
 * the own vector leaves, 3/5-family factors run vendored FFTW codelets
 * (topic-compiled AVX2+FMA subset library; calling convention locked by a
 * numeric call-in/call-out probe).  Requires AVX2+FMA3 (the vendored
 * simd-avx2.h uses FMA in VZMUL/VZMULJ); without it the host layer keeps
 * the promoted pad route.  h is NOT clobbered.  Returns NULL on illegal N
 * / NULL h / unfactorable N / missing features / emit / OOM.
 * Topic-local knobs (no public API): FFT_WELD=off, FFT_WELD_SEARCH=off,
 * FFT_WELD_ORDER=r1,r2,... (forced decomposition), FFT_WELD_LOG=1.
 */
fft_jit_kernel *fft_jit_kernel_create_conv_weld_fp64(int N, const double *h);

/*
 * topic jit-anyN-fwd-inv (DELTA F1/F3; draft {#API-FFT-013} /
 * {#BEH-FFT-019}): SINGLE fp64 forward / inverse transform kernel at
 * ORIGINAL non-2^k size N (>= 6, factorable over weld_factors incl. the
 * R1 vendored radix 11/14 n1fv leaves (11/14 = WSK_N1 n1fv+TW-sweep mode,
 * the radix-13 incumbent pattern), >= 2
 * stages) — the WSK engine pulled out of the conv weld two-half context
 * (mode-0 direct forward / boundary-sunk conjugate inverse).  pow2 N never
 * reaches here (the family path owns it, zero-change guard).  Returns
 * NULL on illegal N / missing AVX2+FMA / emit / OOM — never a silent
 * fallback.  Topic-local, NOT public API (the public face is
 * fft_plan_create_fp64 / fft_plan_create_fp64_inverse in the host layer).
 */
fft_jit_kernel *fft_jit_kernel_create_anyN_forward_fp64(int N);
fft_jit_kernel *fft_jit_kernel_create_anyN_inverse_fp64(int N);

/* topic jit-anyN-fwd-inv: runtime AVX2+FMA3 probe for the host entry gate
 * (fail-closed NULL on the non-pow2档 without them; pow2 unaffected). */
int fft_jit_has_avx2_fma(void);

/*
 * topic jit-fp64-inverse (amend resume 2026-09-17, DELTA F3; draft
 * {#API-FFT-009} / {#BEH-FFT-010}): opt-in FUSED fp64 INVERSE kernel at
 * N = 2^n (32 <= N <= 32768).  IFFT(x) = conj(F(conj(x))) / N as ONE
 * welded kernel — copy-then-edit of the conv weld's single-half form
 * (SchedEmitterI; the Trunk SchedEmitterC stays untouched):
 *   - ENTRY: the conjugate is sunk into the stage-0 gather LOAD boundary
 *     (per gathered pair vector one vxorpd against the rip sign mask);
 *   - MIDDLE: the promoted forward sweeps / twiddles run unchanged (no
 *     inverse twiddle table, no second emitter, no new ps/pd branch);
 *   - EXIT: the final stage stores conj(.) * (1/N) via one vmulpd
 *     against the (invN, -invN) rip constant (the conv half-B idiom).
 * Decomposition = the same-N fp64 forward framework (n <= 5 forced k>=2
 * splits; n >= 6 the promoted plan-time (decomposition x tile) search).
 * The N < 32 family floor stays with the HOST wrapper (forward kernel +
 * independent conj / conj*invN sweeps — the retired three-segment form).
 * Returns NULL on illegal N / missing AVX / emit / OOM.  MUST NOT
 * silently fall back to the fp64 forward path or fp32.
 */
fft_jit_kernel *fft_jit_kernel_create_inverse_fused_fp64(int N);

/* topic jit-fp64-linear-conv (draft {#API-FFT-012} / {#BEH-FFT-017}):
 * OLS stream entry of the fused linear-conv kernel.  rdi = padded input
 * stream P = [0^{nh-1} | x | 0-tail] (double complex), rsi = y_out (the
 * valid-output stream, sized >= nblocks*hop + (nh-1) slack so the discard
 * zone / spectrum residue never overruns), rdx = nblocks.  One call runs
 * the emitted block loop: window formation via the plan-time ROTATED
 * gather table (the overlap ring sunk into the stage-0 load boundary),
 * hop advance, valid-segment stores at the inverse final-store boundary
 * (discard zone not written in the default skip mode).  nblocks == 1 is
 * the wrapper-fed single block (O-ii form b).  Topic-local, NOT a public
 * API entry. */
typedef void (*fft_jit_ols_fn_t)(void *padded_stream, void *y_out,
                                 long nblocks);

/*
 * topic jit-fp64-linear-conv (DELTA F1/F2; draft {#API-FFT-012}): opt-in
 * fp64 FUSED overlap-save LINEAR-convolution weld at the pow2 block size
 * N = 2^n (4 <= N <= 32768), kernel length 0 < nh <= N.  h_pad = the
 * caller's ZERO-PADDED kernel (N complex doubles, natural order); H[k] =
 * F(h_pad) materialized at plan time on a scratch copy (h_pad NOT
 * clobbered; hMul/hSwap tables separate from the twiddle LUT — the
 * conv-fuse pattern, no fourth twiddle subsystem).
 *
 * ONE welded kernel per block (the conv weld machinery with the OLS
 * boundary edits — SchedEmitterO, a copy-then-edit of SchedEmitterC which
 * stays untouched):
 *   - half A stage-0 gathers the window through a ROTATED table
 *     (slot v <- natural (v+nh-1) mod N): the OLS window shift sinks into
 *     the load boundary, so the wrap-around-polluted head lands at the
 *     END of the cyclic result and the valid segment is the natural
 *     PREFIX [0, N-nh];
 *   - half B final stores write ONLY the valid prefix (the discard zone
 *     [N-nh+1, N-1] not written — skip mode, default; or written-then-
 *     abandoned — write mode, FFT_OLS_DISCARD=write adjudication knob);
 *   - the intermediate spectrum transits a plan-owned natural buffer S
 *     exactly once (split-base body: window base rdi / out base rdx), so
 *     the SAME body serves the in-place single-block contract face
 *     (fn(io), win=out=io) and the stream face (fn_stream(P, y, B) with
 *     the emitted block loop = O-ii form a).
 * Decomposition / knobs = the conv create framework (n <= 5 forced k >= 2
 * splits; n >= 6 the promoted plan-time (decomposition x tile) search;
 * FFT_SCHED_* semantics inherited).  Topic-local knob FFT_OLS_DISCARD =
 * skip|write pins the discard-zone handling.  Returns NULL on illegal
 * (N, nh) / NULL h_pad / missing AVX / emit / OOM.  MUST NOT silently
 * fall back to the fp64 forward / conv / fp32 paths.
 */
fft_jit_kernel *fft_jit_kernel_create_ols_fused_fp64(int N,
                                                     const double *h_pad,
                                                     int nh);

/* OLS stream entry of the OLS kernel (NULL unless the kernel is OLS). */
fft_jit_ols_fn_t fft_jit_kernel_ols_stream_fn(const fft_jit_kernel *k);


/*
 * Opt-in fused Gaussian-window kernel ({#ARCH-FFT-015} / {#API-FFT-003}).
 * Plan-time G[n] = exp(-(n-N/2)^2 / (2 σ^2)) with σ = N/8, then ∑G = 1.
 * Window LUT is a separate table from the twiddle LUT.  Execute MUST NOT
 * call exp.  Returns NULL on N<1 / non-family / AVX missing / emit / OOM /
 * LUT failure.  Missing G is fail-fast (NULL), never a silent unwindowed
 * kernel.
 */
fft_jit_kernel *fft_jit_kernel_create_gauss_window(int N);

/*
 * topic jit-ps-sched-fuse (L-B, internal -- NOT a public ABI entry): opt-in
 * FUSED cyclic-convolution kernel for N >= 32.  One execute = forward half
 * (final stage multiplies X[k]*H[k] + conjugates at the STORE boundary) +
 * conjugate-trick inverse half (stage-0 gathers the natural conj(X*H)
 * directly, final stage stores conj(.)/N at the store boundary).  The
 * wrapper sweeps of the three-segment path (pointwise / conj / conj*invN)
 * are eliminated; the intermediate spectrum transits io exactly once.
 * H is copied into plan-owned tables (separate from the twiddle LUT).
 * Returns NULL on N < 32 / illegal N / emit / OOM -- the caller keeps the
 * three-segment path for those sizes.  MUST NOT be used on non-family N.
 */
fft_jit_kernel *fft_jit_kernel_create_conv_fused(
    int N, const float *H_interleaved); /* re,im interleaved, N complex */

/* Fill caller-owned G[0..N-1] with the plan-time window (σ = N/8, ∑G=1).
 * Same formula the fused kernel materializes.  Returns 0 or -1. */
int fft_gauss_window_fill(int N, float *out);

/* Plan-time G table (natural order), or NULL on the unwindowed kernel. */
const float *fft_jit_kernel_gauss_lut(const fft_jit_kernel *k);

/* Cached executable entry (stable for the kernel's lifetime; NULL only if
 * k is NULL).  Callers MUST NOT regenerate per execute ({#ARCH-FFT-008}). */
fft_jit_fn_t fft_jit_kernel_fn(const fft_jit_kernel *k);

/* Release the kernel (code + twiddle LUT + window LUT).  Safe to pass NULL. */
void fft_jit_kernel_destroy(fft_jit_kernel *k);

/* Emitted machine-code size in bytes (diagnostics / evidence only). */
int fft_jit_kernel_code_size(const fft_jit_kernel *k);

/* ========================================================================
 * topic jit-joint-search-batched (L-E + L-F) — INTERNAL surface (NOT the
 * public ABI; public additions are only fft_plan_create_windowed_batch /
 * fft_execute_windowed_batch in include/fft.h).
 * ======================================================================== */

/* Batched single-core kernel entry: sweeps all B blocks in one call.
 * rdi = io base, rsi = stride in float complex ELEMENTS (runtime);
 * B and the (placement, decomposition, tile) shape are plan-time. */
typedef void (*fft_jit_fn2_t)(void *io, long stride);

/* Fusion placement classes (L-E search dimension; R2 adds PRESWEEP). */
enum {
    FFT_PLACE_STORE = 0,      /* multiply sunk into the final STORE port
                               * (conv: the base finalMode=1 form) */
    FFT_PLACE_LOAD = 1,       /* multiply sunk into the consumer stage-0
                               * LOAD/gather port (window gt rows = the
                               * base gauss form; conv = H rows on the
                               * inverse-half gather, NEW) */
    FFT_PLACE_STANDALONE = 2, /* wrapper-level independent multiply sweep
                               * (not fused — the search's null
                               * hypothesis) */
    FFT_PLACE_PRESWEEP = 3    /* topic R2 (amend r2, DELTA F5 / L-G): the
                               * window runs as ONE batch-level streaming
                               * pre-sweep before the PLAIN (windowless)
                               * batch kernel — the composed leg's sweep
                               * shape without the per-block call face */
};

/* Batch kernel forms (L-F search dimension; R2 adds SLIM). */
enum {
    FFT_BATCH_FORM_SINGLE = 0, /* one JIT kernel sweeps all B blocks */
    FFT_BATCH_FORM_LOOP = 1,   /* host loop calls the per-block kernel */
    FFT_BATCH_FORM_SLIM = 2    /* topic R2 (amend r2, DELTA F5 / L-G):
                                * single-core batch form with the SLIMMED
                                * block loop — block invariants hoisted
                                * out (window-row base, stride bytes), a
                                * plan-time-specialized contiguous
                                * stride==N entry, generic-stride entry
                                * kept for stride > N */
};

/* Stats of the most recent plan-time joint search (single-threaded bench
 * use; diagnostics / evidence only — never a public contract). */
typedef struct {
    int radices[8];
    int k;
    int tile;
    int place;    /* chosen placement (FFT_PLACE_*) */
    int form;     /* chosen batch form (FFT_BATCH_FORM_*) */
    double search_ns;
    int timed;
    int pool;
    int refined;
    double best_place[4]; /* best measured ns/block per placement class
                           * (batch: [load, standalone, presweep, unused];
                           * conv uses [store, load, standalone]) */
    int forced;   /* 1 = a topic-local env knob pinned a dimension */
    int btest;    /* topic R2 (L-H): actual short-test block count the
                   * per-block amortized number was measured over (>= 4
                   * when B >= 4) */
} fft_jit_search_stats;

/* Last conv-placement-search stats of this process (zeroed at start). */
const fft_jit_search_stats *fft_jit_conv_search_last(void);

/* Last batched-windowed-search stats of this process. */
const fft_jit_search_stats *fft_jit_batch_search_last(void);

/*
 * topic jit-joint-search-batched (L-E): conv kernel with the fusion
 * PLACEMENT chosen by a plan-time joint search over (decomposition x
 * tile x placement in {store, load, standalone}).
 *
 *   store      = the base fused form (forward half multiplies X[k]*H[k]
 *                + conjugates at the final STORE boundary);
 *   load       = NEW: forward half stores plain X; the inverse half's
 *                stage-0 gather multiplies by conj(H)-rows at the LOAD
 *                boundary (r1=32 chains excluded — no pair-gather form);
 *   standalone = the kernel returned is the PLAIN searched kernel; the
 *                host composes the three-segment path (its own sweeps).
 *
 * forcedPlace < 0 = search (auto); otherwise pins the placement (env
 * FFT_CONV_PLACE=store|load|standalone does the same).  FFT_JOINT_SEARCH
 * =off restricts the search to the fixed store placement (the base
 * two-dim decomposition x tile search — H-A cost comparison leg).
 * Returns NULL on illegal N / emit / OOM.  H is copied into plan-owned
 * tables (separate from the twiddle LUT).
 */
fft_jit_kernel *fft_jit_kernel_create_conv_searched(int N,
                                                    const float *H_interleaved,
                                                    int forcedPlace);

/*
 * topic jit-joint-search-batched (L-F): batched windowed-forward kernel
 * for B independent blocks of N elements.  Plan-time joint search over
 * (placement in {load, standalone, presweep} x decomposition x tile x
 * batch form in {single-core, per-block loop, slim single-core}) for the
 * given (N, B); the window G (sigma=N/8, sum=1) is materialized plan-side.
 * For the standalone / presweep placements the returned kernel is PLAIN
 * and the host runs its own window sweep before it;
 * fft_gauss_window_fill() provides the same G.  N <= 8 (no schedulable
 * chain) uses the fixed legacy per-block fused kernel.
 * fft_jit_kernel_is_batch_single() tells which fn to use.
 *
 * topic-local debug knobs (R2, amend r2): FFT_WINPLACE=gather|presweep|
 * standalone pins the window placement (gather = the load-boundary gt
 * rows; FFT_BATCH_PLACE=load|standalone is the legacy alias);
 * FFT_BATCH_FORM=single|loop|slim|bare pins the batch form (slim = the
 * R2 slimmed block-loop single-core form; bare = the base-shape anchor:
 * base two-dim search decomposition + presweep sweep + per-block plain
 * kernel, DELTA F6/L-H).  Returns NULL on illegal (N, B) / emit / OOM —
 * fail-fast.
 */
fft_jit_kernel *fft_jit_kernel_create_windowed_batch(int N, int B);

/* 1 = the batch kernel is the single-core form (execute via
 * fft_jit_kernel_fn2 with (io, stride)); 0 = per-block form (execute
 * via fft_jit_kernel_fn on each block base). */
int fft_jit_kernel_is_batch_single(const fft_jit_kernel *k);

/* Single-core batch entry (NULL unless the kernel is batch single). */
fft_jit_fn2_t fft_jit_kernel_fn2(const fft_jit_kernel *k);

/* topic R2 (DELTA F5 / L-G): contiguous stride==N-specialized entry of a
 * SLIM single-core batch kernel (NULL unless the kernel is batch slim).
 * The wrapper MUST call it only when stride == N; stride > N keeps the
 * generic fn2 entry. */
typedef void (*fft_jit_fn1c_t)(void *io);
fft_jit_fn1c_t fft_jit_kernel_fn2c(const fft_jit_kernel *k);

#ifdef __cplusplus
}
#endif

#endif /* MEGAKERNEL_FFT_JIT_FFT_JIT_H */
