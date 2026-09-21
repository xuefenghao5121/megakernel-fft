/*
 * poc/jit-codelet-weld/jit/weld_codelets.h — topic jit-codelet-weld (L-K).
 *
 * C surface shared between the topic C++ JIT layer (fft_jit.cpp) and the
 * codelet glue (weld_glue.c): the FFTW vendored codelet registry, the
 * plan-time t1fv W-table generator, and the single JIT->C stage-runner
 * entry (System V AMD64 call boundary: the JIT passes a plan-time constant
 * stage descriptor + the runtime base; all live JIT cursors sit in
 * callee-saved registers across the call).
 *
 * Codelet calling convention = FFTW internal (vendored lock version,
 * third_party/fftw @ base_sha d024c3f3; kernel/ifftw.h +
 * dft/codelet-dft.h):
 *   kdft  n1fv_r(ri, ii, ro, io, stride is, stride os, INT vl, INT ivs, INT ovs)
 *   kdftw t1fv_r(ri, ii, const R *W, stride rs, INT mb, INT me, INT ms)
 * with the x86-64 PRECOMPUTE_ARRAY_INDICES flavor: `stride` is an INT*
 * array of precomputed element indices (stride[i] = i*s in R units), and
 * strides passed raw (ivs/ovs/ms) are R units.  Interleaved complex:
 * ii = ri + 1, io = ro + 1.  vl (and me-mb) must be multiples of VL=2
 * (AVX2 double) or the vector tail lane writes a padding instance — the
 * weld geometry only ever calls with even counts, except the tail paths
 * below which steer the padding lane's stores into scratch slack.
 *
 * The vendored simd-avx2.h uses _mm256_fmaddsub_pd in VZMUL/VZMULJ
 * unconditionally, so every codelet object requires AVX2+FMA3 hardware;
 * the C++ plan layer gates the whole weld path on the runtime feature
 * check ([[CON-FFT-002]] — no FMA3, no codelets; the family contract then
 * stays on the promoted pow2 pad route).
 */

#ifndef MEGAKERNEL_FFT_WELD_CODELETS_H
#define MEGAKERNEL_FFT_WELD_CODELETS_H

#include <stddef.h>

/* fftw internal types.  C side (glue): the real vendored headers.  The
 * vendored config.h belongs to the fp32 build (FFTW_SINGLE, no include
 * guard), so the shim at fftw-codelets/config.h (first on the -I chain)
 * layers include_next + #undef to select the DOUBLE compile read-only.
 * C++ side: an exact type mirror (plain types, no linkage) — R = double,
 * INT = ptrdiff_t, stride = INT* (PRECOMPUTE_ARRAY_INDICES flavor on
 * x86-64) — ABI-identical, verified by the numeric probe. */
#ifndef __cplusplus
#include "config.h"
#undef FFTW_SINGLE
#include "kernel/ifftw.h"
#include "dft/codelet-dft.h"
#else
#include <stddef.h>
#include <stdint.h>
typedef double R;
typedef ptrdiff_t INT;
typedef INT *stride;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* kdft / kdftw codelet function pointer types (internal signatures). */
typedef void (*weld_n1_fn)(const R *ri, const R *ii, R *ro, R *io,
                           stride is, stride os, INT v, INT ivs, INT ovs);
typedef void (*weld_t1_fn)(R *ri, R *ii, const R *W, stride rs, INT mb,
                           INT me, INT ms);

/*
 * Capture the vendored codelets via their own registrars and NUMERICALLY
 * verify each leaf against a reference DFT (call-in/call-out check on the
 * exact production geometry) before it is ever wired into a weld plan.
 * Idempotent; single-threaded (plan-time).  Every leaf that fails its
 * check is reported unavailable (weld_clet_* -> NULL) and the planner
 * simply never picks that radix — no silent convention drift.
 */
void weld_codelets_init(void);
weld_n1_fn weld_clet_n1(int r); /* NULL when r not in set / check failed */
weld_t1_fn weld_clet_t1(int r);
const char *weld_codelets_probe_report(void); /* one-line diagnostics */

/*
 * t1fv W table: (Bp/2) rows (m pairs, m = the instance/twiddle index k)
 * x (r-1) slots x 4 doubles; slot j (=1..r-1) at row + 4*(j-1) holds
 * {W^<jk_0>, W^<jk_1>} as two consecutive complexes (lane-specific
 * twiddles for the VL=2 instances m, m+1).  Content convention is the one
 * the numeric check locks: slot = e^{+2*pi*i*j*k/Bt} (BYTWJ = VZMULJ
 * multiplies by the conjugate -> effective forward twiddle
 * e^{-2*pi*i*jk/Bt}).  weld_t1_w_doubles() sizes the table.
 */
size_t weld_t1_w_doubles(int Bp, int r);
void weld_t1_w_fill(double *W, int Bp, int r);

/* ---- stage runner ------------------------------------------------------
 * One JIT-visible entry.  `base` is the runtime array base (io for
 * stage-0/final consumers, scratch otherwise; plan-time constants are
 * baked as immediates by the JIT, the io pointer is the kernel argument),
 * `base2` the second array base (scratch for the final codelet stage that
 * reads the scratch and writes io).  All descriptors are plan-time
 * constants owned by the kernel object.
 */
enum {
    WELD_RUN_T1 = 1,        /* t1fv per-block sweep (even Bp, twiddles in W) */
    WELD_RUN_N1 = 2,        /* n1fv per-block sweep, in-place (even Bp) */
    WELD_RUN_N1_ODD = 3,    /* n1fv odd-Bp sweep: per-block vl=Bp-1 calls +
                             * one cross-block tail call for k=Bp-1 (padding
                             * lane steered into d->slack) */
    WELD_RUN_N1_CONTIG = 4, /* stage-0 contiguous n1fv on the digitrev
                             * scratch (odd N/r0 tail steered into slack) */
    WELD_RUN_TWS_V2 = 5,    /* twiddle pre-sweep for an n1fv stage, even Bp:
                             * 32B pairs, 64B TW slots (v2 row layout) */
    /* topic jit-anyN-fwd-inv (DELTA F1 inverse single-transform): the
     * digitrev PERMUTE copy optionally conjugates while copying (the
     * inverse entry boundary conj sunk into the stage-0 permutation —
     * the mode-3 sign flip with no conv operator; d->conj selects). */
    WELD_RUN_TWS_SC = 6,    /* twiddle pre-sweep, odd Bp: 16B elements, 32B
                             * TW slots (single-k layout) */
    WELD_RUN_PERMUTE = 7,   /* digitrev copy: dst[i] <- src[tab[i]] */
    WELD_RUN_CONJMUL = 8,   /* boundary sweep: io <- conj(io .* H) (mode 1) */
    WELD_RUN_INVN = 9       /* boundary sweep: io <- conj(io) * invn (mode 2) */
};

typedef struct weld_stage_desc {
    int kind;
    int radix;              /* r */
    long bstep;             /* block stride in R (double) units = B_t*2 */
    long nblocks;           /* N / B_t */
    weld_n1_fn n1;          /* n1fv codelet */
    weld_t1_fn t1;          /* t1fv codelet */
    const double *W;        /* t1fv W table / TW sweep base */
    const INT *rs;          /* t1fv rs array (r entries) */
    const INT *is;          /* n1fv is array (r entries) */
    const INT *os;          /* n1fv os array (r entries) */
    long vl;                /* n1fv instance count / t1fv me bound */
    long ivs, ovs;          /* n1fv instance strides (R units) */
    long base_off;          /* tail-call base offset in R units */
    double *slack;          /* scratch slack for padding-lane stores */
    const uint32_t *tab;    /* PERMUTE gather table (byte offsets) */
    long n;                 /* element count (PERMUTE/CONJMUL/INVN) */
    const double *hMul;     /* CONJMUL natural H pairs (32B-aligned) */
    const double *hSwap;    /* CONJMUL pre-swapped twin */
    double invn;            /* INVN scale */
    long bp;                /* Bp: instances (k values) per block */
    int twPairs;            /* TWS_V2: k-pairs per block (Bp/2) */
    int conj;               /* topic jit-anyN-fwd-inv: PERMUTE conjugating
                             * copy (inverse entry boundary conj; 0 = plain) */
} weld_stage_desc;

void weld_run_stage(const weld_stage_desc *d, double *base, double *base2);

#ifdef __cplusplus
}
#endif

#endif /* MEGAKERNEL_FFT_WELD_CODELETS_H */
