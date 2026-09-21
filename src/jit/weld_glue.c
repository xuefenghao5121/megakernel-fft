/*
 * poc/jit-codelet-weld/jit/weld_glue.c — topic jit-codelet-weld (L-K).
 *
 * FFTW codelet subset library, topic-local:
 *   - registration-capture stubs for the vendored registrars (the codelet
 *     bodies in fftw-codelets/common/ are VERBATIM vendored copies — R1
 *     (DELTA F3, L-S) adds the upstream n1fv_11/n1fv_14; the avx2/
 *     wrappers only set SIMD_HEADER exactly like
 *     third_party/fftw/dft/simd/avx2/*.c; nothing is patched in place);
 *   - numeric call-in/call-out verification of every leaf on the exact
 *     production geometry before any use (convention lock, vendored
 *     version pinned by base_sha);
 *   - the t1fv W-table generator;
 *   - weld_run_stage: the single JIT->C call boundary used by the weld
 *     emitter for codelet stage sweeps, twiddle sweeps, the digitrev
 *     permutation and the fused-convolution boundary sweeps.
 *
 * Compiled with -mavx2 -mfma (the vendored simd-avx2.h requires FMA3 in
 * VZMUL/VZMULJ); the C++ plan layer feature-gates the whole path.
 */

#include "weld_codelets.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* registration capture: the vendored registrars hand the static codelet
 * function pointer to X(kdft_register)/X(kdft_dit_register); we define
 * those symbols here and read back what was registered.  Plan-time,
 * single-threaded, sequential. */

static weld_n1_fn g_cap_n1;
static weld_t1_fn g_cap_t1;

void fftw_kdft_register(planner *p, kdft codelet, const kdft_desc *desc) {
    (void)p;
    (void)desc;
    g_cap_n1 = (weld_n1_fn)codelet;
}

void fftw_kdft_dit_register(planner *p, kdftw codelet, const ct_desc *desc) {
    (void)p;
    (void)desc;
    g_cap_t1 = (weld_t1_fn)codelet;
}

/* the codelet `desc` structs reference &GENUS; provide the symbols
 * (never dereferenced on this path) + the volatile-stride zero global. */
const kdft_genus fftw_dft_n1fsimd_genus_avx2 = {0, 0};
const ct_genus fftw_dft_t1fsimd_genus_avx2 = {0, 0};
const INT fftw_an_INT_guaranteed_to_be_zero = 0;

/* XSIMD(codelet_nXfv_r) == fftw_codelet_nXfv_r_avx2 (double + _avx2) */
typedef void (*weld_reg_fn)(planner *);

/* R1 (DELTA F3, L-S): radix 11 and 14 vendored from the in-repo upstream
 * (common/n1fv_{11,14}.c byte-for-byte + avx2 wrappers); 11/14 stages
 * walk the WSK_N1 n1fv+TW-sweep mode (the radix-13 incumbent pattern —
 * upstream has no t1fv_11/t1fv_14). */
#define WELD_N1_SET WL(3) WL(5) WL(6) WL(7) WL(9) WL(10) WL(11) WL(12) WL(13) WL(14) WL(15) WL(20) WL(25)
#define WELD_T1_SET WL(3) WL(5) WL(6) WL(7) WL(9) WL(10) WL(12) WL(15) WL(20) WL(25)

#define WELD_MAX_RADIX 25

#define WL(r) extern void fftw_codelet_n1fv_##r##_avx2(planner *p);
WELD_N1_SET
#undef WL
#define WL(r) extern void fftw_codelet_t1fv_##r##_avx2(planner *p);
WELD_T1_SET
#undef WL

/* topic jit-anyN-fwd-inv R1 (DELTA F3, L-S): the RETIRED hand-written
 * radix-11 pair (fftw-codelets/hand/{n1fv,t1fv}_11.c, topic-local; the
 * R0 F2 path-B fallback while the in-repo upstream n1fv_11 was missed).
 * Linked with renamed registrars but registered ONLY under
 * FFT_ANYN_LEAVES=hand (the R0 reproduction knob, INTERFACE Config) —
 * the default pool is the vendored set (11 = vendor n1fv_11 via the N1
 * sweep mode, t1fv_11 unavailable exactly like t1fv_13/14); stock =
 * vendored-only == default now that the vendor leaves are in the
 * vendored pool. */
extern void fftw_hand_codelet_n1fv_11_reg(planner *p);
extern void fftw_hand_codelet_t1fv_11_reg(planner *p);

static int weld_hand11_requested(void) {
    const char *v = getenv("FFT_ANYN_LEAVES");
    return v != NULL && strcmp(v, "hand") == 0;
}

static weld_n1_fn g_n1[WELD_MAX_RADIX + 1];
static weld_t1_fn g_t1[WELD_MAX_RADIX + 1];
static int g_probe_n1[WELD_MAX_RADIX + 1]; /* 0 unknown, 1 ok, -1 failed */
static int g_probe_t1[WELD_MAX_RADIX + 1];
static char g_report[512];

/* ------------------------------------------------------------------ */
/* reference helpers (verification only; plan-time) */

static void ref_dft_r(const double *in /* r complexes */, double *out,
                      int r, int twiddleBt, int m) {
    /* out[j] = sum_j' in[j'] * tw(j') * e^{-2pi i j j'/r}, where
     * tw(j') = e^{-2pi i m j'/Bt} when twiddleBt > 0 (t1fv shape) else 1. */
    for (int j = 0; j < r; ++j) {
        double sr = 0.0, si = 0.0;
        for (int jp = 0; jp < r; ++jp) {
            double xr = in[2 * jp], xi = in[2 * jp + 1];
            if (twiddleBt > 0) {
                const double a = -2.0 * M_PI * (double)m *
                                 (double)jp / (double)twiddleBt;
                const double c = cos(a), s = sin(a);
                const double tr = xr * c - xi * s;
                const double ti = xr * s + xi * c;
                xr = tr;
                xi = ti;
            }
            const double a = -2.0 * M_PI * (double)j * (double)jp /
                             (double)r;
            sr += xr * cos(a) - xi * sin(a);
            si += xr * sin(a) + xi * cos(a);
        }
        out[2 * j] = sr;
        out[2 * j + 1] = si;
    }
}

/* verify one n1fv leaf on the production geometry: Bp instances, element
 * (k, j) at buffer index k + j*Bp (doubles), in-place. */
static int probe_n1(int r, weld_n1_fn fn) {
    const int Bp = 4; /* even vl, exercises the vector path */
    double buf[2 * WELD_MAX_RADIX * Bp];
    double ref[2 * WELD_MAX_RADIX * Bp];
    double keep[2 * WELD_MAX_RADIX * Bp];
    INT is_arr[WELD_MAX_RADIX], os_arr[WELD_MAX_RADIX];
    unsigned long long st = 0x1234u ^ (unsigned)r;
    for (int i = 0; i < 2 * r * Bp; ++i) {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        buf[i] = (double)(int)(st >> 33) / 2147483648.0 - 1.0;
    }
    memcpy(keep, buf, sizeof(buf));
    for (int j = 0; j < r; ++j) {
        is_arr[j] = (INT)(2 * j * Bp);
        os_arr[j] = (INT)(2 * j * Bp);
    }
    fn(buf, buf + 1, buf, buf + 1, is_arr, os_arr, (INT)Bp, 2, 2);
    for (int k = 0; k < Bp; ++k) {
        double inr[2 * WELD_MAX_RADIX], outr[2 * WELD_MAX_RADIX];
        for (int j = 0; j < r; ++j) {
            inr[2 * j] = keep[2 * (k + j * Bp)];
            inr[2 * j + 1] = keep[2 * (k + j * Bp) + 1];
        }
        ref_dft_r(inr, outr, r, 0, 0);
        for (int j = 0; j < r; ++j) {
            const double dr = buf[2 * (k + j * Bp)] - outr[2 * j];
            const double di = buf[2 * (k + j * Bp) + 1] - outr[2 * j + 1];
            if (fabs(dr) > 1e-12 || fabs(di) > 1e-12) {
                return -1;
            }
        }
    }
    return 1;
}

/* verify one t1fv leaf: Bp instances with twiddles W_{Bt}^{mk} from the
 * topic W generator (locks the slot order + sign convention together). */
static int probe_t1(int r, weld_t1_fn fn) {
    const int Bp = 4;
    const int Bt = Bp * r;
    double buf[2 * WELD_MAX_RADIX * Bp];
    double keep[2 * WELD_MAX_RADIX * Bp];
    INT rs_arr[WELD_MAX_RADIX];
    double *W = malloc(weld_t1_w_doubles(Bp, r) * sizeof(double));
    if (W == NULL) {
        return -1;
    }
    weld_t1_w_fill(W, Bp, r);
    unsigned long long st = 0x97531u ^ (unsigned)r;
    for (int i = 0; i < 2 * r * Bp; ++i) {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        buf[i] = (double)(int)(st >> 33) / 2147483648.0 - 1.0;
    }
    memcpy(keep, buf, sizeof(buf));
    for (int j = 0; j < r; ++j) {
        rs_arr[j] = (INT)(2 * j * Bp);
    }
    fn(buf, buf + 1, W, rs_arr, 0, (INT)Bp, 2);
    for (int k = 0; k < Bp; ++k) {
        double inr[2 * WELD_MAX_RADIX], outr[2 * WELD_MAX_RADIX];
        for (int j = 0; j < r; ++j) {
            inr[2 * j] = keep[2 * (k + j * Bp)];
            inr[2 * j + 1] = keep[2 * (k + j * Bp) + 1];
        }
        ref_dft_r(inr, outr, r, Bt, k);
        for (int j = 0; j < r; ++j) {
            const double dr = buf[2 * (k + j * Bp)] - outr[2 * j];
            const double di = buf[2 * (k + j * Bp) + 1] - outr[2 * j + 1];
            if (fabs(dr) > 1e-12 || fabs(di) > 1e-12) {
                free(W);
                return -1;
            }
        }
    }
    free(W);
    return 1;
}

void weld_codelets_init(void) {
    static int done = 0;
    if (done) {
        return;
    }
    done = 1;
    size_t off = 0;
#define WL(r)                                                                    \
    do {                                                                        \
        fftw_codelet_n1fv_##r##_avx2(NULL);                                     \
        g_n1[r] = g_cap_n1;                                                     \
        g_probe_n1[r] = probe_n1(r, g_n1[r]);                                   \
        off += (size_t)snprintf(g_report + off, sizeof(g_report) - off,         \
                                "%sn1fv_%d:%s", off ? " " : "", r,              \
                                g_probe_n1[r] == 1 ? "ok" : "FAIL");            \
    } while (0);
    WELD_N1_SET
#undef WL
#define WL(r)                                                                    \
    do {                                                                         \
        fftw_codelet_t1fv_##r##_avx2(NULL);                                      \
        g_t1[r] = g_cap_t1;                                                      \
        g_probe_t1[r] = probe_t1(r, g_t1[r]);                                    \
        off += (size_t)snprintf(g_report + off, sizeof(g_report) - off,           \
                                "%st1fv_%d:%s", off ? " " : "", r,               \
                                g_probe_t1[r] == 1 ? "ok" : "FAIL");             \
    } while (0);
    WELD_T1_SET
#undef WL
    /* topic jit-anyN-fwd-inv R1 (F3/L-S): the retired hand radix-11 pair
     * overrides the vendored n1fv_11 and re-opens the t1fv_11 slot ONLY
     * under FFT_ANYN_LEAVES=hand (R0 reproduction); same registration +
     * numeric probe discipline as the vendored set. */
    if (weld_hand11_requested()) {
        fftw_hand_codelet_n1fv_11_reg(NULL);
        g_n1[11] = g_cap_n1;
        g_probe_n1[11] = probe_n1(11, g_n1[11]);
        off += (size_t)snprintf(g_report + off, sizeof(g_report) - off,
                                "%sn1fv_11:%s(h)", off ? " " : "",
                                g_probe_n1[11] == 1 ? "ok" : "FAIL");
        fftw_hand_codelet_t1fv_11_reg(NULL);
        g_t1[11] = g_cap_t1;
        g_probe_t1[11] = probe_t1(11, g_t1[11]);
        off += (size_t)snprintf(g_report + off, sizeof(g_report) - off,
                                "%st1fv_11:%s(h)", off ? " " : "",
                                g_probe_t1[11] == 1 ? "ok" : "FAIL");
    }
}

weld_n1_fn weld_clet_n1(int r) {
    weld_codelets_init();
    if (r < 2 || r > WELD_MAX_RADIX || g_probe_n1[r] != 1) {
        return NULL;
    }
    return g_n1[r];
}

weld_t1_fn weld_clet_t1(int r) {
    weld_codelets_init();
    if (r < 2 || r > WELD_MAX_RADIX || g_probe_t1[r] != 1) {
        return NULL;
    }
    return g_t1[r];
}

const char *weld_codelets_probe_report(void) {
    weld_codelets_init();
    return g_report;
}

/* ------------------------------------------------------------------ */
/* t1fv W table */

size_t weld_t1_w_doubles(int Bp, int r) {
    return (size_t)(Bp / 2) * (size_t)(4 * (r - 1));
}

void weld_t1_w_fill(double *W, int Bp, int r) {
    const int Bt = Bp * r;
    const double sgn = +2.0 * M_PI / (double)Bt; /* slot = e^{+i*theta};
                                                  * BYTWJ conj consumes it */
    for (int p = 0; p < Bp / 2; ++p) {
        double *row = W + (size_t)p * (size_t)(4 * (r - 1));
        for (int j = 1; j < r; ++j) {
            double *slot = row + 4 * (j - 1);
            const double a0 = sgn * (double)(j * (2 * p));
            const double a1 = sgn * (double)(j * (2 * p + 1));
            slot[0] = cos(a0);
            slot[1] = sin(a0);
            slot[2] = cos(a1);
            slot[3] = sin(a1);
        }
    }
}

/* ------------------------------------------------------------------ */
/* the stage runner (JIT call target) */

#include <immintrin.h>

static void run_n1_block(const weld_stage_desc *d, double *inb, double *outb,
                         long vl, long ivs, long ovs) {
    /* codelets only read the precomputed stride index arrays */
    d->n1(inb, inb + 1, outb, outb + 1, (stride)d->is, (stride)d->os,
          (INT)vl, (INT)ivs, (INT)ovs);
}

void weld_run_stage(const weld_stage_desc *d, double *base, double *base2) {
    switch (d->kind) {
    case WELD_RUN_T1: {
        /* per block: t1fv(ri=block, W, rs, 0, me=Bp, ms=2) — twiddles are
         * applied inside the leaf (no separate twiddle sweep). */
        for (long b = 0; b < d->nblocks; ++b) {
            double *blk = base + b * d->bstep;
            d->t1(blk, blk + 1, d->W, (stride)d->rs, 0, (INT)d->vl, 2);
        }
        break;
    }
    case WELD_RUN_N1: {
        double *out = (base2 != NULL) ? base2 : base;
        for (long b = 0; b < d->nblocks; ++b) {
            double *bin = base + b * d->bstep;
            double *bo = out + b * d->bstep;
            run_n1_block(d, bin, bo, d->vl, d->ivs, d->ovs);
        }
        break;
    }
    case WELD_RUN_N1_ODD: {
        /* main: instances k = 0..Bp-2 (even count, exact loop); the tail
         * instance k = Bp-1 of every block in one cross-block call
         * (instance stride = block stride).  If the block count is odd the
         * tail call's padding lane writes (r-1)*Bp elements past the end —
         * scratch slack owns that span. */
        double *out = (base2 != NULL) ? base2 : base;
        const long vlMain = d->vl - 1;
        for (long b = 0; b < d->nblocks; ++b) {
            double *bin = base + b * d->bstep;
            double *bo = out + b * d->bstep;
            run_n1_block(d, bin, bo, vlMain, d->ivs, d->ovs);
        }
        /* tail instances (block b, k = Bp-1) are bstep apart — NOT
         * adjacent, so the fv vector dimension cannot span them; each
         * gets a vl=1 call (ivs = 2: the padding lane loads a harmless
         * neighbour; ovs steers its stores into slack). */
        for (long b = 0; b < d->nblocks; ++b) {
            double *tin = base + b * d->bstep + d->base_off;
            double *tou = out + b * d->bstep + d->base_off;
            const long slackOvs = (long)(d->slack - tou);
            d->n1(tin, tin + 1, tou, tou + 1, (stride)d->is, (stride)d->os,
                  1, 2, slackOvs);
        }
        break;
    }
    case WELD_RUN_N1_CONTIG: {
        /* stage 0 on the digitrev-permuted scratch: r-element runs, vl
         * instances at stride 2r.  Odd vl: main vl-1 + tail call whose
         * padding lane lands in slack. */
        double *out = (base2 != NULL) ? base2 : base;
        const long n = d->vl;
        if ((n & 1) == 0) {
            run_n1_block(d, base, out, n, d->ivs, d->ovs);
        } else {
            if (n > 1) {
                run_n1_block(d, base, out, n - 1, d->ivs, d->ovs);
            }
            /* tail: last instance; padding lane's stores at
             * ro + ovs + j*os — steer with a slack-relative ovs. */
            double *ro = out + (n - 1) * d->ivs;
            const long slackOvs =
                (long)(d->slack - ro); /* padding row base = slack */
            d->n1(ro, ro + 1, ro, ro + 1, (stride)d->is, (stride)d->os, 1,
                  d->ivs, slackOvs);
        }
        break;
    }
    case WELD_RUN_TWS_V2: {
        /* twiddle pre-sweep for an n1fv stage (even Bp): for j = 1..r-1,
         * every block's contiguous k run as 32B pairs multiplied by TW
         * pair slots.  TW layout: slot(j, kp) at W + ((j-1)*twPairs +
         * kp) * 8 doubles = [W^<jk>, W^<j(k+1)> | swap twin]. */
        const int r = d->radix;
        const int pairs = d->twPairs;
        for (int j = 1; j < r; ++j) {
            const double *rowj = d->W + (size_t)(j - 1) * pairs * 8;
            for (long m = 0; m < d->nblocks; ++m) {
                double *p = base + m * d->bstep + (long)j * 2 * d->bp;
                const double *tw = rowj;
                for (int kp = 0; kp < pairs; ++kp) {
                    __m256d v = _mm256_loadu_pd(p);
                    __m256d re = _mm256_movedup_pd(v);
                    __m256d im = _mm256_permute_pd(v, 0xF);
                    __m256d w0 = _mm256_loadu_pd(tw);
                    __m256d w1 = _mm256_loadu_pd(tw + 4);
                    __m256d res = _mm256_fmaddsub_pd(
                        re, w0, _mm256_mul_pd(im, w1));
                    _mm256_storeu_pd(p, res);
                    p += 4;
                    tw += 8;
                }
            }
        }
        break;
    }
    case WELD_RUN_TWS_SC: {
        /* twiddle pre-sweep, odd Bp: 16B elements at 32B-slot TW rows
         * (single-k layout: (wr, wi, wi, wr)); k = 0 needs no multiply. */
        const int r = d->radix;
        const long Bp = d->bp;
        for (int j = 1; j < r; ++j) {
            const double *rowj = d->W + (size_t)(j - 1) * Bp * 4;
            for (long m = 0; m < d->nblocks; ++m) {
                double *p = base + m * d->bstep + (long)j * 2 * Bp;
                /* k = 0 pairs with slot 0 = W^{j*0} = 1 (identity, kept
                 * for pairing simplicity — same fold as the v2 sweep) */
                const double *tw = rowj;
                for (long k = 0; k < Bp; ++k) {
                    __m128d v = _mm_loadu_pd(p);
                    __m128d re = _mm_movedup_pd(v); /* (xr, xr) */
                    __m128d im = _mm_shuffle_pd(v, v, 3); /* (xi, xi) */
                    __m128d w = _mm_loadu_pd(tw);
                    __m128d wi = _mm_permute_pd(w, 1);
                    __m128d res =
                        _mm_fmaddsub_pd(re, w, _mm_mul_pd(im, wi));
                    _mm_storeu_pd(p, res);
                    p += 2;
                    tw += 4;
                }
            }
        }
        break;
    }
    case WELD_RUN_PERMUTE: {
        /* tab[i] = BYTE offset (16 per complex) -> double index = >>3.
         * topic jit-anyN-fwd-inv: d->conj flips the imaginary sign while
         * copying — the inverse entry boundary conj sunk into the
         * stage-0 permutation (no separate sweep). */
        for (long i = 0; i < d->n; ++i) {
            const double *src = base + (long)(d->tab[i] >> 3);
            base2[2 * i] = src[0];
            base2[2 * i + 1] = d->conj ? -src[1] : src[1];
        }
        break;
    }
    case WELD_RUN_CONJMUL: {
        /* io <- conj(io .* H): complex mult on natural pairs + imag sign
         * flip; hMul/hSwap are natural 32B H pairs + swapped twins. */
        const __m256d sgn =
            _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
        for (long i = 0; i < d->n / 2; ++i) {
            __m256d v = _mm256_loadu_pd(base + 4 * i);
            __m256d re = _mm256_movedup_pd(v);
            __m256d im = _mm256_permute_pd(v, 0xF);
            __m256d h0 = _mm256_loadu_pd(d->hMul + 4 * i);
            __m256d h1 = _mm256_loadu_pd(d->hSwap + 4 * i);
            __m256d res =
                _mm256_fmaddsub_pd(re, h0, _mm256_mul_pd(im, h1));
            _mm256_storeu_pd(base + 4 * i, _mm256_xor_pd(res, sgn));
        }
        if (d->n & 1) { /* odd tail element */
            __m128d v = _mm_loadu_pd(base + 2 * (d->n - 1));
            __m128d re = _mm_movedup_pd(v);
            __m128d im = _mm_shuffle_pd(v, v, 3); /* (xi, xi) */
            __m128d h0 = _mm_loadu_pd(d->hMul + 2 * (d->n - 1));
            __m128d h1 = _mm_permute_pd(h0, 1); /* (hi, hr) swap twin */
            __m128d res = _mm_fmaddsub_pd(re, h0, _mm_mul_pd(im, h1));
            _mm_storeu_pd(base + 2 * (d->n - 1),
                          _mm_xor_pd(res, _mm_setr_pd(0.0, -0.0)));
        }
        break;
    }
    case WELD_RUN_INVN: {
        /* io <- conj(io) * invn == io * (invn, -invn) broadcast. */
        const __m256d s = _mm256_setr_pd(d->invn, -d->invn, d->invn,
                                         -d->invn);
        for (long i = 0; i < d->n / 2; ++i) {
            __m256d v = _mm256_loadu_pd(base + 4 * i);
            _mm256_storeu_pd(base + 4 * i, _mm256_mul_pd(v, s));
        }
        if (d->n & 1) {
            __m128d v = _mm_loadu_pd(base + 2 * (d->n - 1));
            _mm_storeu_pd(base + 2 * (d->n - 1),
                          _mm_mul_pd(v, _mm_setr_pd(d->invn, -d->invn)));
        }
        break;
    }
    default:
        break;
    }
}
