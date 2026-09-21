/*
 * tests/test_linear_conv.c — opt-in block overlap-save LINEAR convolution vs
 * direct time-domain convolution + unpatched three-seg FFTW (draft
 * API-FFT-006).  Promotes: jit-linear-conv-ola.  Clauses: API-FFT-006,
 * BEH-FFT-007.
 *
 * Scale note (HOW vs oracle):
 *   Product path implements the IFFT conjugate trick including /N
 *   (normalized).  The three-seg FFTW oracle is unnormalized (FFTW
 *   BACKWARD), so its result is L× the linear convolution (L =
 *   next_pow2(nx+nh-1)).  PASS residual = blocked vs oracle / L.
 *
 * Blocked OLS assembly: prepend nh-1 zeros to x, process length-N blocks
 * with hop = valid = N-nh+1; each fft_execute returns the valid samples
 * into the front of the buffer.  Concatenating block outputs reconstructs
 * the full linear convolution y = x * h (length nx+nh-1), including all
 * cross-block overlap boundary samples.
 *
 * FFTW MEASURE clobbers arrays during planning: plan first, then
 * regenerate the deterministic input (h and x) before execute / compare.
 *
 *   - linear create returns NULL on N=0/-1/3/65536, nh=0, nh>N, and NULL h
 *   - blocked OLS vs direct O(nx·nh) convolution and vs three-seg FFTW/L
 *     over representative (N, nh) incl. nh=N=256 degenerate
 */

#include "fft.h"
#include "test_common.h"

#include <fftw3.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Representative (N, nh) sizes, incl. the nh=N degenerate block. */
typedef struct {
    int N;
    int nh;
} case_t;

static const case_t REP_CASES[] = {
    {1024, 64},   /* representative (must) */
    {256, 16},
    {256, 256},   /* nh = N degenerate */
    {4096, 16},
    {4096, 256},
};
#define REP_COUNT ((int)(sizeof(REP_CASES) / sizeof(REP_CASES[0])))

/* Long x length = 4*N (>= 2 blocks always). */
#define X_FACTOR 4L

static void fill_lcg(float complex *io, long n, uint32_t seed) {
    uint32_t rng = seed;
    for (long i = 0; i < n; i++) {
        io[i] = rand_complex(&rng);
    }
}

static int next_pow2(long n) {
    int p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

/* PASS residual: every element satisfies |got-want| <= rel*scale + abs
 * (test_common.h close_complex), scale = max_mag(want).  Also reports the
 * max absolute error across the array for the PASS/FAIL message. */
static int close_all(const float complex *got, const float complex *want,
                     long n, float *max_abs_out) {
    float scale = max_mag(want, (int)n);
    float ma = 0.0f;
    int ok = 1;
    for (long i = 0; i < n; i++) {
        float d = cabsf(got[i] - want[i]);
        if (d > ma) {
            ma = d;
        }
        if (!close_complex(got[i], want[i], scale)) {
            ok = 0;
        }
    }
    *max_abs_out = ma;
    return ok;
}

/* Direct O(nx·nh) time-domain linear convolution y = x * h. */
static void direct_conv(const float complex *x, long nx,
                        const float complex *h, int nh, float complex *out) {
    const long ylen = nx + nh - 1;
    for (long m = 0; m < ylen; m++) {
        float complex acc = 0.0f + 0.0f * I;
        long j0 = (m >= nh) ? (m - nh + 1) : 0;
        long j1 = (m < nx) ? m : (nx - 1);
        for (long j = j0; j <= j1; j++) {
            acc += x[j] * h[m - j];
        }
        out[m] = acc;
    }
}

/* Blocked overlap-save (OLS): prepend nh-1 zeros, process length-N blocks
 * with hop = valid = N-nh+1, concatenate valid outputs into `out`. */
static void blocked_ols(fft_plan *plan, int N, int nh, const float complex *xp,
                        long ylen, float complex *out) {
    const int valid = N - nh + 1;
    float complex *blk = (float complex *)malloc((size_t)N * sizeof(*blk));
    long nblocks = (ylen + valid - 1) / valid;
    for (long b = 0; b < nblocks; b++) {
        long base = b * valid;
        for (int i = 0; i < N; i++) {
            long idx = base + i;
            blk[i] = (idx < ylen) ? xp[idx] : (0.0f + 0.0f * I);
        }
        fft_execute(plan, blk);
        for (int i = 0; i < valid; i++) {
            long oidx = base + i;
            if (oidx < ylen) {
                out[oidx] = blk[i];
            }
        }
    }
    free(blk);
}

static int check_nulls(void) {
    float complex h[8];
    for (int i = 0; i < 8; i++) {
        h[i] = 1.0f + 0.0f * I;
    }
    static const int badN[] = {0, -1, 3, 65536};
    for (int i = 0; i < (int)(sizeof(badN) / sizeof(badN[0])); i++) {
        if (fft_plan_create_linear_conv(badN[i], h, 1) != NULL) {
            fprintf(stderr, "linear create(N=%d) must be NULL\n", badN[i]);
            return 1;
        }
    }
    if (fft_plan_create_linear_conv(1024, h, 0) != NULL) {
        fprintf(stderr, "linear create(nh=0) must be NULL\n");
        return 1;
    }
    if (fft_plan_create_linear_conv(1024, h, 1025) != NULL) {
        fprintf(stderr, "linear create(nh=1025>N=1024) must be NULL\n");
        return 1;
    }
    if (fft_plan_create_linear_conv(1024, NULL, 1) != NULL) {
        fprintf(stderr, "linear create(h=NULL) must be NULL\n");
        return 1;
    }
    printf("linear create NULL on N=0/-1/3/65536 + nh=0 + nh>N + NULL h: PASS\n");
    return 0;
}

/* Blocked OLS vs direct (leg a) and vs three-seg FFTW/L (leg b),
 * full-length (nx+nh-1) including cross-block overlap boundaries. */
static int check_linear_nh(int N, int nh) {
    const long nx = X_FACTOR * (long)N;
    const long ylen = nx + (long)nh - 1;
    const int valid = N - nh + 1;
    const int L = next_pow2(ylen);
    const float invL = 1.0f / (float)L;
    const uint32_t hseed = 0x9e3779b9u ^ (uint32_t)N ^ (uint32_t)nh;
    const uint32_t xseed = 0x5eed1234u ^ (uint32_t)N ^ (uint32_t)nh;

    float complex *h = (float complex *)malloc((size_t)N * sizeof(*h));
    float complex *x = (float complex *)malloc((size_t)nx * sizeof(*x));
    float complex *xp = (float complex *)malloc((size_t)ylen * sizeof(*xp));
    float complex *y_direct =
        (float complex *)malloc((size_t)ylen * sizeof(*y_direct));
    float complex *y_blocked =
        (float complex *)malloc((size_t)ylen * sizeof(*y_blocked));
    float complex *xf = (float complex *)fftwf_malloc((size_t)L * sizeof(*xf));
    float complex *hf = (float complex *)fftwf_malloc((size_t)L * sizeof(*hf));
    float complex *Hfft = (float complex *)malloc((size_t)L * sizeof(*Hfft));
    if (h == NULL || x == NULL || xp == NULL || y_direct == NULL ||
        y_blocked == NULL || xf == NULL || hf == NULL || Hfft == NULL) {
        fprintf(stderr, "N=%d nh=%d OOM\n", N, nh);
        free(h);
        free(x);
        free(xp);
        free(y_direct);
        free(y_blocked);
        fftwf_free(xf);
        fftwf_free(hf);
        free(Hfft);
        return 1;
    }

    /* Plan FFTW first (MEASURE clobbers xf/hf). */
    fftwf_plan fwd = fftwf_plan_dft_1d(L, (fftwf_complex *)xf,
                                       (fftwf_complex *)xf, FFTW_FORWARD,
                                       FFTW_MEASURE);
    fftwf_plan bwd = fftwf_plan_dft_1d(L, (fftwf_complex *)xf,
                                       (fftwf_complex *)xf, FFTW_BACKWARD,
                                       FFTW_MEASURE);
    if (fwd == NULL || bwd == NULL) {
        fprintf(stderr, "N=%d nh=%d fftwf plan (L=%d) failed\n", N, nh, L);
        if (fwd != NULL) {
            fftwf_destroy_plan(fwd);
        }
        if (bwd != NULL) {
            fftwf_destroy_plan(bwd);
        }
        free(h);
        free(x);
        free(xp);
        free(y_direct);
        free(y_blocked);
        fftwf_free(xf);
        fftwf_free(hf);
        free(Hfft);
        return 1;
    }

    /* Regenerate deterministic inputs after MEASURE clobber. */
    fill_lcg(h, nh, hseed);
    for (int i = nh; i < N; i++) {
        h[i] = 0.0f + 0.0f * I;
    }
    fill_lcg(x, nx, xseed);

    /* Padded input: xp = [0^{nh-1}, x] (length ylen). */
    for (long i = 0; i < ylen; i++) {
        xp[i] = 0.0f + 0.0f * I;
    }
    for (long i = 0; i < nx; i++) {
        xp[(long)(nh - 1) + i] = x[i];
    }

    /* Leg (a) direct oracle. */
    direct_conv(x, nx, h, nh, y_direct);

    /* Leg (b) three-seg FFTW oracle: Hfft = FFT(h→L); xf forward × H → back. */
    for (int i = 0; i < L; i++) {
        hf[i] = (i < nh) ? h[i] : (0.0f + 0.0f * I);
    }
    fftwf_execute_dft(fwd, (fftwf_complex *)hf, (fftwf_complex *)hf);
    memcpy(Hfft, hf, (size_t)L * sizeof(*Hfft));
    for (int i = 0; i < L; i++) {
        xf[i] = (i < nx) ? x[i] : (0.0f + 0.0f * I);
    }
    fftwf_execute_dft(fwd, (fftwf_complex *)xf, (fftwf_complex *)xf);
    for (int i = 0; i < L; i++) {
        xf[i] = xf[i] * Hfft[i];
    }
    fftwf_execute_dft(bwd, (fftwf_complex *)xf, (fftwf_complex *)xf);
    for (int i = 0; i < L; i++) {
        xf[i] = xf[i] * invL; /* unnormalized BACKWARD → /L */
    }

    /* Blocked OLS via this repo's opt-in plan. */
    fft_plan *plan = fft_plan_create_linear_conv(N, h, nh);
    if (plan == NULL) {
        fprintf(stderr, "N=%d nh=%d fft_plan_create_linear_conv failed\n", N,
                nh);
        fftwf_destroy_plan(fwd);
        fftwf_destroy_plan(bwd);
        free(h);
        free(x);
        free(xp);
        free(y_direct);
        free(y_blocked);
        fftwf_free(xf);
        fftwf_free(hf);
        free(Hfft);
        return 1;
    }
    blocked_ols(plan, N, nh, xp, ylen, y_blocked);
    fft_plan_destroy(plan);

    float bd_abs = 0.0f, bf_abs = 0.0f;
    int pass_bd = close_all(y_blocked, y_direct, ylen, &bd_abs);
    int pass_bf = close_all(y_blocked, xf, ylen, &bf_abs);

    printf("N=%d nh=%d valid=%d blocked-vs-direct max_abs=%.6e%s\n", N, nh,
           valid, (double)bd_abs, pass_bd ? " PASS" : " FAIL");
    printf("N=%d nh=%d valid=%d blocked-vs-fftw  max_abs=%.6e%s\n", N, nh,
           valid, (double)bf_abs, pass_bf ? " PASS" : " FAIL");

    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(bwd);
    free(h);
    free(x);
    free(xp);
    free(y_direct);
    free(y_blocked);
    fftwf_free(xf);
    fftwf_free(hf);
    free(Hfft);
    return (pass_bd && pass_bf) ? 0 : 1;
}

int main(void) {
    int rc = check_nulls();
    for (int i = 0; i < REP_COUNT; i++) {
        rc |= check_linear_nh(REP_CASES[i].N, REP_CASES[i].nh);
    }
    if (rc != 0) {
        fprintf(stderr, "test_linear_conv: FAILED\n");
        return 1;
    }
    printf("test_linear_conv: PASS (representative (N,nh) + NULL checks)\n");
    return 0;
}
