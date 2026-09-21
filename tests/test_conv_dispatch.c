/*
 * tests/test_conv_dispatch.c — opt-in whole-sequence LINEAR convolution with
 * runtime size dispatch vs direct O(nx·nh) time-domain oracle + manual OLS
 * (draft API-FFT-007).  Promotes: jit-conv-dispatch.  Clauses: API-FFT-007,
 * BEH-FFT-008.
 *
 * The dispatch plan welds y = x * h (x length nx, h length nh, output length
 * nx+nh-1) in ONE fft_execute.  Plan-time leg selection is by the
 * record-only threshold FFT_CONV_DISPATCH_NH_DIRECT_MAX: nh <= threshold ->
 * direct scalar leg (correctness oracle), otherwise -> OLS block leg reusing
 * fft_plan_create_linear_conv(N_block, h, nh) with N_block =
 * next_pow2(2*nh-1).
 *
 *   - dispatch create returns NULL on nx<=0 / nh<=0 / nh>nx / NULL h
 *   - dispatch vs direct and vs manual OLS over representative (nx, nh),
 *     full-length ylen = nx+nh-1 including overlap boundaries
 *   - fft_plan_conv_dispatch_leg deterministic leg selection
 *     (nh <= threshold -> 0/direct, otherwise 1/OLS)
 *   - regression: default forward / gauss / inverse / conv unchanged at N=256
 *
 * FFTW MEASURE clobbers arrays during planning: the regression three-seg
 * FFTW oracle plans first, then regenerates deterministic input before
 * execute / compare (matching test_conv.c).  The direct/OLS dispatch legs
 * carry no FFTW_MEASURE state, so no such clobbering applies to them.
 */

#include "fft.h"
#include "fft_jit.h"
#include "test_common.h"

#include <fftw3.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* POC-local introspection; NOT part of the public ABI (stays out of fft.h). */
extern int fft_plan_conv_dispatch_leg(const fft_plan *plan);

typedef struct {
    int nx;
    int nh;
} case_t;

static const case_t REP_CASES[] = {
    {64, 2},      /* direct */
    {128, 8},     /* direct (threshold boundary nh == NH_DIRECT_MAX) */
    {128, 9},     /* OLS (just above threshold) */
    {256, 64},    /* OLS */
    {256, 256},   /* OLS (nh == nx degenerate) */
    {4096, 1024}, /* OLS (nh >= 1024 band) */
};
#define REP_COUNT ((int)(sizeof(REP_CASES) / sizeof(REP_CASES[0])))

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
 * max absolute error for the PASS/FAIL message. */
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

/* Direct O(nx·nh) time-domain linear convolution y = x * h.  The oracle. */
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

/* Blocked overlap-save (OLS): padded input xp = [0^{nh-1}, x] of length
 * ylen, processed in length-N blocks with hop = valid = N-nh+1, valid
 * samples concatenated into `out`.  Mirrors the dispatch OLS leg. */
static void blocked_ols(fft_plan *plan, int N, int nh, const float complex *xp,
                        long ylen, float complex *out) {
    const int valid = N - nh + 1;
    float complex *blk = (float complex *)malloc((size_t)N * sizeof(*blk));
    long nblocks = (ylen + valid - 1) / valid;
    for (long b = 0; b < nblocks; b++) {
        long base = b * (long)valid;
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
    if (fft_plan_create_conv_dispatch(0, h, 1) != NULL) {
        fprintf(stderr, "dispatch create(nx=0) must be NULL\n");
        return 1;
    }
    if (fft_plan_create_conv_dispatch(-1, h, 1) != NULL) {
        fprintf(stderr, "dispatch create(nx=-1) must be NULL\n");
        return 1;
    }
    if (fft_plan_create_conv_dispatch(64, h, 0) != NULL) {
        fprintf(stderr, "dispatch create(nh=0) must be NULL\n");
        return 1;
    }
    if (fft_plan_create_conv_dispatch(64, h, -1) != NULL) {
        fprintf(stderr, "dispatch create(nh=-1) must be NULL\n");
        return 1;
    }
    if (fft_plan_create_conv_dispatch(64, h, 65) != NULL) {
        fprintf(stderr, "dispatch create(nh=65>nx=64) must be NULL\n");
        return 1;
    }
    if (fft_plan_create_conv_dispatch(64, NULL, 1) != NULL) {
        fprintf(stderr, "dispatch create(h=NULL) must be NULL\n");
        return 1;
    }
    printf("dispatch create NULL on nx<=0 / nh<=0 / nh>nx / NULL h: PASS\n");
    return 0;
}

/* Correctness for one (nx, nh): dispatch vs direct oracle AND vs manual OLS,
 * plus deterministic leg selection.  Full-length (nx+nh-1) coverage. */
static int check_conv_nh(int nx, int nh) {
    const long ylen = (long)nx + (long)nh - 1;
    const int N_block = next_pow2(2 * nh - 1);
    const uint32_t hseed = 0x9e3779b9u ^ (uint32_t)nx ^ (uint32_t)nh;
    const uint32_t xseed = 0x5eed1234u ^ (uint32_t)nx ^ (uint32_t)nh;

    float complex *h = (float complex *)malloc((size_t)nh * sizeof(*h));
    float complex *x = (float complex *)malloc((size_t)nx * sizeof(*x));
    float complex *xp = (float complex *)malloc((size_t)ylen * sizeof(*xp));
    float complex *y_direct =
        (float complex *)malloc((size_t)ylen * sizeof(*y_direct));
    float complex *y_ols =
        (float complex *)malloc((size_t)ylen * sizeof(*y_ols));
    float complex *y_disp =
        (float complex *)malloc((size_t)ylen * sizeof(*y_disp));
    if (h == NULL || x == NULL || xp == NULL || y_direct == NULL ||
        y_ols == NULL || y_disp == NULL) {
        fprintf(stderr, "nx=%d nh=%d OOM\n", nx, nh);
        free(h);
        free(x);
        free(xp);
        free(y_direct);
        free(y_ols);
        free(y_disp);
        return 1;
    }

    fill_lcg(h, nh, hseed);
    fill_lcg(x, nx, xseed);

    /* Leg (1) direct oracle. */
    direct_conv(x, nx, h, nh, y_direct);

    /* Leg (2) manual OLS reference (reuses promoted linear_conv). */
    fft_plan *plan_ols = fft_plan_create_linear_conv(N_block, h, nh);
    if (plan_ols == NULL) {
        fprintf(stderr, "nx=%d nh=%d linear_conv(N_block=%d) create failed\n",
                nx, nh, N_block);
        free(h);
        free(x);
        free(xp);
        free(y_direct);
        free(y_ols);
        free(y_disp);
        return 1;
    }
    for (long i = 0; i < ylen; i++) {
        xp[i] = 0.0f + 0.0f * I;
    }
    for (long i = 0; i < nx; i++) {
        xp[(long)(nh - 1) + i] = x[i];
    }
    blocked_ols(plan_ols, N_block, nh, xp, ylen, y_ols);
    fft_plan_destroy(plan_ols);

    /* Leg (3) dispatch: plan-time leg selection + one fft_execute. */
    fft_plan *plan_disp = fft_plan_create_conv_dispatch(nx, h, nh);
    if (plan_disp == NULL) {
        fprintf(stderr, "nx=%d nh=%d dispatch create failed\n", nx, nh);
        free(h);
        free(x);
        free(xp);
        free(y_direct);
        free(y_ols);
        free(y_disp);
        return 1;
    }
    int leg = fft_plan_conv_dispatch_leg(plan_disp);
    int expected = (nh <= FFT_CONV_DISPATCH_NH_DIRECT_MAX) ? 0 : 1;
    for (long i = 0; i < nx; i++) {
        y_disp[i] = x[i];
    }
    for (long i = nx; i < ylen; i++) {
        y_disp[i] = 0.0f + 0.0f * I;
    }
    fft_execute(plan_disp, y_disp);
    fft_plan_destroy(plan_disp);

    float dd_abs = 0.0f, od_abs = 0.0f, do_abs = 0.0f;
    int pass_dd = close_all(y_disp, y_direct, ylen, &dd_abs);
    int pass_od = close_all(y_ols, y_direct, ylen, &od_abs);
    int pass_do = close_all(y_disp, y_ols, ylen, &do_abs);
    int pass_leg = (leg == expected);

    printf("nx=%d nh=%d N_block=%d leg=%s expected=%s%s\n", nx, nh, N_block,
           (leg == 0) ? "direct" : "OLS",
           (expected == 0) ? "direct" : "OLS", pass_leg ? " PASS" : " FAIL");
    printf("nx=%d nh=%d dispatch-vs-direct max_abs=%.6e%s\n", nx, nh,
           (double)dd_abs, pass_dd ? " PASS" : " FAIL");
    printf("nx=%d nh=%d ols-vs-direct      max_abs=%.6e%s\n", nx, nh,
           (double)od_abs, pass_od ? " PASS" : " FAIL");
    printf("nx=%d nh=%d dispatch-vs-ols    max_abs=%.6e%s\n", nx, nh,
           (double)do_abs, pass_do ? " PASS" : " FAIL");

    free(h);
    free(x);
    free(xp);
    free(y_direct);
    free(y_ols);
    free(y_disp);
    return (pass_dd && pass_od && pass_do && pass_leg) ? 0 : 1;
}

/* Regression sanity: default forward / gauss / inverse / conv semantics
 * unchanged at a representative N=256 (mirrors bench_conv_dispatch
 * check_regression).  linear-conv is exercised by the OLS leg above. */
static int check_regression(void) {
    const int N = 256;
    const float invN = 1.0f / (float)N;
    int rc = 0;

    float complex *src = (float complex *)malloc((size_t)N * sizeof(*src));
    float complex *got = (float complex *)malloc((size_t)N * sizeof(*got));
    float complex *ref = (float complex *)malloc((size_t)N * sizeof(*ref));
    float complex *up = (float complex *)fftwf_malloc((size_t)N * sizeof(*up));
    float complex *H = (float complex *)malloc((size_t)N * sizeof(*H));
    float complex *h = (float complex *)malloc((size_t)N * sizeof(*h));
    float *G = (float *)malloc((size_t)N * sizeof(*G));
    if (src == NULL || got == NULL || ref == NULL || up == NULL || H == NULL ||
        h == NULL || G == NULL) {
        fprintf(stderr, "regression N=%d OOM\n", N);
        free(src);
        free(got);
        free(ref);
        fftwf_free(up);
        free(H);
        free(h);
        free(G);
        return 1;
    }

    /* Plan FFTW first: MEASURE clobbers `up`.  Then regenerate input. */
    fftwf_plan fwd = fftwf_plan_dft_1d(N, (fftwf_complex *)up,
                                       (fftwf_complex *)up, FFTW_FORWARD,
                                       FFTW_MEASURE);
    fftwf_plan bwd = fftwf_plan_dft_1d(N, (fftwf_complex *)up,
                                       (fftwf_complex *)up, FFTW_BACKWARD,
                                       FFTW_MEASURE);
    if (fwd == NULL || bwd == NULL) {
        fprintf(stderr, "regression N=%d fftwf plan failed\n", N);
        if (fwd != NULL) {
            fftwf_destroy_plan(fwd);
        }
        if (bwd != NULL) {
            fftwf_destroy_plan(bwd);
        }
        free(src);
        free(got);
        free(ref);
        fftwf_free(up);
        free(H);
        free(h);
        free(G);
        return 1;
    }

    fft_plan *def = fft_plan_create(N);
    fft_plan *inv = fft_plan_create_inverse(N);
    fft_plan *gauss = fft_plan_create_gauss_window(N);
    fill_lcg(h, N, 0x9e3779b9u);
    fft_plan *conv = fft_plan_create_conv(N, h);
    if (def == NULL || inv == NULL || gauss == NULL || conv == NULL) {
        fprintf(stderr, "regression N=%d plan create failed\n", N);
        fft_plan_destroy(def);
        fft_plan_destroy(inv);
        fft_plan_destroy(gauss);
        fft_plan_destroy(conv);
        fftwf_destroy_plan(fwd);
        fftwf_destroy_plan(bwd);
        free(src);
        free(got);
        free(ref);
        fftwf_free(up);
        free(H);
        free(h);
        free(G);
        return 1;
    }
    if (def == inv || def == gauss || def == conv || inv == gauss ||
        inv == conv || gauss == conv) {
        fprintf(stderr, "regression N=%d cache collision among plans\n", N);
        rc = 1;
    }

    float a = 0.0f;

    /* Default forward vs FFTW forward. */
    fill_lcg(src, N, 0xC0FFEEu);
    memcpy(got, src, (size_t)N * sizeof(*got));
    memcpy(up, src, (size_t)N * sizeof(*up));
    fft_execute(def, got);
    fftwf_execute_dft(fwd, (fftwf_complex *)up, (fftwf_complex *)up);
    int p = close_all(got, up, N, &a);
    printf("regression N=%d default-forward-vs-fftw max_abs=%.6e%s\n", N,
           (double)a, p ? " PASS" : " FAIL");
    rc |= (p ? 0 : 1);

    /* Inverse vs FFTW backward / N. */
    fill_lcg(src, N, 0xC0FFEEu ^ 1u);
    memcpy(got, src, (size_t)N * sizeof(*got));
    memcpy(up, src, (size_t)N * sizeof(*up));
    fft_execute(inv, got);
    fftwf_execute_dft(bwd, (fftwf_complex *)up, (fftwf_complex *)up);
    for (int i = 0; i < N; i++) {
        ref[i] = up[i] * invN;
    }
    p = close_all(got, ref, N, &a);
    printf("regression N=%d inverse-vs-fftw-backward/N max_abs=%.6e%s\n", N,
           (double)a, p ? " PASS" : " FAIL");
    rc |= (p ? 0 : 1);

    /* Gauss fused vs two-pass (window then default forward). */
    if (fft_gauss_window_fill(N, G) != 0) {
        fprintf(stderr, "regression N=%d fft_gauss_window_fill failed\n", N);
        rc = 1;
    } else {
        fill_lcg(src, N, 0xC0FFEEu ^ 2u);
        memcpy(got, src, (size_t)N * sizeof(*got));
        memcpy(ref, src, (size_t)N * sizeof(*ref));
        fft_execute(gauss, got);
        for (int i = 0; i < N; i++) {
            ref[i] *= G[i];
        }
        fft_execute(def, ref);
        p = close_all(got, ref, N, &a);
        printf("regression N=%d gauss-fused-vs-twopass max_abs=%.6e%s\n", N,
               (double)a, p ? " PASS" : " FAIL");
        rc |= (p ? 0 : 1);
    }

    /* Conv vs FFTW three-seg (forward + x FFT(h) + backward) / N. */
    memcpy(up, h, (size_t)N * sizeof(*up));
    fftwf_execute_dft(fwd, (fftwf_complex *)up, (fftwf_complex *)up);
    memcpy(H, up, (size_t)N * sizeof(*H));
    fill_lcg(src, N, 0xC0FFEEu ^ 3u);
    memcpy(got, src, (size_t)N * sizeof(*got));
    memcpy(up, src, (size_t)N * sizeof(*up));
    fft_execute(conv, got);
    fftwf_execute_dft(fwd, (fftwf_complex *)up, (fftwf_complex *)up);
    for (int i = 0; i < N; i++) {
        up[i] = up[i] * H[i];
    }
    fftwf_execute_dft(bwd, (fftwf_complex *)up, (fftwf_complex *)up);
    for (int i = 0; i < N; i++) {
        ref[i] = up[i] * invN;
    }
    p = close_all(got, ref, N, &a);
    printf("regression N=%d conv-vs-threeseg/N max_abs=%.6e%s\n", N,
           (double)a, p ? " PASS" : " FAIL");
    rc |= (p ? 0 : 1);

    fft_plan_destroy(def);
    fft_plan_destroy(inv);
    fft_plan_destroy(gauss);
    fft_plan_destroy(conv);
    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(bwd);
    free(src);
    free(got);
    free(ref);
    fftwf_free(up);
    free(H);
    free(h);
    free(G);
    return rc;
}

int main(void) {
    int rc = check_nulls();
    printf("--- dispatch linear conv: leg selection + dispatch/OLS vs direct ---\n");
    printf("# threshold FFT_CONV_DISPATCH_NH_DIRECT_MAX=%d (record-only, not a must SLA)\n",
           FFT_CONV_DISPATCH_NH_DIRECT_MAX);
    for (int i = 0; i < REP_COUNT; i++) {
        rc |= check_conv_nh(REP_CASES[i].nx, REP_CASES[i].nh);
    }
    printf("--- default/gauss/inverse/conv regression ---\n");
    rc |= check_regression();
    if (rc != 0) {
        fprintf(stderr, "test_conv_dispatch: FAILED\n");
        return 1;
    }
    printf("test_conv_dispatch: PASS (representative (nx,nh) + NULL checks + leg selection + regression)\n");
    return 0;
}
