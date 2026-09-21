/*
 * tests/test_fp64.c — opt-in fp64 forward kernel vs double FFTW oracle
 * (API-FFT-008 / BEH-FFT-009) + fp32 regression.
 * Promotes: jit-fp64-forward.  Clauses: API-FFT-008, BEH-FFT-009.
 *
 * fp64 correctness: fft_plan_create_fp64(N) + fft_execute_fp64 vs
 * fftw_plan_dft_1d(double, FFTW_FORWARD, FFTW_MEASURE) within double
 * tolerance (rel=1e-12, abs=1e-9 — pinned for fp64 precision).  The
 * representative N=1024 runs first, then the full family powers n=0..15
 * (N=1..32768).  Illegal N (0 / -1 / non-power-of-two 3 / above-ceiling
 * 65536) → NULL; fft_execute_fp64 on NULL plan / NULL io is a no-op, and a
 * non-fp64 plan is a defensive no-op.
 *
 * fp32 regression: default / gauss / inverse / conv / linear-conv /
 * conv-dispatch results MUST NOT regress (same fp32 oracle envelope as the
 * other tests; default fp32 create/execute semantics unchanged).
 *
 * The fp64 oracle is the SYSTEM double-precision libfftw3.so.3 (unpatched
 * fftw_*); the vendored third_party/fftw is single-precision only.
 * third_party/fftw/api/fftw3.h (FFTW_DEFINE_API) declares both fftw_* and
 * fftwf_*.  FFTW MEASURE clobbers its buffer: plan the oracle first, then
 * regenerate the deterministic input before execute / compare (same
 * technique as tests/test_inverse.c, type swapped to double).
 */

#include "fft.h"
#include "fft_jit.h"
#include "test_common.h"

#include <fftw3.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* N=2^n n=0..15 (N=1..32768). */
static const int FAMILY_N[] = {1,     2,     4,     8,     16,    32,
                               64,    128,   256,   512,   1024,  2048,
                               4096,  8192,  16384, 32768};
#define FAMILY_COUNT ((int)(sizeof(FAMILY_N) / sizeof(FAMILY_N[0])))

/* fp64 tolerance: double rel=1e-12, abs=1e-9 (BEH-FFT-009). */
#define FP64_REL_TOL 1e-12
#define FP64_ABS_TOL 1e-9

/* ---- double helpers (fp64 vs double FFTW oracle) ---- */

static double randd01(uint32_t *st) {
    return (double)(lcg_next(st) >> 11) / 2097152.0;
}

static double complex rand_complex_d(uint32_t *st) {
    return (2.0 * randd01(st) - 1.0) + (2.0 * randd01(st) - 1.0) * I;
}

static void fill_lcg_d(double complex *io, int N, uint32_t seed) {
    uint32_t rng = seed;
    for (int i = 0; i < N; i++) {
        io[i] = rand_complex_d(&rng);
    }
}

static double max_mag_d(const double complex *v, int N) {
    double m = 0.0;
    for (int i = 0; i < N; i++) {
        double a = cabs(v[i]);
        if (a > m) {
            m = a;
        }
    }
    return m;
}

static double max_abs_err_d(const double complex *a, const double complex *b,
                            int N) {
    double m = 0.0;
    for (int i = 0; i < N; i++) {
        double d = cabs(a[i] - b[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

/* ---- illegal N → NULL + NULL plan/io no-op ----
 * API-FFT-013 (topic jit-anyN-fwd-inv) revised the fp64 N domain to the
 * factor coverage set: N=3 leaves the illegal list (the tiny wrapper-floor
 * tier, accepted); 17 added as the out-of-set prime probe. */
static int check_nulls(void) {
    int bad[] = {0, -1, 65536, 17};
    for (int i = 0; i < (int)(sizeof(bad) / sizeof(bad[0])); i++) {
        fft_plan *p = fft_plan_create_fp64(bad[i]);
        if (p != NULL) {
            fprintf(stderr, "fft_plan_create_fp64(N=%d) must be NULL\n", bad[i]);
            fft_plan_destroy(p);
            return 1;
        }
    }

    /* fft_execute_fp64 NULL plan / NULL io is a no-op; a non-fp64 plan is a
     * defensive no-op; fp64 plan must live in its own cache table (never
     * collide with the fp32 default plan). */
    double complex io[8] = {0};
    fft_plan *plan = fft_plan_create_fp64(8);
    fft_plan *fp32 = fft_plan_create(8);
    if (plan == NULL || fp32 == NULL) {
        fprintf(stderr, "no-op probe plan create failed\n");
        fft_plan_destroy(plan);
        fft_plan_destroy(fp32);
        return 1;
    }
    if (plan == fp32) {
        fprintf(stderr, "fp64 plan collided with fp32 default plan\n");
        fft_plan_destroy(plan);
        fft_plan_destroy(fp32);
        return 1;
    }
    fft_execute_fp64(plan, NULL);
    fft_execute_fp64(NULL, io);
    fft_execute_fp64(fp32, io); /* defensive no-op */
    fft_plan_destroy(plan);
    fft_plan_destroy(fp32);
    printf("fp64 NULL on N=0/-1/65536/17 + NULL plan/io no-op: PASS\n");
    return 0;
}

/* ---- fp64 forward vs double FFTW oracle ---- */
static int check_fp64_size(int N) {
    const uint32_t seed = 0x5eed1234u ^ (uint32_t)N;

    double complex *got =
        (double complex *)malloc((size_t)N * sizeof(*got));
    double complex *ref =
        (double complex *)fftw_malloc((size_t)N * sizeof(*ref));
    if (got == NULL || ref == NULL) {
        fprintf(stderr, "N=%d fp64 OOM\n", N);
        free(got);
        if (ref != NULL) {
            fftw_free(ref);
        }
        return 1;
    }

    /* Plan the double oracle FIRST (MEASURE clobbers `ref`). */
    fftw_plan dp = fftw_plan_dft_1d(N, (fftw_complex *)ref, (fftw_complex *)ref,
                                    FFTW_FORWARD, FFTW_MEASURE);
    if (dp == NULL) {
        fprintf(stderr, "N=%d fftw_plan_dft_1d(double, MEASURE) failed\n", N);
        free(got);
        fftw_free(ref);
        return 1;
    }

    fft_plan *plan = fft_plan_create_fp64(N);
    if (plan == NULL) {
        fprintf(stderr, "N=%d fft_plan_create_fp64 failed\n", N);
        fftw_destroy_plan(dp);
        free(got);
        fftw_free(ref);
        return 1;
    }

    /* Regenerate input AFTER planning. */
    fill_lcg_d(got, N, seed);
    memcpy(ref, got, (size_t)N * sizeof(*got));

    fft_execute_fp64(plan, got);
    fftw_execute_dft(dp, (fftw_complex *)ref, (fftw_complex *)ref);

    double scale = max_mag_d(ref, N);
    double err = max_abs_err_d(got, ref, N);
    int pass = err <= FP64_REL_TOL * scale + FP64_ABS_TOL;

    printf("N=%5d fp64-forward-vs-fftw-double max_abs=%.6e scale=%.6e%s\n", N,
           err, scale, pass ? " PASS" : " FAIL");
    if (!pass) {
        fprintf(stderr, "N=%d FAIL fp64 vs double FFTW (tol rel=%g abs=%g)\n",
                N, FP64_REL_TOL, FP64_ABS_TOL);
    }

    fft_plan_destroy(plan);
    fftw_destroy_plan(dp);
    free(got);
    fftw_free(ref);
    return pass ? 0 : 1;
}

/* ---- fp32 regression (default / gauss / inverse / conv / linear-conv /
 *      conv-dispatch semantics unchanged) ---- */

static void fill_lcg_f(float complex *io, int N, uint32_t seed) {
    uint32_t rng = seed;
    for (int i = 0; i < N; i++) {
        io[i] = rand_complex(&rng);
    }
}

/* PASS residual: every element satisfies |got-want| <= rel*scale + abs
 * (test_common.h close_complex), scale = max_mag(want). */
static int close_all_f(const float complex *got, const float complex *want,
                       int n, float *max_abs_out) {
    float scale = max_mag(want, n);
    float ma = 0.0f;
    int ok = 1;
    for (int i = 0; i < n; i++) {
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
static void direct_linear_conv(const float complex *x, int nx,
                               const float complex *h, int nh,
                               float complex *out) {
    const int ylen = nx + nh - 1;
    for (int m = 0; m < ylen; m++) {
        float complex acc = 0.0f + 0.0f * I;
        int j0 = (m >= nh) ? (m - nh + 1) : 0;
        int j1 = (m < nx) ? m : (nx - 1);
        for (int j = j0; j <= j1; j++) {
            acc += x[j] * h[m - j];
        }
        out[m] = acc;
    }
}

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
    fill_lcg_f(h, N, 0x9e3779b9u);
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
        fprintf(stderr, "regression N=%d cache collision among fp32 plans\n", N);
        rc = 1;
    }

    float a = 0.0f;

    /* Default forward vs FFTW forward. */
    fill_lcg_f(src, N, 0xC0FFEEu);
    memcpy(got, src, (size_t)N * sizeof(*got));
    memcpy(up, src, (size_t)N * sizeof(*up));
    fft_execute(def, got);
    fftwf_execute_dft(fwd, (fftwf_complex *)up, (fftwf_complex *)up);
    int p = close_all_f(got, up, N, &a);
    printf("regression N=%d default-forward-vs-fftw max_abs=%.6e%s\n", N,
           (double)a, p ? " PASS" : " FAIL");
    rc |= (p ? 0 : 1);

    /* Inverse vs FFTW backward / N. */
    fill_lcg_f(src, N, 0xC0FFEEu ^ 1u);
    memcpy(got, src, (size_t)N * sizeof(*got));
    memcpy(up, src, (size_t)N * sizeof(*up));
    fft_execute(inv, got);
    fftwf_execute_dft(bwd, (fftwf_complex *)up, (fftwf_complex *)up);
    for (int i = 0; i < N; i++) {
        ref[i] = up[i] * invN;
    }
    p = close_all_f(got, ref, N, &a);
    printf("regression N=%d inverse-vs-fftw-backward/N max_abs=%.6e%s\n", N,
           (double)a, p ? " PASS" : " FAIL");
    rc |= (p ? 0 : 1);

    /* Gauss fused vs two-pass (window then default forward). */
    if (fft_gauss_window_fill(N, G) != 0) {
        fprintf(stderr, "regression N=%d fft_gauss_window_fill failed\n", N);
        rc = 1;
    } else {
        fill_lcg_f(src, N, 0xC0FFEEu ^ 2u);
        memcpy(got, src, (size_t)N * sizeof(*got));
        memcpy(ref, src, (size_t)N * sizeof(*ref));
        fft_execute(gauss, got);
        for (int i = 0; i < N; i++) {
            ref[i] *= G[i];
        }
        fft_execute(def, ref);
        p = close_all_f(got, ref, N, &a);
        printf("regression N=%d gauss-fused-vs-twopass max_abs=%.6e%s\n", N,
               (double)a, p ? " PASS" : " FAIL");
        rc |= (p ? 0 : 1);
    }

    /* Conv vs FFTW three-seg (forward + x FFT(h) + backward) / N. */
    memcpy(up, h, (size_t)N * sizeof(*up));
    fftwf_execute_dft(fwd, (fftwf_complex *)up, (fftwf_complex *)up);
    memcpy(H, up, (size_t)N * sizeof(*H));
    fill_lcg_f(src, N, 0xC0FFEEu ^ 3u);
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
    p = close_all_f(got, ref, N, &a);
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

    /* ---- linear-conv (OLS single block) vs direct time-domain ---- */
    {
        const int Nl = 256;
        const int nh = 16;
        const int valid = Nl - nh + 1;
        float complex *hl =
            (float complex *)malloc((size_t)nh * sizeof(*hl));
        float complex *xl =
            (float complex *)malloc((size_t)Nl * sizeof(*xl));
        float complex *yl =
            (float complex *)malloc((size_t)(Nl + nh - 1) * sizeof(*yl));
        float complex *ol =
            (float complex *)malloc((size_t)Nl * sizeof(*ol));
        if (hl == NULL || xl == NULL || yl == NULL || ol == NULL) {
            fprintf(stderr, "linear-conv regression OOM\n");
            free(hl);
            free(xl);
            free(yl);
            free(ol);
            return 1;
        }
        fill_lcg_f(hl, nh, 0x1234u);
        fill_lcg_f(xl, Nl, 0x5678u);
        direct_linear_conv(xl, Nl, hl, nh, yl);
        fft_plan *lp = fft_plan_create_linear_conv(Nl, hl, nh);
        if (lp == NULL) {
            fprintf(stderr, "linear-conv regression plan create failed\n");
            free(hl);
            free(xl);
            free(yl);
            free(ol);
            return 1;
        }
        memcpy(ol, xl, (size_t)Nl * sizeof(*ol));
        fft_execute(lp, ol);
        /* Valid samples = yl[nh-1 .. Nl-1] written to ol[0 .. Nl-nh]. */
        float lerr = 0.0f;
        for (int i = 0; i < valid; i++) {
            float d = cabsf(ol[i] - yl[nh - 1 + i]);
            if (d > lerr) {
                lerr = d;
            }
        }
        float lscale = max_mag(yl, Nl + nh - 1);
        p = (lerr <= 1e-4f * lscale + 1e-2f);
        printf("regression linear-conv-vs-direct max_abs=%.6e%s\n",
               (double)lerr, p ? " PASS" : " FAIL");
        rc |= (p ? 0 : 1);
        fft_plan_destroy(lp);
        free(hl);
        free(xl);
        free(yl);
        free(ol);
    }

    /* ---- conv-dispatch (direct leg) vs direct time-domain ---- */
    {
        const int nx = 64;
        const int nh = 8;
        const int ylen = nx + nh - 1;
        float complex *hd =
            (float complex *)malloc((size_t)nh * sizeof(*hd));
        float complex *xd =
            (float complex *)malloc((size_t)nx * sizeof(*xd));
        float complex *yd =
            (float complex *)malloc((size_t)ylen * sizeof(*yd));
        float complex *od =
            (float complex *)malloc((size_t)ylen * sizeof(*od));
        if (hd == NULL || xd == NULL || yd == NULL || od == NULL) {
            fprintf(stderr, "conv-dispatch regression OOM\n");
            free(hd);
            free(xd);
            free(yd);
            free(od);
            return 1;
        }
        fill_lcg_f(hd, nh, 0x1111u);
        fill_lcg_f(xd, nx, 0x2222u);
        direct_linear_conv(xd, nx, hd, nh, yd);
        fft_plan *dp_plan = fft_plan_create_conv_dispatch(nx, hd, nh);
        if (dp_plan == NULL) {
            fprintf(stderr, "conv-dispatch regression plan create failed\n");
            free(hd);
            free(xd);
            free(yd);
            free(od);
            return 1;
        }
        memcpy(od, xd, (size_t)nx * sizeof(*od));
        for (int i = nx; i < ylen; i++) {
            od[i] = 0.0f + 0.0f * I;
        }
        fft_execute(dp_plan, od);
        float derr = 0.0f;
        for (int i = 0; i < ylen; i++) {
            float d = cabsf(od[i] - yd[i]);
            if (d > derr) {
                derr = d;
            }
        }
        float dscale = max_mag(yd, ylen);
        p = (derr <= 1e-4f * dscale + 1e-2f);
        printf("regression conv-dispatch-vs-direct max_abs=%.6e%s\n",
               (double)derr, p ? " PASS" : " FAIL");
        rc |= (p ? 0 : 1);
        fft_plan_destroy(dp_plan);
        free(hd);
        free(xd);
        free(yd);
        free(od);
    }

    return rc;
}

int main(void) {
    int rc = check_nulls();
    printf("--- fp64 forward vs double FFTW oracle ---\n");
    printf("# tolerance: rel=%g abs=%g (double, fp64 precision)\n", FP64_REL_TOL,
           FP64_ABS_TOL);
    rc |= check_fp64_size(1024);
    for (int i = 0; i < FAMILY_COUNT; i++) {
        rc |= check_fp64_size(FAMILY_N[i]);
    }
    printf("--- fp32 default/gauss/inverse/conv/linear-conv/conv-dispatch regression ---\n");
    rc |= check_regression();
    if (rc != 0) {
        fprintf(stderr, "test_fp64: FAILED\n");
        return 1;
    }
    printf("test_fp64: PASS (fp64-vs-double-oracle + illegal NULL + fp32 regression)\n");
    return 0;
}
