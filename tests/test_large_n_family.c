/*
 * tests/test_large_n_family.c — large-N family JIT verification (topic
 * provenance), copy-then-edit from the topic bench validation legs
 * (poc/jit-large-n-family/bench/bench_large_n.c check-family / check-gauss).
 *
 * Topic: jit-large-n-family (promote).  Clauses: ARCH-FFT-016, ARCH-FFT-008,
 * ARCH-FFT-011.  For every large family N=2^n n=11..15 (N in
 * {2048,4096,8192,16384,32768}) verify through the public ABI:
 *
 *   - (a) unwindowed family JIT vs the Trunk FFTW adapt reference within
 *     float tolerance (execute MUST go via the plan-cached JIT pointer).
 *   - (b) opt-in gauss fused (fft_plan_create_gauss_window + fft_execute)
 *     vs the two-pass oracle (same G from fft_gauss_window_fill, sigma=N/8,
 *     sum(G)=1, then plain unwindowed execute) within float tolerance;
 *     default path must differ from windowed on nonzero input.
 *   - boundary: N=3 / N=65536 stay adapt-only (JIT create NULL) and gauss
 *     create returns NULL (no silent windowed adapt); N<1 create NULL.
 *
 * This is an additional family sweep; the stable VER-FFT-001..003 N=1024
 * pins (tests/test_{linearity,impulse,shift}.c) are NOT rewritten.
 */

#include "fft.h"
#include "fft_adapt.h"
#include "fft_jit.h"
#include "test_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef FUSED_TOL
#define FUSED_TOL 1e-4f
#endif

static const int kLargeN[] = {2048, 4096, 8192, 16384, 32768};
#define LARGE_N_COUNT ((int)(sizeof(kLargeN) / sizeof(kLargeN[0])))

static float max_abs_err(const float complex *a, const float complex *b, int N) {
    float m = 0.0f;
    for (int i = 0; i < N; ++i) {
        float dr = fabsf(crealf(a[i]) - crealf(b[i]));
        float di = fabsf(cimagf(a[i]) - cimagf(b[i]));
        if (dr > m) {
            m = dr;
        }
        if (di > m) {
            m = di;
        }
    }
    return m;
}

static int cmp_bufs(const float complex *a, const float complex *b, int N,
                    float *err_out) {
    float scale = max_mag(b, N);
    float maxd = 0.0f;
    int fail = 0;
    for (int i = 0; i < N; i++) {
        float d = cabsf(a[i] - b[i]);
        if (d > maxd) {
            maxd = d;
        }
        if (!close_complex(a[i], b[i], scale)) {
            fail++;
        }
    }
    if (err_out) {
        *err_out = maxd;
    }
    return fail;
}

static int check_n1_null(void) {
    if (fft_plan_create(0) != NULL || fft_plan_create(-1) != NULL) {
        fprintf(stderr, "fft_plan_create(N<1) must be NULL\n");
        return 1;
    }
    if (fft_jit_kernel_create(0) != NULL || fft_jit_kernel_create(-1) != NULL) {
        fprintf(stderr, "fft_jit_kernel_create(N<1) must be NULL\n");
        return 1;
    }
    if (fft_plan_create_gauss_window(0) != NULL ||
        fft_plan_create_gauss_window(-1) != NULL) {
        fprintf(stderr, "fft_plan_create_gauss_window(N<1) must be NULL\n");
        return 1;
    }
    printf("N<1: NULL PASS (default + jit + gauss)\n");
    return 0;
}

static int check_gauss_off_family(void) {
    fft_plan *p3 = fft_plan_create_gauss_window(3);
    if (p3 != NULL) {
        fprintf(stderr, "gauss create(N=3) must be NULL (no silent windowed adapt)\n");
        fft_plan_destroy(p3);
        return 1;
    }
    fft_plan *pbig = fft_plan_create_gauss_window(65536);
    if (pbig != NULL) {
        fprintf(stderr, "gauss create(N=65536) must be NULL (ceiling is n<=15)\n");
        fft_plan_destroy(pbig);
        return 1;
    }
    printf("gauss create off-family (N=3, N=65536): NULL PASS\n");
    return 0;
}

static int check_adapt_only(int N, const char *why) {
    fft_jit_kernel *jk = fft_jit_kernel_create(N);
    if (jk != NULL) {
        fprintf(stderr, "N=%d: JIT create must be NULL (%s)\n", N, why);
        fft_jit_kernel_destroy(jk);
        return 1;
    }
    fft_plan *p = fft_plan_create(N);
    if (p == NULL) {
        fprintf(stderr, "N=%d: adapt plan_create failed (%s)\n", N, why);
        return 1;
    }
    float complex *io = (float complex *)calloc((size_t)N, sizeof(*io));
    if (io == NULL) {
        fft_plan_destroy(p);
        fprintf(stderr, "N=%d OOM\n", N);
        return 1;
    }
    io[0] = 1.0f;
    fft_execute(p, io);
    int fail = 0;
    /* impulse -> all-ones; cheap O(N) sanity that adapt still works there */
    for (int i = 0; i < N; i++) {
        if (fabsf(cabsf(io[i]) - 1.0f) > 1e-2f) {
            fail++;
        }
    }
    if (fail) {
        fprintf(stderr, "N=%d adapt impulse fail (%d)\n", N, fail);
    }
    free(io);
    fft_plan_destroy(p);
    if (fail) {
        return 1;
    }
    printf("N=%d adapt-only (%s): PASS\n", N, why);
    return 0;
}

/* (a) unwindowed family JIT vs the Trunk FFTW adapt reference. */
static int check_jit_vs_adapt(int N) {
    fft_jit_kernel *jk = fft_jit_kernel_create(N);
    if (jk == NULL || fft_jit_kernel_fn(jk) == NULL) {
        fprintf(stderr, "N=%d: family JIT create failed\n", N);
        fft_jit_kernel_destroy(jk);
        return 1;
    }
    int code = fft_jit_kernel_code_size(jk);
    fft_jit_kernel_destroy(jk);

    fft_plan *jp = fft_plan_create(N);
    fft_adapt_plan *ap = fft_adapt_plan_create(N);
    if (jp == NULL || ap == NULL) {
        fprintf(stderr, "N=%d: plan create failed jit=%p adapt=%p\n", N,
                (void *)jp, (void *)ap);
        fft_plan_destroy(jp);
        fft_adapt_plan_destroy(ap);
        return 1;
    }

    float complex *in = (float complex *)malloc((size_t)N * sizeof(*in));
    float complex *jit = (float complex *)malloc((size_t)N * sizeof(*jit));
    float complex *ad = (float complex *)malloc((size_t)N * sizeof(*ad));
    if (!in || !jit || !ad) {
        fprintf(stderr, "N=%d OOM\n", N);
        free(in);
        free(jit);
        free(ad);
        fft_plan_destroy(jp);
        fft_adapt_plan_destroy(ap);
        return 1;
    }

    uint32_t rng = 0x5eed1234u;
    for (int i = 0; i < N; i++) {
        in[i] = rand_complex(&rng);
        jit[i] = in[i];
        ad[i] = in[i];
    }
    fft_execute(jp, jit);
    fft_adapt_execute(ap, ad);

    float err = 0.0f;
    int fail = cmp_bufs(jit, ad, N, &err);
    if (fail) {
        fprintf(stderr, "N=%d JIT vs adapt FAILED mismatches=%d max_abs=%g code=%d\n",
                N, fail, (double)err, code);
    } else {
        printf("N=%d jit-vs-adapt: PASS (max_abs=%.6e code=%d)\n", N, (double)err, code);
    }

    free(in);
    free(jit);
    free(ad);
    fft_plan_destroy(jp);
    fft_adapt_plan_destroy(ap);
    return fail ? 1 : 0;
}

/* (b) opt-in gauss fused vs two-pass oracle (same G, then plain execute). */
static int check_fused_vs_twopass(int N) {
    fft_plan *def = fft_plan_create(N);
    fft_plan *gauss = fft_plan_create_gauss_window(N);
    if (def == NULL || gauss == NULL) {
        fprintf(stderr, "N=%d plan create failed (def=%p gauss=%p)\n", N,
                (void *)def, (void *)gauss);
        fft_plan_destroy(def);
        fft_plan_destroy(gauss);
        return 1;
    }
    if (def == gauss) {
        fprintf(stderr, "N=%d cache collision: default plan == gauss plan\n", N);
        fft_plan_destroy(def);
        fft_plan_destroy(gauss);
        return 1;
    }

    float *G = (float *)malloc((size_t)N * sizeof(float));
    float complex *src = (float complex *)malloc((size_t)N * sizeof(*src));
    float complex *fused = (float complex *)malloc((size_t)N * sizeof(*fused));
    float complex *twopass = (float complex *)malloc((size_t)N * sizeof(*twopass));
    float complex *unwin = (float complex *)malloc((size_t)N * sizeof(*unwin));
    if (!G || !src || !fused || !twopass || !unwin ||
        fft_gauss_window_fill(N, G) != 0) {
        fprintf(stderr, "N=%d OOM or G fill failed\n", N);
        fft_plan_destroy(def);
        fft_plan_destroy(gauss);
        free(G);
        free(src);
        free(fused);
        free(twopass);
        free(unwin);
        return 1;
    }

    uint32_t rng = 0x51a55eedu ^ (uint32_t)N;
    for (int i = 0; i < N; ++i) {
        src[i] = rand_complex(&rng);
    }

    for (int i = 0; i < N; ++i) {
        fused[i] = src[i];
        twopass[i] = src[i];
        unwin[i] = src[i];
    }

    for (int i = 0; i < N; ++i) {
        twopass[i] *= G[i];
    }
    fft_execute(def, twopass);
    fft_execute(gauss, fused);
    fft_execute(def, unwin);

    float err = max_abs_err(fused, twopass, N);
    float vs_unwin = max_abs_err(fused, unwin, N);
    int rc = 0;
    if (err > FUSED_TOL) {
        fprintf(stderr, "N=%d FAIL fused vs two-pass (err=%g tol=%g)\n", N, err,
                FUSED_TOL);
        rc = 1;
    } else {
        printf("N=%d fused-vs-twopass: PASS (max_abs_err=%.6e)\n", N, err);
    }

    if (vs_unwin < 1e-6f) {
        fprintf(stderr,
                "N=%d FAIL default path matches windowed (nonzero input)\n", N);
        rc = 1;
    } else {
        printf("N=%d default != windowed: PASS\n", N);
    }

    fft_plan_destroy(def);
    fft_plan_destroy(gauss);
    free(G);
    free(src);
    free(fused);
    free(twopass);
    free(unwin);
    return rc;
}

int main(void) {
    int rc = check_n1_null();
    rc |= check_gauss_off_family();
    rc |= check_adapt_only(3, "non-pow2");
    rc |= check_adapt_only(65536, "N>32768");
    for (int i = 0; i < LARGE_N_COUNT; i++) {
        rc |= check_jit_vs_adapt(kLargeN[i]);
        rc |= check_fused_vs_twopass(kLargeN[i]);
    }
    if (rc != 0) {
        fprintf(stderr, "test_large_n_family: FAILED\n");
        return 1;
    }
    printf("test_large_n_family: PASS (family n=11..15 + boundary checks)\n");
    return 0;
}
