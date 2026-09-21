/*
 * tests/test_gauss_window.c — opt-in fused gauss vs two-pass (BEH-FFT-004).
 *
 * Promotes: jit-gauss-window-fuse.  Clauses: ARCH-FFT-015, BEH-FFT-004,
 * API-FFT-003.  This is an additional opt-in check; VER-FFT-001..003 N=1024
 * unwindowed pins are NOT rewritten and MUST still run on fft_plan_create.
 *
 *   - fused (fft_plan_create_gauss_window + fft_execute) vs two-pass
 *     (same G from fft_gauss_window_fill, then unwindowed fft_execute)
 *     within float tolerance, family N=2^n n=0..10.
 *   - gauss create returns NULL on N<1 / non-family (not power-of-two or
 *     N>32768; ceiling lifted to n<=15 by jit-large-n-family).
 *   - default create ≠ windowed on nonzero input for N>1 (cache key
 *     (N, windowed) must not collide).
 */

#include "fft.h"
#include "fft_jit.h"
#include "test_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef FUSED_TOL
#define FUSED_TOL 1e-4f
#endif

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

static int check_nulls(void) {
    if (fft_plan_create_gauss_window(0) != NULL ||
        fft_plan_create_gauss_window(-1) != NULL) {
        fprintf(stderr, "gauss create(N<1) must be NULL\n");
        return 1;
    }
    if (fft_plan_create_gauss_window(3) != NULL) {
        fprintf(stderr, "gauss create(non-family) must be NULL\n");
        return 1;
    }
    if (fft_plan_create_gauss_window(65536) != NULL) {
        fprintf(stderr, "gauss create(N>32768) must be NULL\n");
        return 1;
    }
    printf("gauss NULL on N<1 / non-family: PASS\n");
    return 0;
}

static int check_size(int N) {
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

    memcpy(fused, src, (size_t)N * sizeof(*src));
    memcpy(twopass, src, (size_t)N * sizeof(*src));
    memcpy(unwin, src, (size_t)N * sizeof(*src));

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
        printf("N=%4d fused-vs-twopass: PASS (max_abs_err=%.6e)\n", N, err);
    }

    if (N == 1) {
        printf("N=   1 default == windowed (G[0]=1): expected\n");
    } else if (vs_unwin < 1e-6f) {
        fprintf(stderr,
                "N=%d FAIL default path matches windowed (nonzero input)\n", N);
        rc = 1;
    } else {
        printf("N=%4d default != windowed: PASS\n", N);
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
    int rc = check_nulls();
    for (int n = 0; n <= 10; ++n) {
        rc |= check_size(1 << n);
    }
    if (rc != 0) {
        fprintf(stderr, "test_gauss_window: FAILED\n");
        return 1;
    }
    printf("test_gauss_window: PASS (family n=0..10 + NULL checks)\n");
    return 0;
}
