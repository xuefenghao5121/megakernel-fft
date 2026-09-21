/*
 * tests/test_linearity.c — VER-FFT-001 (linearity)
 *
 *   FFT(alpha*X + beta*Y) == alpha*FFT(X) + beta*FFT(Y)
 *
 * Verified at fixed N = 1024 using two deterministic pseudo-random inputs and
 * fixed complex scalars, all computed through the public include/fft.h ABI.
 */

#include "fft.h"
#include "test_common.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int N = TEST_N;
    const float complex alpha = 0.5f - 0.25f * I;
    const float complex beta = -0.75f + 0.5f * I;
    uint32_t rng = 0x5eed1234u;

    float complex *X = (float complex *)malloc((size_t)N * sizeof(float complex));
    float complex *Y = (float complex *)malloc((size_t)N * sizeof(float complex));
    float complex *LHS = (float complex *)malloc((size_t)N * sizeof(float complex));
    float complex *FX = (float complex *)malloc((size_t)N * sizeof(float complex));
    float complex *FY = (float complex *)malloc((size_t)N * sizeof(float complex));
    float complex *RHS = (float complex *)malloc((size_t)N * sizeof(float complex));
    if (!X || !Y || !LHS || !FX || !FY || !RHS) {
        fprintf(stderr, "OOM\n");
        return 2;
    }

    for (int i = 0; i < N; i++) {
        X[i] = rand_complex(&rng);
        Y[i] = rand_complex(&rng);
    }

    fft_plan *plan = fft_plan_create(N);
    if (plan == NULL) {
        fprintf(stderr, "fft_plan_create failed\n");
        return 2;
    }

    /* LHS = FFT(alpha*X + beta*Y) */
    for (int i = 0; i < N; i++) {
        LHS[i] = alpha * X[i] + beta * Y[i];
    }
    fft_execute(plan, LHS);

    /* RHS = alpha*FFT(X) + beta*FFT(Y) */
    for (int i = 0; i < N; i++) {
        FX[i] = X[i];
    }
    fft_execute(plan, FX);
    for (int i = 0; i < N; i++) {
        FY[i] = Y[i];
    }
    fft_execute(plan, FY);
    for (int i = 0; i < N; i++) {
        RHS[i] = alpha * FX[i] + beta * FY[i];
    }

    float scale = max_mag(RHS, N);
    int failures = 0;
    for (int k = 0; k < N; k++) {
        if (!close_complex(LHS[k], RHS[k], scale)) {
            if (failures < 5) {
                fprintf(stderr,
                        "linearity mismatch at k=%d: got %g%+gi, want %g%+gi\n",
                        k, crealf(LHS[k]), cimagf(LHS[k]), crealf(RHS[k]),
                        cimagf(RHS[k]));
            }
            failures++;
        }
    }

    fft_plan_destroy(plan);
    free(X);
    free(Y);
    free(LHS);
    free(FX);
    free(FY);
    free(RHS);

    if (failures != 0) {
        fprintf(stderr, "VER-FFT-001 linearity: FAILED (%d/%d mismatches)\n",
                failures, N);
        return 1;
    }
    printf("VER-FFT-001 linearity: PASS (N=%d)\n", N);
    return 0;
}
