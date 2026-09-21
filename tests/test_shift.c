/*
 * tests/test_shift.c — VER-FFT-003 (shift)
 *
 *   Time-domain circular shift by m <=> frequency-domain phase rotation:
 *   if y[n] = x[(n-m) mod N] then FFT(y)[k] = W_N^{k*m} * FFT(x)[k],
 *   with W_N = exp(-j 2*pi/N).
 *
 * Verified at fixed N = 1024, shift m = 137, through the public ABI.
 */

#include "fft.h"
#include "test_common.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int N = TEST_N;
    const int m = 137;
    const float pi = 3.14159265358979323846f;
    uint32_t rng = 0xdeadbeefu;

    float complex *x = (float complex *)malloc((size_t)N * sizeof(float complex));
    float complex *y = (float complex *)malloc((size_t)N * sizeof(float complex));
    float complex *FX = (float complex *)malloc((size_t)N * sizeof(float complex));
    float complex *FY = (float complex *)malloc((size_t)N * sizeof(float complex));
    if (!x || !y || !FX || !FY) {
        fprintf(stderr, "OOM\n");
        return 2;
    }

    for (int n = 0; n < N; n++) {
        x[n] = rand_complex(&rng);
    }
    /* Circular shift: y[n] = x[(n - m) mod N] (m > 0 => shift right). */
    for (int n = 0; n < N; n++) {
        int src = (n - m) % N;
        if (src < 0) {
            src += N;
        }
        y[n] = x[src];
    }

    fft_plan *plan = fft_plan_create(N);
    if (plan == NULL) {
        fprintf(stderr, "fft_plan_create failed\n");
        return 2;
    }

    for (int i = 0; i < N; i++) {
        FX[i] = x[i];
    }
    fft_execute(plan, FX);
    for (int i = 0; i < N; i++) {
        FY[i] = y[i];
    }
    fft_execute(plan, FY);

    float scale = max_mag(FX, N);
    int failures = 0;
    for (int k = 0; k < N; k++) {
        /* Expected: FX[k] * exp(-j 2*pi*k*m / N). */
        float ang = -2.0f * pi * (float)k * (float)m / (float)N;
        float complex rot = cosf(ang) + sinf(ang) * I;
        float complex want = FX[k] * rot;
        if (!close_complex(FY[k], want, scale)) {
            if (failures < 5) {
                fprintf(stderr,
                        "shift mismatch at k=%d: got %g%+gi, want %g%+gi\n",
                        k, crealf(FY[k]), cimagf(FY[k]), crealf(want),
                        cimagf(want));
            }
            failures++;
        }
    }

    fft_plan_destroy(plan);
    free(x);
    free(y);
    free(FX);
    free(FY);

    if (failures != 0) {
        fprintf(stderr, "VER-FFT-003 shift: FAILED (%d/%d mismatches)\n",
                failures, N);
        return 1;
    }
    printf("VER-FFT-003 shift: PASS (N=%d)\n", N);
    return 0;
}
