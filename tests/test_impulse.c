/*
 * tests/test_impulse.c — VER-FFT-002 (impulse)
 *
 *   x[0] = 1, x[n>0] = 0  =>  X[k] = 1 for all k (constant magnitude).
 *
 * For natural-order forward DFT (BEH-FFT-003), the unit impulse maps to an
 * all-ones spectrum: X[k] = sum_n delta[n] * W_N^{kn} = 1.  We assert both
 * |X[k]| ~= 1 (constant magnitude) and X[k] ~= 1+0i (exact phase/order),
 * the latter against the naive DFT reference to pin output order.
 */

#include "fft.h"
#include "test_common.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int N = TEST_N;
    float complex *io = (float complex *)calloc((size_t)N, sizeof(float complex));
    float complex *ref = (float complex *)malloc((size_t)N * sizeof(float complex));
    if (!io || !ref) {
        fprintf(stderr, "OOM\n");
        return 2;
    }

    io[0] = 1.0f + 0.0f * I; /* unit impulse */
    dft_naive(io, ref, N);    /* reference: all ones */

    fft_plan *plan = fft_plan_create(N);
    if (plan == NULL) {
        fprintf(stderr, "fft_plan_create failed\n");
        return 2;
    }
    fft_execute(plan, io);

    float scale = max_mag(ref, N); /* == 1.0 */
    int failures = 0;
    for (int k = 0; k < N; k++) {
        float mag = cabsf(io[k]);
        if (fabsf(mag - 1.0f) > 1e-2f) {
            if (failures < 5) {
                fprintf(stderr, "impulse magnitude mismatch at k=%d: |X|=%.6f\n",
                        k, (double)mag);
            }
            failures++;
        }
        if (!close_complex(io[k], ref[k], scale)) {
            if (failures < 5) {
                fprintf(stderr,
                        "impulse value mismatch at k=%d: got %g%+gi, want "
                        "%g%+gi\n",
                        k, crealf(io[k]), cimagf(io[k]), crealf(ref[k]),
                        cimagf(ref[k]));
            }
            failures++;
        }
    }

    fft_plan_destroy(plan);
    free(io);
    free(ref);

    if (failures != 0) {
        fprintf(stderr, "VER-FFT-002 impulse: FAILED (%d checks)\n", failures);
        return 1;
    }
    printf("VER-FFT-002 impulse: PASS (N=%d)\n", N);
    return 0;
}
