/*
 * tests/test_family.c - N=2^n family dispatch verification (topic provenance).
 *
 * Topic: jit-twiddle-imm-lut (promote).  Family n=0..10 sweep through the
 * public ABI: for every N=2^n n=0..10 (N in {1,2,4,...,1024}) verify the JIT
 * family dispatch reproduces the natural-order forward DFT via topic-local
 * isomorphic criteria (exact-vs-naive / impulse / linearity / shift).  This
 * is an additional family sweep; the stable VER-FFT-001..003 N=1024 pins
 * (tests/test_{linearity,impulse,shift}.c) are NOT rewritten.
 *
 * Clauses: {#ARCH-FFT-011}.  Promotes: jit-twiddle-imm-lut.
 */

#include "fft.h"
#include "test_common.h"

#include <stdio.h>
#include <stdlib.h>

/* local ilog2 (avoid clashing with jit internals) */
static int ilog2_n(int N) {
    int n = 0;
    while ((1 << n) < N) n++;
    return n;
}

static int check_exact(int N) {
    uint32_t rng = 0x5eed1234u;
    float complex *in = (float complex *)malloc((size_t)N * sizeof(float complex));
    float complex *io = (float complex *)malloc((size_t)N * sizeof(float complex));
    float complex *ref = (float complex *)malloc((size_t)N * sizeof(float complex));
    if (!in || !io || !ref) {
        fprintf(stderr, "N=%d OOM\n", N);
        return 1;
    }
    for (int i = 0; i < N; i++) {
        in[i] = rand_complex(&rng);
        io[i] = in[i];
    }
    dft_naive(in, ref, N);

    fft_plan *plan = fft_plan_create(N);
    if (plan == NULL) {
        fprintf(stderr, "N=%d fft_plan_create failed\n", N);
        free(in); free(io); free(ref);
        return 1;
    }
    fft_execute(plan, io);

    float scale = max_mag(ref, N);
    int failures = 0;
    for (int k = 0; k < N; k++) {
        if (!close_complex(io[k], ref[k], scale)) {
            if (failures < 5) {
                fprintf(stderr,
                        "N=%d exact mismatch at k=%d: got %g%+gi want %g%+gi\n",
                        N, k, crealf(io[k]), cimagf(io[k]), crealf(ref[k]),
                        cimagf(ref[k]));
            }
            failures++;
        }
    }
    fft_plan_destroy(plan);
    free(in); free(io); free(ref);

    if (failures != 0) {
        fprintf(stderr, "N=%d exact vs naive DFT: FAILED (%d)\n", N, failures);
        return 1;
    }
    printf("N=%3d exact-vs-naive: PASS (n=%d)\n", N, ilog2_n(N));
    return 0;
}

static int check_impulse(int N) {
    float complex *io = (float complex *)calloc((size_t)N, sizeof(float complex));
    float complex *ref = (float complex *)malloc((size_t)N * sizeof(float complex));
    if (!io || !ref) {
        fprintf(stderr, "N=%d OOM\n", N);
        return 1;
    }
    io[0] = 1.0f + 0.0f * I;
    dft_naive(io, ref, N);
    fft_plan *plan = fft_plan_create(N);
    if (plan == NULL) {
        fprintf(stderr, "N=%d plan fail\n", N);
        free(io); free(ref);
        return 1;
    }
    fft_execute(plan, io);
    float scale = max_mag(ref, N);
    int failures = 0;
    for (int k = 0; k < N; k++) {
        if (fabsf(cabsf(io[k]) - 1.0f) > 1e-2f) failures++;
        if (!close_complex(io[k], ref[k], scale)) failures++;
    }
    fft_plan_destroy(plan);
    free(io); free(ref);
    if (failures != 0) {
        fprintf(stderr, "N=%d impulse: FAILED (%d)\n", N, failures);
        return 1;
    }
    printf("N=%3d impulse: PASS\n", N);
    return 0;
}

static int check_linearity(int N) {
    const float complex alpha = 0.5f - 0.25f * I;
    const float complex beta = -0.75f + 0.5f * I;
    uint32_t rng = 0x5eed1234u;
    float complex *X = malloc((size_t)N * sizeof(float complex));
    float complex *Y = malloc((size_t)N * sizeof(float complex));
    float complex *L = malloc((size_t)N * sizeof(float complex));
    float complex *FX = malloc((size_t)N * sizeof(float complex));
    float complex *FY = malloc((size_t)N * sizeof(float complex));
    float complex *R = malloc((size_t)N * sizeof(float complex));
    if (!X || !Y || !L || !FX || !FY || !R) {
        fprintf(stderr, "N=%d OOM\n", N);
        return 1;
    }
    for (int i = 0; i < N; i++) { X[i] = rand_complex(&rng); Y[i] = rand_complex(&rng); }
    fft_plan *plan = fft_plan_create(N);
    if (plan == NULL) {
        fprintf(stderr, "N=%d plan fail\n", N);
        return 1;
    }
    for (int i = 0; i < N; i++) L[i] = alpha * X[i] + beta * Y[i];
    fft_execute(plan, L);
    for (int i = 0; i < N; i++) FX[i] = X[i];
    fft_execute(plan, FX);
    for (int i = 0; i < N; i++) FY[i] = Y[i];
    fft_execute(plan, FY);
    for (int i = 0; i < N; i++) R[i] = alpha * FX[i] + beta * FY[i];
    float scale = max_mag(R, N);
    int failures = 0;
    for (int k = 0; k < N; k++) if (!close_complex(L[k], R[k], scale)) failures++;
    fft_plan_destroy(plan);
    free(X); free(Y); free(L); free(FX); free(FY); free(R);
    if (failures != 0) {
        fprintf(stderr, "N=%d linearity: FAILED (%d)\n", N, failures);
        return 1;
    }
    printf("N=%3d linearity: PASS\n", N);
    return 0;
}

static int check_shift(int N) {
    const int m = (N > 137) ? 137 : (N > 1 ? N / 2 : 1);
    const float pi = 3.14159265358979323846f;
    uint32_t rng = 0xdeadbeefu;
    float complex *x = malloc((size_t)N * sizeof(float complex));
    float complex *y = malloc((size_t)N * sizeof(float complex));
    float complex *FX = malloc((size_t)N * sizeof(float complex));
    float complex *FY = malloc((size_t)N * sizeof(float complex));
    if (!x || !y || !FX || !FY) {
        fprintf(stderr, "N=%d OOM\n", N);
        return 1;
    }
    for (int n = 0; n < N; n++) x[n] = rand_complex(&rng);
    for (int n = 0; n < N; n++) {
        int src = (n - m) % N; if (src < 0) src += N; y[n] = x[src];
    }
    fft_plan *plan = fft_plan_create(N);
    if (plan == NULL) {
        fprintf(stderr, "N=%d plan fail\n", N);
        return 1;
    }
    for (int i = 0; i < N; i++) FX[i] = x[i];
    fft_execute(plan, FX);
    for (int i = 0; i < N; i++) FY[i] = y[i];
    fft_execute(plan, FY);
    float scale = max_mag(FX, N);
    int failures = 0;
    for (int k = 0; k < N; k++) {
        float ang = -2.0f * pi * (float)k * (float)m / (float)N;
        float complex rot = cosf(ang) + sinf(ang) * I;
        float complex want = FX[k] * rot;
        if (!close_complex(FY[k], want, scale)) failures++;
    }
    fft_plan_destroy(plan);
    free(x); free(y); free(FX); free(FY);
    if (failures != 0) {
        fprintf(stderr, "N=%d shift: FAILED (%d)\n", N, failures);
        return 1;
    }
    printf("N=%3d shift: PASS\n", N);
    return 0;
}

int main(void) {
    int rc = 0;
    for (int n = 0; n <= 10; ++n) {
        const int N = 1 << n;
        printf("== family N=%d (n=%d) ==\n", N, n);
        rc |= check_exact(N);
        rc |= check_impulse(N);
        rc |= check_linearity(N);
        rc |= check_shift(N);
    }
    if (rc != 0) {
        fprintf(stderr, "test_family: FAILED\n");
        return 1;
    }
    printf("test_family: PASS (family n=0..10)\n");
    return 0;
}
