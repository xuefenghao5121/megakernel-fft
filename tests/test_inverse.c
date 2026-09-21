/*
 * tests/test_inverse.c — opt-in inverse vs unpatched FFTW_BACKWARD (API-FFT-004).
 *
 * Promotes: jit-inverse-1d.  Clauses: API-FFT-004, BEH-FFT-005.
 * Additional opt-in check; VER-FFT-001..003 N=1024 unwindowed pins are
 * NOT rewritten and MUST still run on fft_plan_create.
 *
 * Scale note (HOW vs oracle):
 *   Product path implements the conjugate trick including /N.
 *   Oracle is unpatched vendored FFTW_BACKWARD + FFTW_MEASURE
 *   (fftwf_plan_dft_1d / fftwf_execute_dft, buffers via fftwf_malloc).
 *   Oracle is unnormalized.  PASS residual = product vs oracle/N.
 *
 * FFTW MEASURE clobbers arrays during planning: plan first, then
 * regenerate the deterministic input before execute / compare.
 *
 *   - inverse create returns NULL on N=0 / -1 / 3 / 65536
 *   - default create ≠ inverse on nonzero input for N>1 (cache key
 *     (N, windowed, inverse) must not collide)
 *   - golden family N=2..32768 (2^n n=1..15) vs FFTW_BACKWARD/N
 */

#include "fft.h"
#include "test_common.h"

#include <fftw3.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int FAMILY_N[] = {2,    4,    8,    16,    32,    64,   128,
                               256,  512,  1024, 2048,  4096,  8192, 16384,
                               32768};
#define FAMILY_COUNT ((int)(sizeof(FAMILY_N) / sizeof(FAMILY_N[0])))

static float max_abs_err(const float complex *a, const float complex *b, int N) {
    float m = 0.0f;
    for (int i = 0; i < N; ++i) {
        float d = cabsf(a[i] - b[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

static void fill_lcg(float complex *io, int N, uint32_t seed) {
    uint32_t rng = seed;
    for (int i = 0; i < N; i++) {
        io[i] = rand_complex(&rng);
    }
}

static int check_nulls(void) {
    if (fft_plan_create_inverse(0) != NULL ||
        fft_plan_create_inverse(-1) != NULL) {
        fprintf(stderr, "inverse create(N<1) must be NULL\n");
        return 1;
    }
    if (fft_plan_create_inverse(3) != NULL) {
        fprintf(stderr, "inverse create(non-family) must be NULL\n");
        return 1;
    }
    if (fft_plan_create_inverse(65536) != NULL) {
        fprintf(stderr, "inverse create(N>32768) must be NULL\n");
        return 1;
    }
    printf("inverse NULL on N=0 / -1 / 3 / 65536: PASS\n");
    return 0;
}

static int check_size(int N) {
    const uint32_t seed = 0x5eed1234u ^ (uint32_t)N;
    const float invN = 1.0f / (float)N;

    float complex *src = (float complex *)malloc((size_t)N * sizeof(*src));
    float complex *got = (float complex *)malloc((size_t)N * sizeof(*got));
    float complex *fwd = (float complex *)malloc((size_t)N * sizeof(*fwd));
    float complex *up = (float complex *)fftwf_malloc((size_t)N * sizeof(*up));
    float complex *scaled = (float complex *)malloc((size_t)N * sizeof(*scaled));
    if (src == NULL || got == NULL || fwd == NULL || up == NULL ||
        scaled == NULL) {
        fprintf(stderr, "N=%d OOM\n", N);
        free(src);
        free(got);
        free(fwd);
        fftwf_free(up);
        free(scaled);
        return 1;
    }

    /* Plan FFTW first: MEASURE clobbers `up`.  Then regenerate input. */
    fftwf_plan up_plan = fftwf_plan_dft_1d(N, (fftwf_complex *)up,
                                           (fftwf_complex *)up, FFTW_BACKWARD,
                                           FFTW_MEASURE);
    if (up_plan == NULL) {
        fprintf(stderr, "N=%d fftwf_plan_dft_1d(FFTW_BACKWARD, MEASURE) failed\n",
                N);
        free(src);
        free(got);
        free(fwd);
        fftwf_free(up);
        free(scaled);
        return 1;
    }

    fft_plan *inv = fft_plan_create_inverse(N);
    fft_plan *def = fft_plan_create(N);
    if (inv == NULL || def == NULL) {
        fprintf(stderr, "N=%d plan create failed inv=%p def=%p\n", N,
                (void *)inv, (void *)def);
        fft_plan_destroy(inv);
        fft_plan_destroy(def);
        fftwf_destroy_plan(up_plan);
        free(src);
        free(got);
        free(fwd);
        fftwf_free(up);
        free(scaled);
        return 1;
    }
    if (inv == def) {
        fprintf(stderr, "N=%d cache collision: inverse plan == default plan\n",
                N);
        fft_plan_destroy(inv);
        fft_plan_destroy(def);
        fftwf_destroy_plan(up_plan);
        free(src);
        free(got);
        free(fwd);
        fftwf_free(up);
        free(scaled);
        return 1;
    }

    fill_lcg(src, N, seed);
    memcpy(got, src, (size_t)N * sizeof(*got));
    memcpy(fwd, src, (size_t)N * sizeof(*fwd));
    memcpy(up, src, (size_t)N * sizeof(*up));

    fft_execute(inv, got);
    fft_execute(def, fwd);
    fftwf_execute_dft(up_plan, (fftwf_complex *)up, (fftwf_complex *)up);

    for (int i = 0; i < N; i++) {
        scaled[i] = up[i] * invN;
    }

    float sc_abs = max_abs_err(got, scaled, N);
    float vs_fwd = max_abs_err(got, fwd, N);
    float scale = max_mag(scaled, N);
    /* Same envelope as VER close_complex / POC bench_inverse (rel=1e-4, abs=1e-2). */
    int pass = (sc_abs <= 1e-4f * scale + 1e-2f);

    int rc = 0;
    if (!pass) {
        fprintf(stderr,
                "N=%d FAIL inverse vs FFTW_BACKWARD/N (err=%g scale=%g)\n", N,
                (double)sc_abs, (double)scale);
        rc = 1;
    } else {
        printf("N=%5d inverse-vs-fftw-backward/N: PASS (max_abs_err=%.6e)\n", N,
               (double)sc_abs);
    }

    if (N > 1 && vs_fwd < 1e-3f) {
        fprintf(stderr, "N=%d FAIL default path matches inverse (nonzero input)\n",
                N);
        rc = 1;
    } else if (N > 1) {
        printf("N=%5d default != inverse: PASS\n", N);
    }

    fft_plan_destroy(inv);
    fft_plan_destroy(def);
    fftwf_destroy_plan(up_plan);
    free(src);
    free(got);
    free(fwd);
    fftwf_free(up);
    free(scaled);
    return rc;
}

int main(void) {
    int rc = check_nulls();
    for (int i = 0; i < FAMILY_COUNT; i++) {
        rc |= check_size(FAMILY_N[i]);
    }
    if (rc != 0) {
        fprintf(stderr, "test_inverse: FAILED\n");
        return 1;
    }
    printf("test_inverse: PASS (family N=2..32768 + NULL checks)\n");
    return 0;
}
