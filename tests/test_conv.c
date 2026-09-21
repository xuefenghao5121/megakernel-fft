/*
 * tests/test_conv.c — opt-in cyclic convolution vs unpatched three-seg FFTW
 * (API-FFT-005).  Promotes: jit-filter-mul-fuse.  Clauses: API-FFT-005,
 * BEH-FFT-006.
 *
 * Scale note (HOW vs oracle):
 *   Product path implements the IFFT conjugate trick including /N
 *   (normalized).  Oracle is unpatched vendored FFTW FORWARD + × FFT(h) +
 *   FFTW BACKWARD (FFTW_MEASURE); BACKWARD is unnormalized, so the
 *   three-segment result is N× the product result.  PASS residual =
 *   product vs oracle/N.
 *
 * FFTW MEASURE clobbers arrays during planning: plan first, then
 * regenerate the deterministic input (h and src) before execute / compare.
 *
 *   - conv create returns NULL on N=0 / -1 / 3 / 65536, and on NULL h
 *   - conv plan ≠ default plan (cache key (N, windowed, inverse, conv)
 *     must not collide)
 *   - golden family N=2..32768 (2^n n=1..15) vs three-seg FFTW/N
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
    float complex h[1] = {1.0f + 0.0f * I};
    if (fft_plan_create_conv(0, h) != NULL ||
        fft_plan_create_conv(-1, h) != NULL) {
        fprintf(stderr, "conv create(N<1) must be NULL\n");
        return 1;
    }
    if (fft_plan_create_conv(3, h) != NULL) {
        fprintf(stderr, "conv create(non-family) must be NULL\n");
        return 1;
    }
    if (fft_plan_create_conv(65536, h) != NULL) {
        fprintf(stderr, "conv create(N>32768) must be NULL\n");
        return 1;
    }
    if (fft_plan_create_conv(1024, NULL) != NULL) {
        fprintf(stderr, "conv create(NULL h) must be NULL\n");
        return 1;
    }
    printf("conv NULL on N=0 / -1 / 3 / 65536 + NULL h: PASS\n");
    return 0;
}

static int check_size(int N) {
    const uint32_t hseed = 0x9e3779b9u ^ (uint32_t)N;
    const uint32_t seed = 0x5eed1234u ^ (uint32_t)N;
    const float invN = 1.0f / (float)N;

    float complex *h = (float complex *)malloc((size_t)N * sizeof(*h));
    float complex *src = (float complex *)malloc((size_t)N * sizeof(*src));
    float complex *got = (float complex *)malloc((size_t)N * sizeof(*got));
    float complex *oracle =
        (float complex *)malloc((size_t)N * sizeof(*oracle));
    float complex *scaled =
        (float complex *)malloc((size_t)N * sizeof(*scaled));
    float complex *H = (float complex *)malloc((size_t)N * sizeof(*H));
    float complex *up = (float complex *)fftwf_malloc((size_t)N * sizeof(*up));
    if (h == NULL || src == NULL || got == NULL || oracle == NULL ||
        scaled == NULL || H == NULL || up == NULL) {
        fprintf(stderr, "N=%d OOM\n", N);
        free(h);
        free(src);
        free(got);
        free(oracle);
        free(scaled);
        free(H);
        fftwf_free(up);
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
        fprintf(stderr,
                "N=%d fftwf_plan_dft_1d(FORWARD/BACKWARD, MEASURE) failed\n", N);
        if (fwd != NULL) {
            fftwf_destroy_plan(fwd);
        }
        if (bwd != NULL) {
            fftwf_destroy_plan(bwd);
        }
        free(h);
        free(src);
        free(got);
        free(oracle);
        free(scaled);
        free(H);
        fftwf_free(up);
        return 1;
    }

    /* Regenerate deterministic input (h and src) after MEASURE clobber. */
    fill_lcg(h, N, hseed);
    fill_lcg(src, N, seed);

    fft_plan *conv = fft_plan_create_conv(N, h);
    fft_plan *def = fft_plan_create(N);
    if (conv == NULL || def == NULL) {
        fprintf(stderr, "N=%d plan create failed conv=%p def=%p\n", N,
                (void *)conv, (void *)def);
        fft_plan_destroy(conv);
        fft_plan_destroy(def);
        fftwf_destroy_plan(fwd);
        fftwf_destroy_plan(bwd);
        free(h);
        free(src);
        free(got);
        free(oracle);
        free(scaled);
        free(H);
        fftwf_free(up);
        return 1;
    }
    if (conv == def) {
        fprintf(stderr, "N=%d cache collision: conv plan == default plan\n", N);
        fft_plan_destroy(conv);
        fft_plan_destroy(def);
        fftwf_destroy_plan(fwd);
        fftwf_destroy_plan(bwd);
        free(h);
        free(src);
        free(got);
        free(oracle);
        free(scaled);
        free(H);
        fftwf_free(up);
        return 1;
    }

    /* H = FFT(h) via unpatched FFTW forward. */
    memcpy(up, h, (size_t)N * sizeof(*up));
    fftwf_execute_dft(fwd, (fftwf_complex *)up, (fftwf_complex *)up);
    memcpy(H, up, (size_t)N * sizeof(*H));

    /* Three-segment oracle: FORWARD(src) → × H → BACKWARD. */
    memcpy(up, src, (size_t)N * sizeof(*up));
    fftwf_execute_dft(fwd, (fftwf_complex *)up, (fftwf_complex *)up);
    for (int i = 0; i < N; i++) {
        up[i] = up[i] * H[i];
    }
    fftwf_execute_dft(bwd, (fftwf_complex *)up, (fftwf_complex *)up);
    memcpy(oracle, up, (size_t)N * sizeof(*oracle));

    /* Product: fused conv execute. */
    memcpy(got, src, (size_t)N * sizeof(*got));
    fft_execute(conv, got);

    for (int i = 0; i < N; i++) {
        scaled[i] = oracle[i] * invN;
    }

    float sc_abs = max_abs_err(got, scaled, N);
    float scale = max_mag(scaled, N);
    /* Same envelope as VER close_complex / POC bench_conv (rel=1e-4, abs=1e-2). */
    int pass = (sc_abs <= 1e-4f * scale + 1e-2f);

    int rc = 0;
    if (!pass) {
        fprintf(stderr,
                "N=%d FAIL conv vs three-seg FFTW/N (err=%g scale=%g)\n", N,
                (double)sc_abs, (double)scale);
        rc = 1;
    } else {
        printf("N=%5d conv-vs-threeseg-fftw/N: PASS (max_abs_err=%.6e)\n", N,
               (double)sc_abs);
    }

    fft_plan_destroy(conv);
    fft_plan_destroy(def);
    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(bwd);
    free(h);
    free(src);
    free(got);
    free(oracle);
    free(scaled);
    free(H);
    fftwf_free(up);
    return rc;
}

int main(void) {
    int rc = check_nulls();
    for (int i = 0; i < FAMILY_COUNT; i++) {
        rc |= check_size(FAMILY_N[i]);
    }
    if (rc != 0) {
        fprintf(stderr, "test_conv: FAILED\n");
        return 1;
    }
    printf("test_conv: PASS (family N=2..32768 + NULL checks)\n");
    return 0;
}
