#ifndef MEGAKERNEL_FFT_TEST_COMMON_H
#define MEGAKERNEL_FFT_TEST_COMMON_H

/*
 * Shared test helpers for VER-FFT-001..003.  Deterministic, no external deps
 * beyond <complex.h> + <math.h>.  A naive O(N^2) DFT serves as the reference
 * for pinning exact values (natural-order forward DFT, BEH-FFT-001).
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>

#define TEST_N 1024

/* Deterministic LCG (Numerical Recipes) for reproducible pseudo-random input. */
static inline uint32_t lcg_next(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

/* Uniform float in [0, 1). */
static inline float randf01(uint32_t *state) {
    return (float)(lcg_next(state) >> 8) / 16777216.0f;
}

/* Uniform complex with real/imag in [-1, 1). */
static inline float complex rand_complex(uint32_t *state) {
    float re = 2.0f * randf01(state) - 1.0f;
    float im = 2.0f * randf01(state) - 1.0f;
    return re + im * I;
}

/* Naive O(N^2) forward DFT (reference). */
static inline void dft_naive(const float complex *in, float complex *out, int N) {
    const float pi = 3.14159265358979323846f;
    for (int k = 0; k < N; k++) {
        float complex acc = 0.0f + 0.0f * I;
        for (int n = 0; n < N; n++) {
            float ang = -2.0f * pi * (float)k * (float)n / (float)N;
            float complex tw = cosf(ang) + sinf(ang) * I;
            acc += in[n] * tw;
        }
        out[k] = acc;
    }
}

/* Max magnitude over a buffer — used to scale the comparison tolerance. */
static inline float max_mag(const float complex *v, int N) {
    float m = 0.0f;
    for (int i = 0; i < N; i++) {
        float a = cabsf(v[i]);
        if (a > m) {
            m = a;
        }
    }
    return m;
}

/*
 * Scale-aware closeness: |got - want| <= rel_tol * scale + abs_tol.
 * Single-precision FFTW has ~1e-6 relative error; these tolerances leave a
 * wide safety margin while still catching any actual sign/order mistake.
 */
static inline int close_complex(float complex got, float complex want, float scale) {
    const float rel_tol = 1e-4f;
    const float abs_tol = 1e-2f;
    float d = cabsf(got - want);
    return d <= rel_tol * scale + abs_tol;
}

#endif /* MEGAKERNEL_FFT_TEST_COMMON_H */
