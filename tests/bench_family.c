/*
 * tests/bench_family.c — post-promote Trunk family observed-golden harness
 * (VER-FFT-004 / META-006 refresh; promotes trunk-large-n-baseline-refresh).
 *
 * Measures the 1D complex in-place execute path, single thread, across the
 * N=2^n family N in {2,...,32768} (n=1..15), for two subjects under one
 * protocol:
 *
 *   subject "project"  — the **current post-promote Trunk mainline** public
 *                        ABI include/fft.h (fft_plan_create + fft_execute).
 *                        N=2^n n=0..15 dispatches to the family JIT kernel
 *                        (src/jit/fft_jit.cpp, Xbyak; [[ARCH-FFT-008]] /
 *                        [[ARCH-FFT-011]] / [[ARCH-FFT-016]]), NOT the
 *                        Genesis FFTW-ESTIMATE adapt skeleton, NOT gauss
 *                        opt-in.  This harness MUST be built against a
 *                        default libfft_megakernel.a linked WITH src/jit
 *                        (no -DFFT_JIT_DISABLE).
 *   subject "upstream" — unpatched upstream FFTW MEASURE (fftwf_plan_dft_1d
 *                        with FFTW_MEASURE), the DEC-named comparison target
 *                        (name-only external reference, [[CON-FFT-003]]).
 *
 * Protocol (inherits cfg-family-pow2-1d-complex-inplace, parameterized by N):
 *   - deterministic input (LCG seed=0x5eed1234, same family as test_common.h)
 *   - single thread (no fftwf_init_threads / OpenMP)
 *   - plan creation timed once per subject (reported, not part of execute)
 *   - execute-only wall time via clock_gettime(CLOCK_MONOTONIC)
 *   - min per-execute retained over ROUNDS × ITERS
 *   - throughput = 1 / per-execute seconds
 *
 * Knobs (printed on every row):
 *   N <  2048: WARMUP=10000 / ITERS=200000 / ROUNDS=9
 *   N >= 2048: WARMUP=50    / ITERS=200    / ROUNDS=3  (both legs identical)
 *
 * Checksum note: execute timing is data-independent for a fixed codelet/JIT
 * path, so the checksum is taken from the deterministic *input* (before any
 * transform) purely as a reproducibility marker.
 *
 * Buffer note (fairness): "project" uses plain malloc (public ABI makes no
 * alignment guarantee, [[API-FFT-001]]); "upstream" uses fftwf_malloc
 * (aligned path FFTW MEASURE expects).
 */

#include "fft.h" /* public ABI — post-promote Trunk mainline */

#include <fftw3.h> /* upstream FFTW reference */

#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SMALL_WARMUP 10000L
#define SMALL_ITERS 200000L
#define SMALL_ROUNDS 9

#define LARGE_WARMUP 50L
#define LARGE_ITERS 200L
#define LARGE_ROUNDS 3
#define LARGE_N_FLOOR 2048

/* N=2^n n=1..15 (N=1 intentionally excluded). */
static const int FAMILY_N[] = {2,    4,    8,    16,    32,    64,   128,
                               256,  512,  1024, 2048,  4096,  8192, 16384,
                               32768};
#define FAMILY_COUNT ((int)(sizeof(FAMILY_N) / sizeof(FAMILY_N[0])))

static void knobs_for(int N, long *warmup, long *iters, int *rounds) {
    if (N >= LARGE_N_FLOOR) {
        *warmup = LARGE_WARMUP;
        *iters = LARGE_ITERS;
        *rounds = LARGE_ROUNDS;
    } else {
        *warmup = SMALL_WARMUP;
        *iters = SMALL_ITERS;
        *rounds = SMALL_ROUNDS;
    }
}

static inline uint32_t lcg_next(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

static inline float randf01(uint32_t *s) {
    return (float)(lcg_next(s) >> 8) / 16777216.0f;
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Run one N, printing one row. Returns 0 on success. */
static int bench_one(int N) {
    long warmup, iters;
    int rounds;
    knobs_for(N, &warmup, &iters, &rounds);

    uint32_t rng = 0x5eed1234u;

    float complex *io = (float complex *)malloc((size_t)N * sizeof(float complex));
    float complex *up = (float complex *)fftwf_malloc((size_t)N * sizeof(float complex));
    if (io == NULL || up == NULL) {
        fprintf(stderr, "N=%d OOM\n", N);
        return 1;
    }
    for (int i = 0; i < N; i++) {
        float re = 2.0f * randf01(&rng) - 1.0f;
        float im = 2.0f * randf01(&rng) - 1.0f;
        io[i] = re + im * I;
        up[i] = re + im * I;
    }

    /* Deterministic input checksum (reproducibility marker). */
    float checksum = 0.0f;
    for (int i = 0; i < N; i++) {
        checksum += crealf(io[i]) + cimagf(io[i]);
    }

    /* --- project kernel (post-promote Trunk mainline, JIT on) --- */
    double t0 = now_ns();
    fft_plan *proj = fft_plan_create(N);
    double proj_plan_ns = now_ns() - t0;
    if (proj == NULL) {
        fprintf(stderr, "fft_plan_create(N=%d) failed\n", N);
        return 2;
    }

    /* --- upstream FFTW MEASURE --- */
    t0 = now_ns();
    fftwf_plan up_plan = fftwf_plan_dft_1d(N, (fftwf_complex *)up, (fftwf_complex *)up,
                                            FFTW_FORWARD, FFTW_MEASURE);
    double up_plan_ns = now_ns() - t0;
    if (up_plan == NULL) {
        fprintf(stderr, "fftwf_plan_dft_1d(FFTW_MEASURE, N=%d) failed\n", N);
        return 2;
    }

    /* --- warmup (same knobs both legs) --- */
    for (long i = 0; i < warmup; i++) {
        fft_execute(proj, io);
    }
    for (long i = 0; i < warmup; i++) {
        fftwf_execute_dft(up_plan, (fftwf_complex *)up, (fftwf_complex *)up);
    }

    /* --- execute-only timing (min over rounds = noise floor) --- */
    double proj_best = 1e18;
    double up_best = 1e18;
    for (int r = 0; r < rounds; r++) {
        double s = now_ns();
        for (long k = 0; k < iters; k++) {
            fft_execute(proj, io);
        }
        double e = now_ns();
        double ns = (e - s) / (double)iters;
        if (ns < proj_best) {
            proj_best = ns;
        }
    }
    for (int r = 0; r < rounds; r++) {
        double s = now_ns();
        for (long k = 0; k < iters; k++) {
            fftwf_execute_dft(up_plan, (fftwf_complex *)up, (fftwf_complex *)up);
        }
        double e = now_ns();
        double ns = (e - s) / (double)iters;
        if (ns < up_best) {
            up_best = ns;
        }
    }

    double proj_throughput = 1e9 / proj_best;
    double up_throughput = 1e9 / up_best;

    /* One stable, machine-parseable row per N. */
    printf("N=%d threads=1 protocol=execute-only CLOCK_MONOTONIC min over %d rounds x %ld iters "
           "warmup=%ld | checksum=%.6f "
           "project.plan_ns=%.1f project.ns_per_exec=%.3f project.throughput=%.1f "
           "upstream.plan_ns=%.1f upstream.ns_per_exec=%.3f upstream.throughput=%.1f "
           "ratio.project_vs_upstream=%.3f\n",
           N, rounds, iters, warmup, (double)checksum,
           proj_plan_ns, proj_best, proj_throughput,
           up_plan_ns, up_best, up_throughput,
           proj_best / up_best);

    fft_plan_destroy(proj);
    fftwf_destroy_plan(up_plan);
    free(io);
    fftwf_free(up);
    return 0;
}

int main(void) {
    printf("# bench_family — post-promote Trunk mainline family golden (VER-FFT-004 refresh)\n");
    printf("# project = default unwindowed fft_execute (JIT on, no FFT_JIT_DISABLE, no gauss)\n");
    printf("# upstream = unpatched FFTW MEASURE (comparison target, CON-FFT-003)\n");
    printf("# knobs: N<2048 WARMUP=%ld ITERS=%ld ROUNDS=%d; N>=2048 WARMUP=%ld ITERS=%ld ROUNDS=%d\n",
           SMALL_WARMUP, SMALL_ITERS, SMALL_ROUNDS, LARGE_WARMUP, LARGE_ITERS,
           LARGE_ROUNDS);
    int rc = 0;
    for (int i = 0; i < FAMILY_COUNT; i++) {
        rc |= bench_one(FAMILY_N[i]);
    }
    if (rc != 0) {
        fprintf(stderr, "bench_family: FAILED (%d)\n", rc);
        return 1;
    }
    printf("# bench_family: PASS (N=%d..%d family)\n", FAMILY_N[0],
           FAMILY_N[FAMILY_COUNT - 1]);
    return 0;
}
