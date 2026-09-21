/*
 * tests/bench_baseline.c — GENESIS baseline measure harness (VER-FFT-004)
 *
 * Measures the fixed-N 1D complex in-place execute path, single-thread, N=1024,
 * for two subjects under one protocol:
 *
 *   subject "project"  — public ABI include/fft.h (fft_plan_create + fft_execute),
 *                        which wraps the vendored FFTW 1D path via src/adapt with
 *                        FFTW_ESTIMATE (plan-once / execute-many, ARCH-FFT-001/007).
 *   subject "upstream" — unpatched upstream FFTW MEASURE (fftwf_plan_dft_1d with
 *                        FFTW_MEASURE), the DEC-named comparison target (name-only
 *                        external reference, [[CON-FFT-003]]). Not the product kernel.
 *
 * Protocol:
 *   - deterministic input (LCG, same family as tests/test_common.h)
 *   - single thread (no fftwf_init_threads / OpenMP)
 *   - plan creation timed once per subject (reported, not part of execute number)
 *   - warmup 10k executes per subject
 *   - execute-only wall time via clock_gettime(CLOCK_MONOTONIC)
 *   - R rounds x ITERS executes, min per-execute retained (noise floor)
 *   - throughput = 1 / per-execute seconds
 *
 * Note: the in-place DFT is iterated many times on the same buffer; DFT(DFT(x))
 * = N*reverse(x), so magnitudes grow and eventually overflow. Execute timing is
 * data-independent for FFTW (fixed codelet path), so the checksum is taken from
 * the deterministic *input* (before any transform) purely as a reproducibility
 * marker — the repeated in-place outputs are intentionally not re-read.
 *
 * Buffer note (fairness, documented): the product ABI is caller-allocated and makes
 * no alignment guarantee (API-FFT-001), so "project" uses plain malloc; "upstream"
 * uses fftwf_malloc (the aligned path FFTW's MEASURE planner expects).
 */

#include "fft.h" /* public ABI — project kernel */

#include <fftw3.h> /* upstream FFTW reference */

#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024
#define WARMUP 10000L
#define ITERS 200000L
#define ROUNDS 9

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

int main(void) {
    uint32_t rng = 0x5eed1234u;

    float complex *io = (float complex *)malloc((size_t)N * sizeof(float complex));
    float complex *up = (float complex *)fftwf_malloc((size_t)N * sizeof(float complex));
    if (io == NULL || up == NULL) {
        fprintf(stderr, "OOM\n");
        return 2;
    }
    for (int i = 0; i < N; i++) {
        float re = 2.0f * randf01(&rng) - 1.0f;
        float im = 2.0f * randf01(&rng) - 1.0f;
        io[i] = re + im * I;
        up[i] = re + im * I;
    }

    /* Deterministic input checksum (data-independent execute timing; see note below). */
    float checksum = 0.0f;
    for (int i = 0; i < N; i++) {
        checksum += crealf(io[i]) + cimagf(io[i]);
    }

    /* --- project kernel: plan once (FFTW_ESTIMATE via src/adapt) --- */
    double t0 = now_ns();
    fft_plan *proj = fft_plan_create(N);
    double proj_plan_ns = now_ns() - t0;
    if (proj == NULL) {
        fprintf(stderr, "fft_plan_create(N=%d) failed\n", N);
        return 2;
    }

    /* --- upstream FFTW MEASURE: plan once (FFTW_MEASURE) --- */
    t0 = now_ns();
    fftwf_plan up_plan = fftwf_plan_dft_1d(N, (fftwf_complex *)up, (fftwf_complex *)up,
                                            FFTW_FORWARD, FFTW_MEASURE);
    double up_plan_ns = now_ns() - t0;
    if (up_plan == NULL) {
        fprintf(stderr, "fftwf_plan_dft_1d(FFTW_MEASURE, N=%d) failed\n", N);
        return 2;
    }

    /* --- warmup --- */
    for (long i = 0; i < WARMUP; i++) {
        fft_execute(proj, io);
    }
    for (long i = 0; i < WARMUP; i++) {
        fftwf_execute_dft(up_plan, (fftwf_complex *)up, (fftwf_complex *)up);
    }

    /* --- execute-only timing (min over rounds = noise floor) --- */
    double proj_best = 1e18;
    double up_best = 1e18;
    for (int r = 0; r < ROUNDS; r++) {
        double s = now_ns();
        for (long k = 0; k < ITERS; k++) {
            fft_execute(proj, io);
        }
        double e = now_ns();
        double ns = (e - s) / (double)ITERS;
        if (ns < proj_best) {
            proj_best = ns;
        }
    }
    for (int r = 0; r < ROUNDS; r++) {
        double s = now_ns();
        for (long k = 0; k < ITERS; k++) {
            fftwf_execute_dft(up_plan, (fftwf_complex *)up, (fftwf_complex *)up);
        }
        double e = now_ns();
        double ns = (e - s) / (double)ITERS;
        if (ns < up_best) {
            up_best = ns;
        }
    }

    double proj_throughput = 1e9 / proj_best; /* executes / second */
    double up_throughput = 1e9 / up_best;

    printf("N=%d threads=1\n", N);
    printf("protocol=execute-only wall-clock (CLOCK_MONOTONIC) min over %d rounds x %ld iters\n",
           ROUNDS, ITERS);
    printf("warmup=%ld iters/subject\n", WARMUP);
    printf("checksum=%.6f\n", (double)checksum);
    printf("project.plan_create_ns=%.1f\n", proj_plan_ns);
    printf("project.ns_per_exec=%.3f\n", proj_best);
    printf("project.throughput_exec_per_s=%.1f\n", proj_throughput);
    printf("upstream.plan_create_ns=%.1f\n", up_plan_ns);
    printf("upstream.ns_per_exec=%.3f\n", up_best);
    printf("upstream.throughput_exec_per_s=%.1f\n", up_throughput);
    printf("ratio.project_vs_upstream=%.3f\n", proj_best / up_best);

    fft_plan_destroy(proj);
    fftwf_destroy_plan(up_plan);
    free(io);
    fftwf_free(up);
    return 0;
}
