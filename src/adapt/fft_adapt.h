#ifndef MEGAKERNEL_FFT_ADAPT_H
#define MEGAKERNEL_FFT_ADAPT_H

/*
 * src/adapt — secondary adaptation surface (internal, NOT public ABI).
 *
 * Fixed-N, in-place, plan/execute alignment over the vendored FFTW 1D
 * complex path (ARCH-FFT-007).  This header is private to the product tree;
 * the public contract lives in include/fft.h (API-FFT-001).
 */

#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fft_adapt_plan fft_adapt_plan;

/* Create a fixed-N in-place forward plan.  NULL on invalid N or OOM. */
fft_adapt_plan *fft_adapt_plan_create(int N);

/* Execute in-place on caller-owned io[0..N-1]. */
void fft_adapt_execute(fft_adapt_plan *plan, float complex *io);

/* Release an adapt plan.  Safe to pass NULL. */
void fft_adapt_plan_destroy(fft_adapt_plan *plan);

#ifdef __cplusplus
}
#endif

#endif /* MEGAKERNEL_FFT_ADAPT_H */
