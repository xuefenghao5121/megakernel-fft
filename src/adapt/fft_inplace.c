/*
 * src/adapt/fft_inplace.c — fixed-N in-place wrap over vendored FFTW 1D.
 *
 * ARCH-FFT-007: first skeleton is the upstream FFTW 1D complex path, adapted
 * to a fixed-N, in-place, plan/execute ABI.  This is the ONLY place the
 * vendored FFTW headers/functions are referenced (adapt surface).
 *
 * The plan is created once with FFTW_ESTIMATE (no PATIENT/Wisdom — see
 * CHR-FFT-002) using the "guru" new-array interface so the caller's buffer
 * is passed at execute time, not embedded in the plan (API-FFT-001).
 */

#include "fft_adapt.h"

#include <fftw3.h>
#include <stdlib.h>

struct fft_adapt_plan {
    fftwf_plan fftw; /* vendored FFTW single-precision plan (new-array execute) */
    int N;           /* fixed transform size */
};

fft_adapt_plan *fft_adapt_plan_create(int N) {
    if (N < 1) {
        return NULL;
    }

    fft_adapt_plan *plan = (fft_adapt_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        return NULL;
    }

    /*
     * Guru interface with NULL in/out arrays: yields a plan that must be
     * executed via fftwf_execute_dft(plan, io, io).  FFTW_ESTIMATE performs
     * no data-touching measurement at plan time, so NULL is valid.
     */
    fftwf_iodim dims[1];
    dims[0].n = N;
    dims[0].is = 1; /* contiguous input stride */
    dims[0].os = 1; /* contiguous output stride */

    plan->fftw = fftwf_plan_guru_dft(
        /* rank= */ 1, dims,
        /* howmany_rank= */ 0, /* howmany_dims= */ NULL,
        /* in= */ NULL, /* out= */ NULL,
        /* sign= */ FFTW_FORWARD,
        /* flags= */ FFTW_ESTIMATE);

    if (plan->fftw == NULL) {
        free(plan);
        return NULL;
    }
    plan->N = N;
    return plan;
}

void fft_adapt_execute(fft_adapt_plan *plan, float complex *io) {
    if (plan == NULL || io == NULL) {
        return;
    }
    /*
     * float complex (C11) is layout-compatible with fftwf_complex (float[2]).
     * In-place: same pointer for input and output.
     */
    fftwf_execute_dft(plan->fftw, (fftwf_complex *)io, (fftwf_complex *)io);
}

void fft_adapt_plan_destroy(fft_adapt_plan *plan) {
    if (plan == NULL) {
        return;
    }
    fftwf_destroy_plan(plan->fftw);
    free(plan);
}
