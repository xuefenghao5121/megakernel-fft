# Vendored upstream: FFTW 3.3.10

> schema: ndf-vendor-record/v1
> project: megakernel-fft
> vendor_root: third_party/fftw/

## Provenance

| Field | Value |
|-------|-------|
| name | FFTW (Fastest Fourier Transform in the West) |
| version | 3.3.10 |
| upstream_url | https://fftw.org |
| upstream_tag | `fftw-3.3.10` |
| upstream_git_sha | `7184fc796279cfa70e4ba62519ac2938054584e6` |
| tarball_url | https://fftw.org/fftw-3.3.10.tar.gz |
| tarball_sha256 | `56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467` |
| license | GPL-2.0-or-later (see `COPYING`, GNU GPL v2; FFTW dual GPL/commercial) |
| vendored_on | 2026-08-28 |
| vendored_for | megakernel-fft genesis Trunk — 1D complex in-place FFT baseline (API-FFT-001) |

## Scope of use

Vendored as the first-skeleton transform kernel for the fixed-N 1D complex in-place FFT
product ([[CHR-FFT-001]] / [[ARCH-FFT-007]]). The product wraps only the **1D complex
forward DFT** path (`fftwf_plan_guru_dft` + `fftwf_execute_dft`) behind the `include/` C ABI.

Explicitly **out of scope** for this cycle ([[CHR-FFT-002]]): `fftw-patient`
PATIENT/Wisdom codelet caching, multi-dimensional row/column dispatch, and
`fftw_*` double / `fftwl_*` long-double precisions as product kernels.

## Build note

Built statically (single precision) so the product does **not** dynamic-link a system
`libfftw3` as its only kernel:

```text
./configure --disable-shared --enable-static --disable-fortran --disable-doc \
            --disable-threads --disable-openmp --disable-mpi --enable-single
make
```

Static archive: `third_party/fftw/.libs/libfftw3f.a`.  Build artifacts are gitignored
(see `third_party/fftw/.gitignore`); only the pristine upstream tree is committed.
