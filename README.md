# megakernel-fft

A fixed-N 1D complex FFT library built **on top of open-source FFTW** with
runtime JIT kernels and fused composite operators.

- **JIT-specialized kernels**: every plan emits a straight-line kernel for its
  exact transform size — twiddle constants folded, no loops — selected at plan
  time by a search over decompositions, tiling shapes, fusion placement and
  schedule variants.
- **Megakernel fusion**: composite operators run as one welded kernel. The
  pointwise multiply, conjugate and `1/N` scaling of `FFT → ×H → IFFT` are sunk
  into load/store boundaries, so the intermediate spectrum never materializes
  in memory.
- **fp32 and fp64** families, power-of-two and (fp64) arbitrary sizes, with
  FFTW's own generated codelets (`n1fv_*`/`t1fv_*`, radices 3…25) reused as the
  non-power-of-two leaves.

## Relationship to FFTW

This library **vendors FFTW 3.3.10** (`third_party/fftw/`, single-precision
build) and **reuses** it in three ways:

| role | what | link |
|---|---|---|
| fallback | non-power-of-two **fp32** transforms run the vendored FFTW "adapt" path | static `libfftw3f.a`, linked into the product |
| leaves | FFTW's generated `n1fv`/`t1fv` codelets are byte-for-byte vendored and compiled with AVX2+FMA as the radix-3…25 leaves of the **fp64** kernels | compiled into the product |
| oracle | the **fp64 test binary** links the *system* double-precision `libfftw3.so.3` (unpatched) as the correctness oracle | test-only, not the product |

The product library itself does **not** dynamically link a system FFTW. If your
application already uses FFTW, you can link this library alongside it — the
symbols are fully namespaced (`fft_*`, no `fftw_*` clash), so both can coexist
in one process.

## Requirements

- x86-64, **AVX2 + FMA3** (runtime feature-checked; power-of-two paths keep
  graceful non-FMA code paths, but the non-power-of-two fp64 engine requires
  AVX2+FMA and returns `NULL` without it)
- g++ (C++17) + cc, `make`
- the vendored FFTW build (see below)

## Build

```sh
# 1. Build the vendored single-precision FFTW once (static lib, no docs):
cd third_party/fftw
./configure --enable-single --disable-doc
make
cd ../..

# 2. Build the product library:
make -C src                 # -> src/libfft_megakernel.a

# 3. (optional) run the correctness suites:
make -C src test            # 11 test binaries; test_fp64 additionally
                            # links /usr/lib/x86_64-linux-gnu/libfftw3.so.3
```

## Linking into your application

```sh
g++ -O2 -march=native -I<repo>/include app.c \
    <repo>/src/libfft_megakernel.a \
    <repo>/third_party/fftw/.libs/libfftw3f.a \
    -lm -pthread
# (the library contains one C++ TU; link with g++ — or add -lstdc++ when
#  driving the link with cc)
```

## API

All plans share one type (`fft_plan *`), are created per size/kind, executed
**in place** on caller-owned buffers, and released with `fft_plan_destroy`.
Output is in natural order (matching upstream FFTW 1D). `plan-once,
execute-many`; plans for equal parameters are cached. Every create entry
returns `NULL` on invalid parameters — never a silent fallback to a different
transform.

| entry | domain | one-line semantics |
|---|---|---|
| `fft_plan_create(N)` / `fft_execute` / `fft_plan_destroy` | fp32, N = 2^n ≤ 32768 (else vendored-FFTW adapt) | default unwindowed forward DFT |
| `fft_plan_create_gauss_window(N)` | fp32, family | fused plan-time Gaussian window (σ = N/8, ΣG = 1) |
| `fft_plan_create_inverse(N)` | fp32, family | conjugate-trick inverse (same `fft_execute` symbol) |
| `fft_plan_create_conv(N, h)` | fp32, family | fused cyclic convolution |
| `fft_plan_create_linear_conv(N, h, nh)` | fp32, family | block-pinned overlap-save linear convolution |
| `fft_plan_create_conv_dispatch(nx, h, nh)` | fp32 | whole-sequence linear convolution, runtime direct/OLS dispatch |
| `fft_plan_create_fp64(N)` / `fft_execute_fp64` | fp64, **any N ≤ 32768 with prime factors in {2,3,5,7,11,13}** | forward DFT |
| `fft_plan_create_fp64_inverse(N)` / `fft_execute_fp64_inverse` | fp64, same domain | inverse (boundary-sunk conjugate trick, `1/N` included) |
| `fft_plan_create_conv_fp64(N, h)` / `fft_execute_conv_fp64` | fp64, **any N ≤ 32768** | fused cyclic convolution |
| `fft_plan_create_linear_conv_fp64(N, h, nh)` / `fft_execute_linear_conv_fp64` | fp64, N = 2^n block | block-pinned overlap-save linear convolution |
| `fft_plan_create_windowed_batch(N, B)` / `fft_execute_windowed_batch(plan, io, stride)` | fp32, family N; B blocks | batched windowed forward, all B blocks in one call |

**N domains in detail**

- **fp32 family**: N = 2^n, n ≤ 15. Other N take the vendored FFTW adapt path
  (correct, but not JIT-accelerated).
- **fp64 forward / inverse**: any N ≤ 32768 whose prime factors all lie in
  {2,3,5,7,11,13} (e.g. 5850 = 2·3²·5²·13 ✓, 3352 = 8·419 ✗ — 419 is not
  covered → `NULL`). Power-of-two N uses the family kernel unchanged.
- **fp64 cyclic convolution**: any N ≤ 32768 — uncovered prime factors take a
  zero-pad + wraparound-correction route that is **exact** (never an
  approximate convolution).

## Examples

### 1. fp32 default forward

```c
#include <fft.h>
#include <complex.h>

void fwd1024(float complex *io) {
    fft_plan *p = fft_plan_create(1024);
    if (!p) return;                     /* OOM only for a legal N */
    fft_execute(p, io);                 /* io: 1024 samples in, spectrum out */
    fft_plan_destroy(p);
}
```

### 2. fp64 forward, arbitrary size

```c
fft_plan *p = fft_plan_create_fp64(5850);
if (!p) { /* N not in the covered set, or no AVX2+FMA */ }
fft_execute_fp64(p, io);                /* double complex io[5850] */
fft_plan_destroy(p);
```

### 3. fp64 inverse (normalized)

```c
fft_plan *p = fft_plan_create_fp64_inverse(N);
fft_execute_fp64_inverse(p, io);        /* io ← DFT⁻¹(io); the 1/N is included */
fft_plan_destroy(p);
```

### 4. fp64 cyclic convolution — the flagship

One `fft_execute_conv_fp64` call does `forward → pointwise ×H → inverse`
as a single fused kernel (the intermediate spectrum transits the buffer
exactly once). `h` is copied at plan time and not modified.

```c
fft_plan *p = fft_plan_create_conv_fp64(N, h);   /* double complex h[N] */
if (!p) return;                        /* bad N / NULL h / no AVX2+FMA */
fft_execute_conv_fp64(p, io);           /* io ← x ⊛ h (cyclic), in place */
fft_plan_destroy(p);
```

### 5. fp64 linear convolution (overlap-save)

Block-pinned OLS: each execute consumes one fresh block of N samples and
writes N−nh+1 valid linear-convolution samples to the **front** of `io`; the
trailing nh−1 positions are an overlap/discard zone maintained inside the
plan. Feed successive blocks and concatenate the valid fronts.

```c
fft_plan *p = fft_plan_create_linear_conv_fp64(N, h, nh);  /* 0 < nh <= N */
for (;;) {
    /* fill io[0..N-1] with the next input block */
    fft_execute_linear_conv_fp64(p, io);
    /* consume io[0..N-nh] as valid output */
}
fft_plan_destroy(p);
```

### 6. Batched windowed forward

One plan for B independent blocks; one execute processes all of them.

```c
fft_plan *p = fft_plan_create_windowed_batch(N, B);   /* fp32, family N */
fft_execute_windowed_batch(p, io, N);  /* block b at io + b*N; stride >= N */
fft_plan_destroy(p);
```

### Error handling and ownership

- Every create returns `NULL` on illegal parameters (unsupported N, `NULL h`,
  `nh` out of range, emit/OOM). There is no silent fallback to a different
  transform — check the return value.
- Buffers are always caller-owned; the kernels never allocate or free `io`.
- `fft_execute*` on a `NULL` plan or `NULL` io is a no-op.
- Plans are refcounted and cached per (kind, size); `fft_plan_destroy(NULL)`
  is safe.

## Performance posture

Single-thread, execute-only, vs system SIMD FFTW (double) with FTZ/DAZ
hygiene, 5-roll plan brackets and data-window guards:

| workload (fp64 fused cyclic convolution) | vs FFTW composite (2×FFT + pointwise) | plan cost |
|---|---|---|
| N = 3352 | **0.57×** (1.8× faster) | ~60× cheaper |
| N = 5850 / 20000 | **0.72–0.76×** | 45–147× cheaper |
| N = 154 (2·7·11) | **0.72×** | — |
| power-of-two 64–32768 | 0.84–1.05× | 42–140× cheaper |

Batched short windowed transforms (N ≤ 32, B = 1024) run at **0.42–0.53×** of
a plan-once FFTW loop. fp64 forward/inverse on covered arbitrary N sit in the
**1.03–1.40×** band; the bare-DFT mid-band (512–8192) is the known weak range
(1.25–1.5× — FFTW's hand-tuned scheduling territory). Plan-time search is the
killer feature for plan-frequent workloads: FFTW MEASURE planning costs
hundreds of ms, this library plans in milliseconds.

## Design notes

- **Plan-time search** picks decomposition (mixed-radix incl. codelet leaves),
  tiling, fusion placement and schedule variants by actual measurement at plan
  time — the search space includes choices FFTW's planner never considers
  (fusion boundaries).
- **Welded composites**: the fusion saves the pointwise/conjugate/scaling
  sweeps entirely (≈3–5× over the unfused three-stage composition measured
  in-process).
- **FFTW codelet leaves**: `src/jit/fftw-codelets/` holds byte-for-byte copies
  of FFTW's generated `n1fv_*`/`t1fv_*` bodies compiled with AVX2+FMA; the
  retired hand-written radix-11 pair remains behind `FFT_ANYN_LEAVES=hand` as
  a fallback-mode reproduction knob.
- **Debug knobs** (plan-time only, not a stable interface): `FFT_SCHED_ORDER`
  (force decomposition), `FFT_SCHED_SEARCH=off` (skip search), `FFT_SCHED_LOG`
  (diagnostics), `FFT_ANYN_LEAVES` (leaf pool selection).

## Testing

```sh
make -C src test
```

runs 11 suites: linearity / impulse / shift / family / gauss / large-N /
inverse / conv / linear-conv / conv-dispatch (fp32) and `test_fp64` (fp64
forward, inverse, conv, linear conv, batched — oracles against the system
double-precision FFTW). Requires `/usr/lib/x86_64-linux-gnu/libfftw3.so.3`
for the fp64 oracle binary only.

## License

The vendored FFTW under `third_party/fftw` is GPLv2+; the product links it,
so this tree is effectively GPL-licensed. The borrowed codelet bodies carry
FFTW's copyright and license.
