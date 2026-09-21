# jit/ - Xbyak emitter + twiddle LUT (src/jit)

JIT kernel for the N=2^n family (n=0..10, i.e. N in {1,2,4,...,1024})
1D complex in-place forward FFT (`{#ARCH-FFT-008}` / `{#ARCH-FFT-009}` /
`{#ARCH-FFT-011}`; promoted from `poc/jit-twiddle-imm-lut`, generalizing
`poc/jit-n1024-xbyak`).

- `fft_jit.h` / `fft_jit.cpp` - plan-time Xbyak emission of a radix-2 DIT
  kernel (in-place bit-reversal + log2(N) stages, AVX, natural-order out),
  with the twiddle specialization chosen per length:
  - n=0 (N=1) degenerate identity kernel;
  - n=1..4 (N=2..16) fully-unrolled register-resident immediate kernels;
  - n=5..8 (N=32..256) loop immediate path (twiddles as a rip-relative
    constant pool, no execute-resident LUT);
  - n=9..10 (N=512,1024) plan-time twiddle LUT whose base addresses are
    baked into the machine code.
  `fft_jit_kernel_create` returns NULL for N outside the family window
  (N<1, non-power-of-two, N>1024) or missing AVX (fail-fast, DESIGN failure
  modes).  Opt-in `fft_jit_kernel_create_gauss_window` adds a plan-time
  Gaussian LUT (`σ = N/8`, ∑G=1), stored separately from the twiddle LUT
  (`{#ARCH-FFT-015}`), and fuses `x[n]·G[n]` into the first stage.
- `xbyak/` - vendored Xbyak headers (BSD-3-Clause, herumi/Xbyak).

## Xbyak provenance

Vendored from the oneDNN-bundled copy of Xbyak
(`.../onednn/src/cpu/x64/xbyak/`), version `0x7050` (7.05) as reported by
`Xbyak::VERSION`.  Headers copied verbatim; license text preserved in each file:

- `xbyak.h`, `xbyak_mnemonic.h`, `xbyak_bin2hex.h`

Upstream: <https://github.com/herumi/xbyak> (modified BSD-3-Clause,
Copyright (c) 2007 MITSUNARI Shigeo / Intel Corporation).
