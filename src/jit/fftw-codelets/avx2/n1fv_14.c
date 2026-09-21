/* topic jit-anyN-fwd-inv R1 (DELTA F3, L-S): vendored upstream radix-14
 * leaf — common/n1fv_14.c is a byte-for-byte copy of
 * third_party/fftw/dft/simd/common/n1fv_14.c (md5-aligned with the R0
 * 21-leaf precedent; upstream is read-only, never patched in place).
 * AVX2 SIMD_HEADER wrapper in the vendored avx2-dir style; the fp32
 * config.h shim (fftw-codelets/config.h) selects the DOUBLE compile. */
#define SIMD_HEADER "simd-support/simd-avx2.h"
#include "../common/n1fv_14.c"
