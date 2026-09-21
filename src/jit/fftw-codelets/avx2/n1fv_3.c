/* topic jit-codelet-weld: AVX2 SIMD_HEADER wrapper (avx2-dir style, vendored
 * lock).  The vendored config.h is the fp32 build (FFTW_SINGLE); this topic
 * compiles the DOUBLE codelets, so pre-include + undef — the config.h
 * include guard makes the later inclusions no-ops (vendored file untouched). */
#define SIMD_HEADER "simd-support/simd-avx2.h"
#include "../common/n1fv_3.c"
