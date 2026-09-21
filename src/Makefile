# src/Makefile - build the fixed-N 1D complex FFT product + verification tests.
#
# Product layout (ARCH-FFT-006):
#   include/            public C ABI (fft.h)
#   src/host/           plan cache + ABI dispatch
#   src/adapt/          fixed-N in-place wrap over vendored FFTW
#   src/jit/            Xbyak N=2^n family JIT kernel (ARCH-FFT-008/009/011)
#   third_party/fftw/   vendored FFTW 3.3.10 (static, single precision)
#   tests/              VER-FFT-001..003 at N=1024 + opt-in gauss / inverse
#
# The product links the vendored static libfftw3f.a; it does NOT dynamic-link a
# system libfftw3 (ARCH-FFT-007 / CHR-FFT-002).  N = 2^n (n=0..15, i.e.
# N in {1,2,4,...,32768}) executes the family JIT kernel (ARCH-FFT-008 /
# ARCH-FFT-011 / ARCH-FFT-016); every other N keeps the FFTW ESTIMATE adapt
# path.

CC       ?= cc
CXX      ?= g++
CFLAGS   ?= -O2 -std=c11 -Wall -Wextra -D_POSIX_C_SOURCE=200809L
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
CPPFLAGS = -I../include -I. -I./adapt -I./jit -I../third_party/fftw/api
LDLIBS   = -lm -pthread

FFTW_LIB = ../third_party/fftw/.libs/libfftw3f.a

# topic jit-codelet-weld (BEH-FFT-016): vendored AVX2+FMA codelet subset under
# jit/fftw-codelets/ — read-only copies of third_party/fftw dft codelet bodies
# (md5-identical) + avx2 SIMD_HEADER wrappers + a double-precision config shim
# (config.h include_next + #undef FFTW_SINGLE; the vendored fp32 build and
# third_party/ itself stay untouched).  fftw-codelets must come FIRST on the
# include chain so the shim wins; the plan layer feature-gates AVX2+FMA3 at
# runtime and keeps the promoted pad route without them.
# topic jit-anyN-fwd-inv R1 (API-FFT-013): radix 11/14 vendored from the
# in-repo upstream (common/n1fv_{11,14}.c byte-for-byte + avx2 wrappers);
# 11/14 stages walk the WSK_N1 n1fv+TW-sweep mode (radix-13 pattern — no
# upstream t1fv_11/t1fv_14).
WELD_N1 := 3 5 6 7 9 10 11 12 13 14 15 20 25
WELD_T1 := 3 5 6 7 9 10 12 15 20 25
WELD_CODELETS := $(foreach r,$(WELD_N1),jit/cn1fv_$(r).o) \
                 $(foreach r,$(WELD_T1),jit/ct1fv_$(r).o) \
                 jit/hand_cn1fv_11.o jit/hand_ct1fv_11.o
WELD_INC := -I./jit/fftw-codelets -I../third_party/fftw
WELD_CFLAGS := -O3 -mavx2 -mfma -DARCH_PREFERS_FMA

LIB_OBJS = host/plan_cache.o adapt/fft_inplace.o jit/fft_jit.o jit/weld_glue.o $(WELD_CODELETS)
LIB      = libfft_megakernel.a

TESTS    = test_linearity test_impulse test_shift test_family test_gauss_window test_large_n_family test_inverse test_conv test_linear_conv test_conv_dispatch test_fp64

# fp64 test oracle links the SYSTEM double-precision libfftw3 (unpatched
# fftw_*); the product lib and all other tests stay on the vendored
# single-precision libfftw3f.a only (ARCH-FFT-007 / CHR-FFT-002).
FFTW_DOUBLE_LIB := /usr/lib/x86_64-linux-gnu/libfftw3.so.3

all: $(LIB)

%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# jit/fft_jit.cpp includes jit/weld_codelets.h (which resolves the double-mode
# config shim), so it needs the weld include chain and the validated AVX2+FMA
# codegen flags of the topic build.
jit/fft_jit.o: jit/fft_jit.cpp jit/fft_jit.h jit/weld_codelets.h
	$(CXX) $(CXXFLAGS) -mavx2 -mfma $(CPPFLAGS) $(WELD_INC) -c $< -o $@

# codelet registry + the single JIT->C stage runner (immintrin: needs AVX2+FMA)
jit/weld_glue.o: jit/weld_glue.c jit/weld_codelets.h
	$(CC) $(WELD_CFLAGS) $(WELD_INC) -I./jit -c $< -o $@

jit/cn1fv_%.o: jit/fftw-codelets/avx2/n1fv_%.c
	$(CC) $(WELD_CFLAGS) $(WELD_INC) -c $< -o $@

jit/ct1fv_%.o: jit/fftw-codelets/avx2/t1fv_%.c
	$(CC) $(WELD_CFLAGS) $(WELD_INC) -c $< -o $@

# jit-anyN-fwd-inv: the RETIRED hand-written radix-11 pair (hand/{n1fv,t1fv}_11.c,
# the R0 fallback while the in-repo upstream n1fv_11 was missed) — registered
# ONLY under FFT_ANYN_LEAVES=hand (the R0 reproduction knob); the default pool
# is the vendored set above.
jit/hand_cn1fv_11.o: jit/fftw-codelets/hand/n1fv_11.c
	$(CC) $(WELD_CFLAGS) $(WELD_INC) -I./jit -c $< -o $@

jit/hand_ct1fv_11.o: jit/fftw-codelets/hand/t1fv_11.c
	$(CC) $(WELD_CFLAGS) $(WELD_INC) -I./jit -c $< -o $@

$(LIB): $(LIB_OBJS)
	ar rcs $@ $^

# The library contains one C++ TU (src/jit/fft_jit.cpp), so the final link is
# driven by $(CXX); the C test TU is compiled separately with $(CC).
../tests/test_%: ../tests/test_%.c ../tests/test_common.h $(LIB) $(FFTW_LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) -I../tests -c $< -o $@.o
	$(CXX) $(CXXFLAGS) -o $@ $@.o $(LIB) $(FFTW_LIB) $(LDLIBS)

# test_fp64 alone links the system double-precision libfftw3.so.3 (its
# fp64 oracle); the product lib and every other test stay fp32-only.
../tests/test_fp64: ../tests/test_fp64.c ../tests/test_common.h $(LIB) $(FFTW_LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) -I../tests -c $< -o $@.o
	$(CXX) $(CXXFLAGS) -o $@ $@.o $(LIB) $(FFTW_LIB) $(FFTW_DOUBLE_LIB) $(LDLIBS)

test: $(LIB) $(TESTS:%=../tests/%)
	@set -e; for t in $(TESTS); do echo "=== $$t ==="; ../tests/$$t; done

clean:
	rm -f $(LIB_OBJS) $(LIB) $(TESTS:%=../tests/%) $(TESTS:%=../tests/%.o)

.PHONY: all test clean
