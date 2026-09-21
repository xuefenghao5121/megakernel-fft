#ifndef MEGAKERNEL_FFT_FFT_H
#define MEGAKERNEL_FFT_FFT_H

/*
 * megakernel-fft — public C ABI (API-FFT-001 / API-FFT-003 / API-FFT-004)
 *
 * Fixed-N (power of two) 1D complex in-place FFT.  Plan once, execute many.
 * Default fft_plan_create / fft_execute is the unwindowed forward DFT
 * (API-FFT-001 / BEH-FFT-001).  Opt-in fft_plan_create_gauss_window
 * (API-FFT-003) fuses a plan-time Gaussian window.  Opt-in
 * fft_plan_create_inverse (API-FFT-004) wraps the family forward kernel
 * with the conjugate trick; default create/execute stay unwindowed
 * forward.  Opt-in fp64 forward (draft {#API-FFT-008}) adds
 * fft_plan_create_fp64 / fft_execute_fp64 for `double complex` buffers.
 *
 * topic jit-joint-search-batched: the ONLY API change of this topic adds
 * the opt-in batched Gaussian-windowed forward entry points
 * fft_plan_create_windowed_batch / fft_execute_windowed_batch (draft
 * {#API-FFT-010}); every existing symbol and contract above is unchanged
 * and only regressed.
 * Topic jit-fp64-conv-fuse adds the opt-in fp64 cyclic convolution (draft
 * {#API-FFT-011}): fft_plan_create_conv_fp64 / fft_execute_conv_fp64 —
 * fused boundary-sunk conv weld on the promoted pd SchedEmitter base plus
 * the arbitrary-N semantic-preservation path.  All fp32 contracts and the
 * fp64 forward stay unchanged (regression-only).
 * Topic jit-fp64-inverse (amend resume) adds the opt-in fp64 inverse
 * (draft {#API-FFT-009} / {#BEH-FFT-010}): fft_plan_create_fp64_inverse /
 * fft_execute_fp64_inverse — boundary-sunk conjugate / 1-N fused inverse
 * on the promoted forward machinery.  Every existing symbol and contract
 * above is unchanged and only regressed.
 * Topic jit-fp64-linear-conv adds the opt-in fp64 block-pinned OLS LINEAR
 * convolution (draft {#API-FFT-012} / {#BEH-FFT-017}):
 * fft_plan_create_linear_conv_fp64 + fft_execute_linear_conv_fp64 — the
 * promoted fused conv weld driven as an overlap-save block stream with the
 * valid-segment write sunk to the inverse final-store boundary.  Every
 * existing symbol and contract above is unchanged and only regressed.
 */

#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Opaque plan handle.  Callers MUST NOT dereference it or assume it wraps a
 * raw FFTW plan.  Owned and released via fft_plan_create / fft_plan_destroy.
 */
typedef struct fft_plan fft_plan;

/*
 * Plan once for a fixed transform size N (power of two, e.g. 1024).
 *
 * Returns an opaque handle on success, or NULL on invalid N (N < 1) or
 * allocation failure.  Plans for equal N MAY be cached and reused by the
 * host layer (ARCH-FFT-002); the returned handle is refcounted and remains
 * valid until the matching fft_plan_destroy call.
 */
fft_plan *fft_plan_create(int N);

/*
 * Opt-in Gaussian-window plan (API-FFT-003 / BEH-FFT-004 / ARCH-FFT-015).
 *
 * Plan-time G[n] = exp(-(n-N/2)^2 / (2 σ^2)) with σ = N/8, ∑G = 1.
 * Subsequent fft_execute uses the same public symbol; the plan carries
 * the fused kernel.  MUST NOT change fft_plan_create(int N) or default
 * unwindowed semantics (API-FFT-001 / BEH-FFT-001).
 *
 * Returns NULL on N<1, non-family N (not a power of two or N>32768), or
 * emit / OOM / LUT failure.  MUST NOT silently window the adapt path.
 * σ is not an execute argument.  The buffer stays caller-owned.
 */
fft_plan *fft_plan_create_gauss_window(int N);

/*
 * Opt-in inverse plan (API-FFT-004).  Same fft_plan type and the same
 * fft_execute symbol.  HOW = conjugate trick on the already-promoted
 * family forward kernel: io ← conj(io) → forward → io ← conj(io) / N.
 *
 * MUST NOT change fft_plan_create(int N) or default unwindowed-forward
 * semantics.  Family-only (N = 2^n, n≤15 / N≤32768).  Returns NULL on
 * illegal N, emit failure, or OOM.  MUST NOT silently fall back to the
 * default forward path or adapt.
 */
fft_plan *fft_plan_create_inverse(int N);

/*
 * Opt-in cyclic-convolution plan (draft API-FFT-005).  Same fft_plan type
 * and the same fft_execute symbol; MUST NOT invent a second execute symbol.
 *
 * Plan-time: materialize H[k] = FFT(h) by reusing the already-promoted
 * family forward kernel on a scratch copy of h (the caller's h is NOT
 * clobbered).  H is a separate table from the twiddle LUT and the gauss G
 * table (MUST NOT build a fourth twiddle subsystem); no exp at plan time.
 *
 * Execute: forward → pointwise io[k] *= H[k] → IFFT (conjugate trick on the
 * forward kernel) → write back the same io, all in ONE fft_execute call.
 * The intermediate spectrum is NOT written back to the buffer followed by a
 * separate fft_execute.
 *
 * MUST NOT change fft_plan_create(int N) or default unwindowed-forward
 * semantics; MUST NOT change gauss (API-FFT-003) or inverse (API-FFT-004)
 * contracts.  Family-only (N = 2^n, n≤15 / N≤32768).  Returns NULL on
 * illegal N, NULL h, emit failure, or OOM.  MUST NOT silently fall back to
 * the default forward path or adapt.
 */
fft_plan *fft_plan_create_conv(int N, const float complex *h);

/*
 * Opt-in block-pinned overlap-save LINEAR convolution plan (draft
 * API-FFT-006).  Same fft_plan type and the same fft_execute symbol;
 * MUST NOT invent a second execute symbol.  This is LINEAR convolution of a
 * long x against a short kernel h — NOT cyclic convolution
 * (fft_plan_create_conv) and NOT a change to the default forward path.
 *
 * Plan-time: pin the block length N and the kernel h (length nh,
 * 0 < nh ≤ N).  Zero-pad h to length N and materialize H[k] = FFT(h) by
 * reusing the already-promoted family forward kernel on a scratch copy
 * (the caller's h is NOT clobbered).  H is a separate table from the
 * twiddle LUT and the gauss G table (MUST NOT build a fourth twiddle
 * subsystem); no exp at plan time.
 *
 * Execute: forward → pointwise io[k] *= H[k] → IFFT (conjugate trick with
 * /N) → OLS drop the first nh−1 wrap-around-polluted samples and write the
 * N−nh+1 valid linear-conv samples back into the FRONT of the same io.
 * The trailing nh−1 positions are a discard zone (content unpromised).
 * Overlap state is kept inside the plan; callers MUST NOT add/drop samples
 * themselves.  Each fft_execute consumes one length-N block of x and
 * produces one block of valid output independently.
 *
 * nh is an explicit parameter because C cannot infer a kernel's length from
 * a bare `const float complex *`; this lets the plan validate 0 < nh ≤ N.
 *
 * MUST NOT change fft_plan_create(int N) or default unwindowed-forward
 * semantics; MUST NOT change gauss (API-FFT-003), inverse (API-FFT-004),
 * or conv (API-FFT-005) contracts.  Family-only (N = 2^n, n≤15 /
 * N≤32768).  Returns NULL on illegal N, NULL h, nh≤0, nh>N, emit failure,
 * or OOM.  MUST NOT silently fall back to the default forward path, adapt,
 * or cyclic convolution.
 */
fft_plan *fft_plan_create_linear_conv(int N, const float complex *h, int nh);

/*
 * Record-only, re-calibratable dispatch threshold (NOT a stable must SLA;
 * [[CON-POC-001]]).  nh <= FFT_CONV_DISPATCH_NH_DIRECT_MAX selects the
 * direct scalar-convolution leg; otherwise the OLS block leg.  Initial
 * value taken from the conv-size-dispatch R0 table (direct wins only at
 * nh≈2; direct→OLS flip around nh ∈ (2, 8]).  Compile-time tunable; a
 * future fftw-patient cycle re-calibrates at plan time.  MUST NOT be
 * promoted to a must SLA.
 */
#ifndef FFT_CONV_DISPATCH_NH_DIRECT_MAX
#define FFT_CONV_DISPATCH_NH_DIRECT_MAX 8
#endif

/*
 * Opt-in whole-sequence LINEAR convolution with runtime size dispatch
 * (draft API-FFT-007).  Same fft_plan type and the same fft_execute
 * symbol; MUST NOT invent a second execute symbol.
 *
 * This is LINEAR convolution of a FULL sequence x (length nx) against a
 * kernel h (length nh) — y = x * h, output length nx+nh-1.  Unlike
 * fft_plan_create_linear_conv (which pins a BLOCK length N and produces one
 * block of valid output per fft_execute), this plan consumes the WHOLE
 * sequence nx and welds the WHOLE nx+nh-1 output in ONE fft_execute.
 *
 * Plan-time leg selection (record-only threshold, re-calibratable):
 *   - nh <= FFT_CONV_DISPATCH_NH_DIRECT_MAX  → "direct" leg: O(nx·nh)
 *     time-domain scalar convolution (correctness oracle; no FFT, no
 *     kernel, no padding, no transform state).
 *   - otherwise                              → "OLS block" leg: reuses the
 *     promoted fft_plan_create_linear_conv(N_block, h, nh) (API-FFT-006)
 *     with block length N_block = next_pow2(2*nh-1) pinned by nh; the
 *     block overlap-save loop runs INSIDE fft_execute.
 *
 * Execute (one call): y[0..nx+nh-2] = x * h written back into io.  The
 * caller sizes io to at least nx+nh-1 and fills io[0..nx-1] with x on
 * entry.  Overlap/drop state is kept inside the plan; callers MUST NOT
 * add/drop samples themselves.
 *
 * MUST NOT change fft_plan_create(int N) / default unwindowed-forward
 * semantics; MUST NOT change gauss / inverse / conv / linear-conv
 * contracts.  Returns NULL on illegal nx (<=0), nh<=0, nh>nx, NULL h, or
 * (OLS leg) emit/OOM.  MUST NOT silently fall back to default forward,
 * adapt, cyclic convolution, or single-block linear convolution.
 */
fft_plan *fft_plan_create_conv_dispatch(int nx, const float complex *h, int nh);

/*
 * Opt-in fp64 forward plan (API-FFT-008, N-domain REVISED by draft
 * {#API-FFT-013} / {#BEH-FFT-009}).  Signature and every other semantic
 * unchanged; the accepted N domain widens from power-of-two family-only
 * to ANY N <= 32768 within the FACTOR COVERAGE SET S = {2,3,4,5,6,7,8,9,
 * 10,11,12,13,15,16,20,25,32} (S = Trunk weld_factors ∪ the F2 hand
 * radix-11 leaf; prime closure {2,3,5,7,11,13}): every prime factor of N
 * must lie in S (S-smooth).  N = 2^n keeps the existing family kernel
 * path byte-for-byte (regression guard).  Coverage-set edge and
 * uncovered N (incl. primes like 17/19, N > 32768, N < 1) -> NULL,
 * fail-closed: MUST NOT silently fall back to adapt/pad/FFTW/any fp32 or
 * other opt-in plan (unlike the conv entry's auto semantic-preservation
 * route, a plain transform has no zero-pad semantic correction
 * available).  In-set N that cannot form a >=2-stage weld (bare primes,
 * tiny N=1..3) compute on a host wrapper floor (oracle-tolerance exact
 * O(N^2) DFT).  Non-pow2 in-set N requires AVX2+FMA (WSK engine) — NULL
 * without it (fail-fast).
 */
fft_plan *fft_plan_create_fp64(int N);

/*
 * Opt-in fp64 cyclic-convolution plan (draft {#API-FFT-011}).
 * `double complex` buffers, own plan handle + INDEPENDENT execute symbol
 * fft_execute_conv_fp64 (C has no overloading; MUST NOT shove a double
 * buffer through the fp32 fft_plan_create_conv / fft_execute — the fp32
 * conv contract (API-FFT-005) is UNCHANGED).
 *
 * Arbitrary N within the family ceiling (N <= 32768): N = 2^n runs the
 * fused kernel directly; other N take the semantic-preservation path
 * (zero-pad + wraparound correction — adjudicated in this topic against
 * Bluestein, see DESIGN §L-J / ndf/evidence) whose output equals the
 * N-point cyclic convolution y[n] = sum_j x[j] h[(n-j) mod N] within
 * double-precision oracle tolerance.  When 2N-1 exceeds the 32768 ceiling
 * the exact route is main-weld @32768 + tail-correlation weld (up to
 * N = 24576) or multi-block OLA on the same weld (up to N = 32768) —
 * always EXACT, never a silently approximated cyclic convolution.
 *
 * Plan-time: materialize H[k] = FFT(h) by reusing the promoted fp64
 * forward kernel on a scratch copy of h (the caller's h is NOT clobbered).
 * H is a separate table from the twiddle LUT (no fourth twiddle
 * subsystem); no exp at plan time.  Plan-cost is recorded vs FFTW
 * (report-only).
 *
 * Execute: ONE fused call completes forward -> pointwise io[k] *= H[k] ->
 * inverse (conjugate in-register, 1/N sunk to the inverse final store);
 * the intermediate spectrum transits the buffer exactly once (written by
 * the forward final stage, gathered by the inverse stage-0 — no separate
 * pointwise / conjugate / scaling sweeps).  In-place, natural order in/out
 * (io is both input x and output y).  Buffers are caller-owned; the
 * kernel MUST NOT take ownership.  Note: the welded JIT kernel covers
 * transform sizes >= 4; the degenerate conv sizes N = 1 and N = 2 compute
 * the same cyclic convolution directly in the wrapper (a single complex
 * product / 2x2 MAC — no transform is meaningful there).
 *
 * Returns NULL on illegal N (N < 1 or N > 32768 ceiling), NULL h, emit
 * failure, or OOM; MUST NOT silently fall back to the fp64 forward path,
 * the fp32 conv path, adapt, or route to FFTW.
 */
fft_plan *fft_plan_create_conv_fp64(int N, const double complex *h);

/*
 * Execute an opt-in fp64 cyclic-convolution plan in-place on
 * `io[0 .. N-1]` (double complex).  Independent execute symbol (C has no
 * overloading).  One call completes forward -> pointwise -> inverse with
 * the boundary-sunk multiplies (see fft_plan_create_conv_fp64).  `io` is
 * caller-owned; the kernel MUST NOT take ownership.  Calling with a NULL
 * plan / NULL io is a no-op; a non-conv-fp64 plan is a no-op (defensive).
 * MUST NOT be used on an fp32 plan, and the fp32 fft_execute MUST NOT run
 * the fp64 kernel.
 */
void fft_execute_conv_fp64(fft_plan *plan, double complex *io);

/*
 * Opt-in fp64 INVERSE plan (topic jit-fp64-inverse, draft
 * {#API-FFT-009} / {#BEH-FFT-010}).  `double complex` buffer, own plan
 * handle + INDEPENDENT execute symbol fft_execute_fp64_inverse (C has no
 * overloading; MUST NOT shove a double buffer through the fp32
 * fft_plan_create_inverse / fft_execute — the fp32 inverse contract
 * (API-FFT-004) is UNCHANGED).
 *
 * HOW = boundary-sunk conjugate trick on the promoted fp64 forward
 * machinery: IFFT(x) = conj(F(conj(x))) / N computed by ONE fused JIT
 * kernel (N >= 32) whose stage-0 gather conjugates AT THE LOAD BOUNDARY
 * and whose final stage stores conj(.) * (1/N) AT THE STORE BOUNDARY —
 * no independent conjugate / scaling sweeps of the caller's buffer, no
 * fp64 inverse twiddle table, no second emitter (forward twiddles are
 * consumed unchanged).  The N < 32 family floor keeps the host wrapper
 * (forward kernel + independent conj / conj*invN sweeps).  In-place,
 * natural order in / out (output order matches same-N upstream
 * FFTW_BACKWARD / N).
 *
 * fft_plan_create_fp64_inverse: same N-domain revision on API-FFT-009
 * (topic jit-anyN-fwd-inv, draft {#API-FFT-013}) — boundary-sunk
 * conjugate trick, inv64 plan table, N<32 wrapper floor and all other
 * semantics unchanged; the non-pow2 coverage-set档 runs the WSK
 * boundary-conjugate single-transform inverse (entry conj at the
 * stage-0 gather / conjugating permute, exit conj(.)·(1/N) at the final
 * store; output order matches same-N upstream FFTW_BACKWARD / N).
 * Family pow2 N keeps the promoted path byte-for-byte (guard); coverage-
 * set edge and uncovered N -> NULL fail-closed; bare primes / tiny N on
 * the host wrapper floor; non-pow2 in-set N requires AVX2+FMA.
 */
fft_plan *fft_plan_create_fp64_inverse(int N);

/*
 * Execute an opt-in fp64 inverse plan in-place on `io[0 .. N-1]`
 * (double complex).  Independent execute symbol (C has no overloading).
 * One call computes IFFT(io) with the boundary-sunk conjugate / 1/N (see
 * fft_plan_create_fp64_inverse).  `io` is caller-owned; the kernel MUST
 * NOT take ownership.  Calling with a NULL plan / NULL io is a no-op; a
 * non-inverse-fp64 plan is a no-op (defensive).  MUST NOT be used on an
 * fp32 plan, and the fp32 fft_execute / fp64 forward fft_execute_fp64 /
 * fp64 conv fft_execute_conv_fp64 MUST NOT run this kernel.
 */
void fft_execute_fp64_inverse(fft_plan *plan, double complex *io);

/*
 * Opt-in fp64 block-pinned overlap-save LINEAR convolution plan
 * (draft {#API-FFT-012}).  `double complex` buffers, own plan handle +
 * INDEPENDENT execute symbol fft_execute_linear_conv_fp64 (C has no
 * overloading; MUST NOT shove a double buffer through the fp32
 * fft_plan_create_linear_conv / fft_execute — the fp32 linear-conv
 * contract (API-FFT-006) is UNCHANGED; this plan MUST NOT be executed by
 * fft_execute / fft_execute_fp64 / fft_execute_conv_fp64 /
 * fft_execute_fp64_inverse — those are no-ops on it, defensive).
 *
 * This is LINEAR convolution of a long x against a short kernel h — NOT
 * cyclic convolution (fft_plan_create_conv_fp64) and NOT a change to any
 * fp32 path.  The plan pins the FFT BLOCK length N (power of two, family
 * ceiling N <= 32768) and the kernel h (length nh, 0 < nh <= N — nh is an
 * explicit parameter because C cannot infer a kernel's length from a bare
 * `const double complex *`); the STREAM length nx is arbitrary and is NOT
 * a contract parameter (the caller streams blocks via repeated execute).
 *
 * Plan-time: zero-pad h to length N and materialize H[k] = FFT(h_pad) by
 * reusing the promoted fp64 forward weld on a scratch copy (the caller's
 * h is NOT clobbered).  H is a separate table from the twiddle LUT (no
 * fourth twiddle subsystem); no exp at plan time.  The block-N candidates
 * are searched/measured at plan/bench level ({1024, 4096} in this topic's
 * grid).  Plan-cost is recorded vs FFTW (report-only).
 *
 * Execute: ONE fused welded call per block — forward -> pointwise
 * io[k] *= H[k] -> conjugate-trick inverse with 1/N sunk to the inverse
 * final store; the intermediate spectrum does NOT land in main memory as
 * separate sweeps.  Each call CONSUMES one length-N block of x from
 * io[0..N-1] and writes the N-nh+1 valid linear-convolution samples back
 * into the FRONT of the same io (io[0 .. N-nh]); the trailing nh-1
 * positions are a discard zone (content unpromised; not written or
 * written-then-abandoned — adjudicated by measurement, semantics
 * identical).  The OLS add/drop bookkeeping (dropping the first nh-1
 * wrap-around-polluted samples, compacting the valid region) is INSIDE
 * the plan's execute at the store boundary; callers MUST NOT add/drop
 * samples themselves.  Overlap state is kept inside the plan (the
 * retained nh-1 input tail + stream bookkeeping, zero-initialized head
 * prefix); concatenated valid outputs of consecutive calls equal the
 * linear convolution of the concatenated input stream.  Tail block: when
 * the stream end is shorter than N the orchestration zero-pads the last
 * block to N (NOT a shrunk block) and the valid outputs are truncated to
 * y[0 .. nx+nh-2].  In-place; buffers are caller-owned; the kernel MUST
 * NOT take ownership.
 *
 * Returns NULL on illegal N (N < 1, non-power-of-two, or N > 32768
 * ceiling), nh <= 0, nh > N, NULL h, emit failure, or OOM; MUST NOT
 * silently fall back to the fp64 forward path, fp64 cyclic convolution,
 * any fp32 path, adapt, or route to FFTW.
 */
fft_plan *fft_plan_create_linear_conv_fp64(int N, const double complex *h,
                                           int nh);

/*
 * Execute an opt-in fp64 linear-convolution plan in-place on one block:
 * io[0..N-1] holds the length-N input block on entry; on return
 * io[0..N-nh] holds the N-nh+1 valid linear-convolution samples and the
 * trailing nh-1 positions are a discard zone (see
 * fft_plan_create_linear_conv_fp64).  Independent execute symbol (C has
 * no overloading).  `io` is caller-owned; the kernel MUST NOT take
 * ownership.  Calling with a NULL plan / NULL io is a no-op; a
 * non-linear-conv-fp64 plan is a no-op (defensive).  MUST NOT be used on
 * an fp32 plan, and no other execute symbol may run this kernel.
 */
void fft_execute_linear_conv_fp64(fft_plan *plan, double complex *io);

/*
 * Execute the planned 1D complex DFT in-place on `io[0 .. N-1]`.
 *
 * Default / gauss plans: forward DFT
 *
 *     X[k] = sum_{n=0}^{N-1} io[n] * exp(-j 2*pi*k*n / N)
 *
 * Opt-in inverse plans: conjugate-trick inverse (see
 * fft_plan_create_inverse).  Same public symbol; the plan carries the
 * wrapper.  MUST NOT invent a second execute symbol.
 *
 * `io` is both input and output (BEH-FFT-002).  The buffer is allocated and
 * freed by the caller; the kernel MUST NOT take ownership (API-FFT-001).
 * Output order follows the upstream FFTW 1D natural order (BEH-FFT-003);
 * inverse output order matches same-N upstream FFTW_BACKWARD.
 *
 * Calling with a NULL plan or NULL io is a no-op.  A plan created by
 * fft_plan_create_fp64 MUST be executed with fft_execute_fp64 (double
 * complex), never this symbol.
 */
void fft_execute(fft_plan *plan, float complex *io);

/*
 * Execute an opt-in fp64 forward plan in-place on `io[0 .. N-1]` (double
 * complex).  Independent execute symbol for the fp64 plan (C has no
 * overloading).  `io` is caller-owned; the kernel MUST NOT take ownership.
 * Forward DFT, natural order out.  Calling with a NULL plan / NULL io is a
 * no-op; a non-fp64 plan is a no-op (defensive).  MUST NOT be used on an
 * fp32 plan, and the fp32 fft_execute MUST NOT run the fp64 kernel.
 */
void fft_execute_fp64(fft_plan *plan, double complex *io);

/*
 * Opt-in batched Gaussian-windowed forward plan (draft {#API-FFT-010}).
 *
 * Plan once for B independent transforms of the same size N.  The window
 * semantics are IDENTICAL to fft_plan_create_gauss_window (API-FFT-003):
 * plan-time G[n] = exp(-(n-N/2)^2 / (2 sigma^2)) with sigma = N/8 and
 * sum(G) = 1; sigma is not an execute argument.  Each block b computes
 *      X_b[k] = sum_{n=0}^{N-1} G[n]*io_base[b*N + n] * exp(-j2pi*k*n/N)
 * independently, in-place, natural order in / natural order out; blocks
 * are independent with NO cross-block overlap-add (that is a later STFT
 * topic, NOT this contract).
 *
 * Family-only: N = 2^n, n <= 15 (N <= 32768); B in [1, 2^20] blocks.
 * Returns NULL on illegal (N, B) (N < 1 / non-power-of-two / N > 32768 /
 * B < 1 / B > 2^20), emit failure, or OOM — fail-fast, no silent
 * fallback to per-block default forward or any adapt path.
 *
 * MUST NOT change fft_plan_create(int N) or any of the six existing
 * opt-in contracts; this is a NEW entry point, not a mode of them.
 */
fft_plan *fft_plan_create_windowed_batch(int N, int B);

/*
 * Execute a batched windowed-forward plan in-place over B contiguous
 * blocks.  Block b lives at io + b*stride (stride counted in float
 * complex ELEMENTS); the canonical contiguous layout is stride = N and
 * the buffer holds at least B*stride elements.  stride >= N is legal
 * (padded/row layouts); stride < N (overlapping blocks) is illegal —
 * the call is a no-op on NULL plan / NULL io / non-batched plan /
 * stride < N (defensive; no UB, no partial writes outside blocks).
 *
 * The buffer is allocated and freed by the caller; the kernel MUST NOT
 * take ownership.  One call processes ALL B blocks (the kernel sweeps
 * the batch in-core); per-block amortized cost is the measured subject.
 * MUST NOT be used on the six legacy contracts' plans (default /
 * gauss / inverse / conv / linear-conv / conv-dispatch) and MUST NOT
 * run the fp64 kernel; those keep their own execute paths unchanged.
 */
void fft_execute_windowed_batch(fft_plan *plan, float complex *io, int stride);

/*
 * Release a plan handle (refcounted).  Safe to pass NULL.
 */
void fft_plan_destroy(fft_plan *plan);

#ifdef __cplusplus
}
#endif

#endif /* MEGAKERNEL_FFT_FFT_H */
