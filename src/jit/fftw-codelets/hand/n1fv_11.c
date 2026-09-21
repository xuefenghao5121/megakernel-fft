/*
 * poc/jit-anyN-fwd-inv/src/jit/fftw-codelets/hand/n1fv_11.c — topic
 * jit-anyN-fwd-inv F2 leaf-pool expansion (DELTA F2, DESIGN §Levers L-R
 * path B: hand-written leaf).  RETIRED from the default build in R1
 * (amend r1 / DELTA F3, L-S): the upstream third_party n1fv_11 SIMD leaf
 * is vendored instead; this hand pair remains linked ONLY as the
 * FFT_ANYN_LEAVES=hand reproduction knob ("fallback mode when upstream
 * has no leaf" evidence, R0 probe records retained).
 *
 * HAND-WRITTEN radix-11 notwiddled forward codelet (n1fv_11), F2 tool
 * adjudication path A (genfft factory) being unavailable: the OCaml
 * toolchain is not on this machine and installing it (sudo + network) is a
 * human decision (ndf/evidence/tool-adjudication-r0.log).  This file is
 * topic-local (MUST NOT be written back into third_party/).
 *
 * Calling convention: locked to the LIVE vendored convention (not memory) —
 * dft/codelet-dft.h + kernel/ifftw.h with the x86-64
 * PRECOMPUTE_ARRAY_INDICES flavor:
 *
 *   void n1fv_11(const R *ri, const R *ii, R *ro, R *io,
 *                stride is, stride os, INT v, INT ivs, INT ovs)
 *
 * stride is an INT* array (is[j] = element offset in R units); interleaved
 * complex (ii = ri + 1, io = ro + 1).  For each instance k in [0, v):
 *
 *   out_j = sum_{j'=0..10} in_j' * e^{-2*pi*i*j*j'/11}
 *
 * with in_j' = (ri[k*ivs + is[j']], ii[...]), out written at
 * ro[k*ovs + os[j]].  Unlike the VL-vectorized vendored leaves this hand
 * leaf processes EXACTLY v instances (no padding lane), so the odd-Bp tail
 * calls (v = 1) are exact without slack steering.  No transcendentals at
 * execute: the 11th roots of unity are compile-time constants.  The leaf
 * is numerically probed (call-in/call-out vs the reference DFT on the
 * production geometry) by weld_codelets_init before any plan may use it.
 */

#include "weld_codelets.h" /* C mode: config shim + ifftw.h + codelet-dft.h */

/* e^{-2*pi*i*m/11} = C11[m] - i*S11[m], m = 0..10 (double, 17 digits). */
static const double hand_c11[11] = {
    1.00000000000000000e+00,  8.41253532831181206e-01,
    4.15415013001886435e-01,  -1.42314838273285005e-01,
    -6.54860733945284990e-01, -9.59492973614497369e-01,
    -9.59492973614497480e-01, -6.54860733945285212e-01,
    -1.42314838273285227e-01, 4.15415013001886047e-01,
    8.41253532831181206e-01,
};
static const double hand_s11[11] = {
    0.00000000000000000e+00,  5.40640817455597555e-01,
    9.09631995354518330e-01,  9.89821441880932795e-01,
    7.55749574354258269e-01,  2.81732556841429671e-01,
    -2.81732556841429393e-01, -7.55749574354258158e-01,
    -9.89821441880932684e-01, -9.09631995354518552e-01,
    -5.40640817455597444e-01,
};

static void hand_n1fv_11(const R *ri, const R *ii, R *ro, R *io,
                         stride is, stride os, INT v, INT ivs, INT ovs) {
    double yr[11], yi[11];
    for (INT k = 0; k < v; ++k) {
        const R *xri = ri + k * ivs;
        const R *xii = ii + k * ivs;
        R *ori = ro + k * ovs;
        R *oii = io + k * ovs;
        for (int j = 0; j < 11; ++j) {
            double sr = 0.0, si = 0.0;
            for (int jp = 0; jp < 11; ++jp) {
                const double xr = xri[is[jp]];
                const double xi = xii[is[jp]];
                const int m = (j * jp) % 11;
                const double wr = hand_c11[m];
                const double wi = -hand_s11[m]; /* e^{-2*pi*i*m/11} */
                sr += xr * wr - xi * wi;
                si += xr * wi + xi * wr;
            }
            yr[j] = sr;
            yi[j] = si;
        }
        for (int j = 0; j < 11; ++j) {
            ori[os[j]] = yr[j];
            oii[os[j]] = yi[j];
        }
    }
}

/* R1 (DELTA F3, L-S): registrar renamed from fftw_codelet_n1fv_11_avx2
 * so the retired hand pair can stay linked for the FFT_ANYN_LEAVES=hand
 * reproduction knob without clashing with the vendored upstream n1fv_11
 * registrar of that name.  Leaf body untouched (R0 probe-locked). */
void fftw_hand_codelet_n1fv_11_reg(planner *p) {
    fftw_kdft_register(p, hand_n1fv_11, 0);
}
