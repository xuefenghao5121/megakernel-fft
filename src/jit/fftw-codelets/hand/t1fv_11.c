/*
 * poc/jit-anyN-fwd-inv/src/jit/fftw-codelets/hand/t1fv_11.c — topic
 * jit-anyN-fwd-inv F2 leaf-pool expansion (DELTA F2, DESIGN §Levers L-R
 * path B: hand-written leaf).  RETIRED from the default build in R1
 * (amend r1 / DELTA F3, L-S): the upstream n1fv_11 is vendored and the
 * 11-factor stages walk the WSK_N1 n1fv+TW-sweep mode (the radix-13
 * incumbent pattern — upstream has no t1fv_11); this hand pair remains
 * linked ONLY as the FFT_ANYN_LEAVES=hand reproduction knob (R0 probe
 * records retained).
 *
 * HAND-WRITTEN radix-11 twiddled forward codelet (t1fv_11), the twiddle
 * twin of hand/n1fv_11.c (F2 tool adjudication: genfft factory
 * unavailable — OCaml not on machine; see ndf/evidence/
 * tool-adjudication-r0.log).  Topic-local, never written back into
 * third_party/.
 *
 * Calling convention locked to the LIVE vendored convention
 * (dft/codelet-dft.h kdftw + kernel/ifftw.h PRECOMPUTE_ARRAY_INDICES):
 *
 *   void t1fv_11(R *ri, R *ii, const R *W, stride rs, INT mb, INT me,
 *                INT ms)
 *
 * In-place (per FFTW t1fv shape, ri/ii are both input and output); for
 * each instance m in [mb, me), reading in_j' = (ri[m*ms + rs[j']],
 * ii[m*ms + rs[j']]):
 *
 *   out_j = sum_{j'} in_j' * e^{-2*pi*i*m*j'/Bt} * e^{-2*pi*i*j*j'/11}
 *
 * (Bt = Bp * 11, the stage's mixed-radix block size — exactly the
 * reference in weld_glue.c probe_t1 / ref_dft_r).  The stage twiddles
 * e^{-2*pi*i*m*j'/Bt} are read from the PLAN-TIME W table built by
 * weld_t1_w_fill (slot = e^{+2*pi*i*j*k/Bt}; the codelet consumes the
 * conjugate, the BYTWJ/VZMULJ idiom): row p = (m >> 1), lane = m & 1,
 * slot(j') at W + p*4*(r-1) + 4*(j'-1) + 2*lane.  No transcendentals at
 * execute (butterfly roots are compile-time constants).  Exact instance
 * loop (no VL padding lane) — the weld only calls t1 with me-mb = Bp even.
 */

#include "weld_codelets.h" /* C mode: config shim + ifftw.h + codelet-dft.h */

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

static void hand_t1fv_11(R *ri, R *ii, const R *W, stride rs, INT mb,
                         INT me, INT ms) {
    enum { R11 = 11 };
    double yr[R11], yi[R11];
    const INT rowDoubles = 4 * (R11 - 1);
    for (INT m = mb; m < me; ++m) {
        const INT p = m >> 1;
        const int lane = (int)(m & 1);
        const double *row = W + (size_t)p * (size_t)rowDoubles;
        R *xri = ri + m * ms;
        R *xii = ii + m * ms;
        for (int j = 0; j < R11; ++j) {
            double sr = 0.0, si = 0.0;
            for (int jp = 0; jp < R11; ++jp) {
                const double xr = xri[rs[jp]];
                const double xi = xii[rs[jp]];
                /* stage twiddle: conj(W slot) = e^{-2*pi*i*m*jp/Bt} */
                double tr = 1.0, ti = 0.0;
                if (jp != 0) {
                    const double *slot = row + 4 * (jp - 1) + 2 * lane;
                    tr = slot[0];
                    ti = -slot[1];
                }
                /* butterfly root e^{-2*pi*i*j*jp/11} */
                const int mm = (j * jp) % R11;
                const double br = hand_c11[mm];
                const double bi = -hand_s11[mm];
                /* (tr + i*ti)(br + i*bi) */
                const double gr = tr * br - ti * bi;
                const double gi = tr * bi + ti * br;
                sr += xr * gr - xi * gi;
                si += xr * gi + xi * gr;
            }
            yr[j] = sr;
            yi[j] = si;
        }
        for (int j = 0; j < R11; ++j) {
            xri[rs[j]] = yr[j];
            xii[rs[j]] = yi[j];
        }
    }
}

/* R1 (DELTA F3, L-S): registrar renamed from fftw_codelet_t1fv_11_avx2
 * (no upstream counterpart symbol — the vendor t1fv set has no radix 11 —
 * but renamed in lockstep with n1fv_11 for a uniform hand-* namespace).
 * Leaf body untouched (R0 probe-locked). */
void fftw_hand_codelet_t1fv_11_reg(planner *p) {
    fftw_kdft_dit_register(p, hand_t1fv_11, 0);
}
