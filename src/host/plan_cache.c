/*
 * poc/jit-fp64-forward/src/host/plan_cache.c
 *
 * Copy-then-edit of Trunk src/host/plan_cache.c (which itself carries the
 * promoted opt-in cyclic-convolution / linear-conv / conv-dispatch / inverse
 * / gauss opt-in contracts).  This topic adds ONLY the opt-in fp64 forward
 * plan (draft {#API-FFT-008} / {#BEH-FFT-009}) on top, in a SEPARATE fp64
 * plan table so it can never collide with, or regress, the fp32 default /
 * gauss / inverse / conv / linear-conv / conv-dispatch semantics.
 *
 * Default fft_plan_create(N) stays the unwindowed fp32 family JIT (cache
 * key = N, windowed=0, inverse=0, conv=0, linear=0, fp64=0).  Gauss
 * opt-in stays windowed=1.  Opt-in inverse stays inverse=1.  Opt-in
 * fft_plan_create_conv(N, h) stays conv=1.  Opt-in
 * fft_plan_create_linear_conv(N, h, nh) uses linear=1 + nh.  Opt-in
 * fft_plan_create_fp64(N) uses fp64=1 and lives in its own table.
 *
 * fp64 HOW = precision-parameterized family forward kernel
 * (fft_jit_kernel_create_fp64): double complex buffer, independent execute
 * symbol fft_execute_fp64 (C has no overloading; MUST NOT shove a double
 * buffer through the fp32 float complex * fft_execute).  Family-only,
 * fail-fast: illegal N / emit / OOM → NULL, MUST NOT silently fall back to
 * fp32 default forward, adapt, or a gauss/inverse/conv plan.
 *
 * Topic jit-fp64-conv-fuse adds ONLY the opt-in fp64 cyclic convolution
 * (draft {#API-FFT-011}): fft_plan_create_conv_fp64(N, h) +
 * fft_execute_conv_fp64 in its own plan table (g_cache_head_conv64).  HOW =
 * fused boundary-sunk conv weld (fft_jit_kernel_create_conv_fused_fp64:
 * forward final store multiplies X*H and conjugates, inverse stage-0
 * gathers conj(X*H), inverse final stores conj*invN — the intermediate
 * spectrum transits the working buffer exactly once) plus the arbitrary-N
 * semantic-preservation layer (DESIGN §L-J, adjudicated zero-pad +
 * wraparound correction; see fft_plan_create_conv_fp64 for the route
 * table).  All fp32 contracts and the fp64 forward stay unchanged.
 *
 * Topic jit-fp64-inverse (amend resume 2026-09-17) adds ONLY the opt-in
 * fp64 INVERSE (draft {#API-FFT-009} / {#BEH-FFT-010}):
 * fft_plan_create_fp64_inverse(N) + fft_execute_fp64_inverse in its own
 * plan table (g_cache_head_inv64).  HOW = the boundary-sunk conjugate
 * trick fused inverse kernel (fft_jit_kernel_create_inverse_fused_fp64:
 * entry conj sunk into the stage-0 gather load boundary, exit conj*(1/N)
 * sunk into the final store boundary — no wrapper sweeps) for N >= 32;
 * the N < 32 family floor keeps the retired three-segment wrapper around
 * the promoted fp64 forward kernel (the Trunk small-N floor).  All fp32
 * contracts, the fp64 forward, and the fp64 conv stay unchanged
 * (regression-only).
 *
 * Topic jit-fp64-linear-conv (R0 2026-09-17) adds ONLY the opt-in fp64
 * block-pinned OLS LINEAR convolution (draft {#API-FFT-012} /
 * {#BEH-FFT-017}): fft_plan_create_linear_conv_fp64(N, h, nh) +
 * fft_execute_linear_conv_fp64 in its own plan table
 * (g_cache_head_lin64, key = (N, nh)).  HOW = the fused OLS weld
 * (fft_jit_kernel_create_ols_fused_fp64 — the conv weld body with the
 * OLS boundary edits): the window shift sinks into the half-A stage-0
 * gather as a plan-time ROTATED table (zero runtime cost — the
 * wrap-around-polluted head lands at the END of the cyclic result, the
 * N-nh+1 valid samples become the natural PREFIX), and the half-B final
 * stores write ONLY the valid prefix (the trailing nh-1 discard zone is
 * not written; FFT_OLS_DISCARD=write is the topic-local
 * written-then-abandoned adjudication knob).  Overlap state (retained
 * nh-1 input tail + stream position counter) lives in the plan,
 * zero-initialized head prefix — callers MUST NOT add/drop samples
 * themselves (the OLS bookkeeping is INSIDE the execute at the store
 * boundary).  The plan also carries the topic-internal stream face
 * (fft_ols_stream_fp64 — NOT a public symbol; the emitted block loop
 * sweeps the whole padded stream [0^{nh-1} | x | 0-tail] in one call:
 * the O-ii form (a) orchestration-sunk subject; form (b) = wrapper loop
 * of single blocks).  N = 1 / 2 keep a tiny direct wrapper (no transform
 * is meaningful below the weld floor N >= 4).  All fp32 contracts, the
 * fp64 forward / inverse / conv stay unchanged (regression-only).
 */

#include "fft.h"
#include "fft_adapt.h"
#if !defined(FFT_JIT_DISABLE)
#include "fft_jit.h"
#endif

/* topic jit-ps-sched-fuse: conv / linear-conv execute moves to the FUSED
 * kernel (fft_jit_kernel_create_conv_fused — forward final stage
 * multiplies X[k]*H[k] + conjugates at the STORE boundary, inverse
 * half's stage-0 gathers the natural conj(X*H) directly, inverse final
 * stores conj(.)/N).  The three wrapper sweeps (pointwise / conj /
 * conj*invN) disappear; the public symbols and semantics are unchanged.
 * FFT_FUSED_CONV=off (topic-local debug knob) restores the promoted
 * three-segment path on the same kernels for A/B comparison. */

#include <complex.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(FFT_JIT_DISABLE)
static int fused_conv_enabled(void) {
    const char *v = getenv("FFT_FUSED_CONV");
    return !(v != NULL && (strcmp(v, "0") == 0 || strcmp(v, "off") == 0));
}
#endif

#if !defined(FFT_JIT_DISABLE)
static int fft_jit_family_member(int N) {
    if (N < 1 || N > 32768) {
        return 0;
    }
    return (N & (N - 1)) == 0;
}

/* topic jit-anyN-fwd-inv (draft {#API-FFT-013}): the FACTOR COVERAGE SET.
 * R1 (DELTA F3, L-S): S = weld_factors[] prime closure = {2,3,5,7,13}
 * vendored base + {11} now from the VENDORED upstream n1fv_11 (the R0
 * hand-written pair retired to the FFT_ANYN_LEAVES=hand reproduction
 * knob); 14 = vendored upstream n1fv_14 joins weld_factors as a
 * composite (2x7 — no new prime, closure unchanged).  Accept = every
 * prime factor of N in the closure (S-smooth) and 1 <= N <= 32768.
 * The FFT_ANYN_LEAVES knob now selects the leaf SOURCE only (vendor11
 * default / hand = R0 reproduction / stock = vendored-only == default,
 * since the vendor 11/14 leaves ARE the vendored pool) — the accepted N
 * domain is knob-independent.  Out-of-set N -> NULL fail-closed: MUST
 * NOT silently walk adapt / zero-pad / FFTW / any other plan (unlike the
 * conv entry's auto semantic-preservation route, a plain transform has
 * no zero-pad semantic correction available). */
static int anyn_prime_covered(int N) {
    static const int primes[] = {2, 3, 5, 7, 11, 13};
    const int npr = (int)(sizeof(primes) / sizeof(primes[0]));
    for (int i = 0; i < npr; ++i) {
        while (N % primes[i] == 0) {
            N /= primes[i];
        }
    }
    return N == 1;
}

static int anyn_covered(int N) {
    if (N < 1 || N > 32768 || ((N & (N - 1)) == 0)) {
        return 0; /* bounds / pow2 = the family path, never anyN */
    }
    return anyn_prime_covered(N);
}

/* bare covered prime or tiny N (cannot form a >= 2-stage weld): the
 * wrapper floor set (DESIGN boundary-degeneracy rule). */
static int anyn_is_bare_prime_or_tiny(int N) {
    if (N <= 3) {
        return 1;
    }
    if (N % 2 == 0) {
        return 0; /* even composite or 2-covered -> factorable shape */
    }
    /* bare prime <=> division by every smaller covered prime leaves N */
    static const int primes[] = {3, 5, 7, 11, 13};
    const int npr = (int)(sizeof(primes) / sizeof(primes[0]));
    for (int i = 0; i < npr; ++i) {
        if (primes[i] < N && N % primes[i] == 0) {
            return 0;
        }
    }
    return 1; /* 5,7,11,13 themselves (and out-of-set primes never reach
               * here — anyn_covered already rejected them) */
}

/* wrapper floor: in-place natural-order DFT / IDFT, double precision,
 * O(N^2) — only ever used at N <= 13 (bare primes + tiny).  Oracle
 * tolerance holds (few-term double sums). */
#include <math.h>
#define ANYN_PI 3.14159265358979323846
static void anyn_floor_dft(double complex *io, int N) {
    double complex *tmp =
        (double complex *)malloc((size_t)N * sizeof(double complex));
    if (tmp == NULL) {
        return; /* OOM: no output mutation (fail-visible in verify) */
    }
    for (int j = 0; j < N; ++j) {
        double complex s = 0.0 + 0.0 * I;
        for (int t = 0; t < N; ++t) {
            const double a = -2.0 * ANYN_PI * (double)(j * t % N) / (double)N;
            s += io[t] * (cos(a) + sin(a) * I);
        }
        tmp[j] = s;
    }
    for (int j = 0; j < N; ++j) {
        io[j] = tmp[j];
    }
    free(tmp);
}

static void anyn_floor_idft(double complex *io, int N) {
    const double invN = 1.0 / (double)N;
    double complex *tmp =
        (double complex *)malloc((size_t)N * sizeof(double complex));
    if (tmp == NULL) {
        return;
    }
    for (int t = 0; t < N; ++t) {
        double complex s = 0.0 + 0.0 * I;
        for (int j = 0; j < N; ++j) {
            const double a = 2.0 * ANYN_PI * (double)(j * t % N) / (double)N;
            s += io[j] * (cos(a) + sin(a) * I);
        }
        tmp[t] = s * invN;
    }
    for (int t = 0; t < N; ++t) {
        io[t] = tmp[t];
    }
    free(tmp);
}
#endif

struct fft_plan {
    int N;
    int windowed; /* 0 = default unwindowed; 1 = gauss opt-in */
    int inverse;  /* 0 = forward; 1 = opt-in inverse (conjugate trick) */
    int conv;     /* 0 = not conv; 1 = opt-in cyclic convolution */
    int linear;   /* 0 = not linear; 1 = opt-in linear conv (OLS) */
    int nh;       /* linear: kernel length (0 < nh <= N) */
    int fp64;     /* 0 = fp32; 1 = opt-in fp64 forward (separate table) */
    int fused_conv; /* topic jit-ps-sched-fuse: 1 = plan->jit_fn is the
                       fused single-kernel conv execute */
    float complex *H; /* conv/linear: plan-time H[k] = FFT(h), natural order */
    /* dispatch (whole-sequence linear conv, draft API-FFT-007) */
    int dispatch;         /* 1 = dispatch plan */
    int dispatch_direct;  /* 1 = direct scalar leg; 0 = OLS block leg */
    int nx;               /* full input sequence length (dispatch only) */
    fft_plan *sub;        /* OLS leg: refcounted linear_conv sub-plan (N_block, nh) */
    float complex *h_copy;/* direct leg: time-domain kernel copy (len nh) */
    float complex *work;  /* OLS leg: padded-input scratch (len nx+nh-1) */
    float complex *blk;   /* OLS leg: block scratch (len sub->N) */
    /* topic jit-fp64-conv-fuse (draft {#API-FFT-011}): fp64 cyclic conv */
    int conv64;            /* 1 = fp64 cyclic-conv plan (own table) */
    int c64_mode;          /* 0 tiny-direct; 1 weld-direct (pow2);
                              2 pad-single; 3 pad-corr; 4 OLA multi-block */
    int Np;                /* main weld transform size (modes 2..4) */
    double complex *buf64; /* main padded work buffer (len Np) */
    int Np2;               /* correction weld size (mode 3; 0 otherwise) */
    int corrL;             /* correction overlap L = 2N-2-Np (mode 3) */
    double complex *buf2_64; /* mode 3: correction work (len >= Np2);
                                mode 4: x copy (len N); tiny N: unused */
    double complex *h64;   /* modes 0 / tiny-corr: time-domain copies
                              (len N for mode 0, len L+1 for tiny corr) */
#if !defined(FFT_JIT_DISABLE)
    fft_jit_kernel *jit2;    /* mode 3: correction weld kernel */
    fft_jit_fn_t jit2_fn;
#endif
    int olaPieces;         /* mode 4: number of OLA pieces */
    int olaP;              /* mode 4: piece length P = Np-N+1 */
    fft_adapt_plan *adapt;
    /* topic jit-fp64-inverse (draft {#API-FFT-009}): fp64 inverse plans
     * live in their OWN table (key = N) and MUST NOT collide with the
     * fp32 inverse, the fp64 forward, or any other plan. */
    int inv64;      /* 1 = fp64 inverse plan (own table) */
    /* topic jit-fp64-linear-conv (draft {#API-FFT-012}): fp64 OLS
     * linear-conv plans live in their OWN table (key = (N, nh)) and
     * MUST NOT collide with the fp32 linear-conv, the fp64 forward /
     * inverse / conv, or any other plan. */
    int lin64;      /* 1 = fp64 OLS linear-conv plan (own table) */
    int ols_tiny;   /* 1 = N < 4 tiny direct wrapper (no weld); the
                      time-domain padded h copy rides h64 */
    double complex *ols_tail; /* retained nh-1 input tail (zero-init
                                 head prefix; updated per execute) */
    long ols_pos;   /* stream bookkeeping: executes so far */
#if !defined(FFT_JIT_DISABLE)
    fft_jit_ols_fn_t jit_ols; /* stream entry (P, y, nblocks) — the
                                 O-ii form (a) emitted block loop */
#endif
    int fused_inv;  /* 1 = plan->jit_fn is the FUSED single-kernel inverse
                       execute (N >= 32); 0 = the small-N wrapper floor
                       (forward kernel + independent conj / conj*invN
                       sweeps) */
    /* topic jit-anyN-fwd-inv (draft {#API-FFT-013}): 1 = coverage-set N
     * below the weld floor (bare prime / tiny N = 1..3) — the host
     * wrapper floor computes the O(N^2) DFT / IDFT directly at execute
     * (oracle-tolerance exact; conv tiny precedent).  jit / jit_fn are
     * NULL on these plans. */
    int anyn_floor;
    /* topic jit-joint-search-batched (draft {#API-FFT-010}): batched
     * windowed-forward plans live in their OWN table (key = (N, B)) and
     * MUST NOT collide with any of the six legacy contracts or fp64. */
    int batch;       /* 1 = windowed-batch plan (separate table) */
    int bcount;      /* batch: B blocks */
    int batch_place; /* batch: FFT_PLACE_LOAD / FFT_PLACE_STANDALONE /
                      * FFT_PLACE_PRESWEEP (topic R2) */
    int batch_form;  /* batch: FFT_BATCH_FORM_SINGLE / LOOP / SLIM
                      * (topic R2) */
    float *G2;       /* batch standalone/presweep: duplicated window rows */
#if !defined(FFT_JIT_DISABLE)
    fft_jit_kernel *jit;
    fft_jit_fn_t jit_fn;
    fft_jit_fn2_t jit_fn2; /* batch single-core entry (io, stride) */
    fft_jit_fn1c_t jit_fn2c; /* topic R2: slim single-core contiguous
                              * stride==N entry */
#endif
    int refcount;
    struct fft_plan *next;
};

static pthread_mutex_t g_cache_lock = PTHREAD_MUTEX_INITIALIZER;
static struct fft_plan *g_cache_head = NULL;
static struct fft_plan *g_cache_head_fp64 = NULL;
static struct fft_plan *g_cache_head_batch = NULL;

static fft_plan *cache_lookup_batch_locked(int N, int B) {
    for (struct fft_plan *p = g_cache_head_batch; p != NULL; p = p->next) {
        if (p->N == N && p->bcount == B) {
            p->refcount++;
            return p;
        }
    }
    return NULL;
}

static struct fft_plan *g_cache_head_conv64 = NULL;

static fft_plan *cache_lookup_conv64_locked(int N) {
    for (struct fft_plan *p = g_cache_head_conv64; p != NULL; p = p->next) {
        if (p->N == N) {
            p->refcount++;
            return p;
        }
    }
    return NULL;
}

/* topic jit-fp64-inverse (draft {#API-FFT-009}): fp64 inverse table. */
static struct fft_plan *g_cache_head_inv64 = NULL;

static fft_plan *cache_lookup_inv64_locked(int N) {
    for (struct fft_plan *p = g_cache_head_inv64; p != NULL; p = p->next) {
        if (p->N == N) {
            p->refcount++;
            return p;
        }
    }
    return NULL;
}

/* topic jit-fp64-linear-conv (draft {#API-FFT-012}): fp64 OLS
 * linear-conv table (key = (N, nh)). */
static struct fft_plan *g_cache_head_lin64 = NULL;

static fft_plan *cache_lookup_lin64_locked(int N, int nh) {
    for (struct fft_plan *p = g_cache_head_lin64; p != NULL; p = p->next) {
        if (p->N == N && p->nh == nh) {
            p->refcount++;
            return p;
        }
    }
    return NULL;
}

/* smallest power of two >= v (v >= 1) */
static int conv64_next_pow2(long v) {
    long p = 1;
    while (p < v) {
        p <<= 1;
    }
    return (int)p;
}

static fft_plan *cache_lookup_locked(int N, int windowed, int inverse,
                                     int conv, int linear, int nh) {
    for (struct fft_plan *p = g_cache_head; p != NULL; p = p->next) {
        if (p->N == N && p->windowed == windowed && p->inverse == inverse &&
            p->conv == conv && p->linear == linear && p->nh == nh) {
            p->refcount++;
            return p;
        }
    }
    return NULL;
}

static fft_plan *cache_lookup_fp64_locked(int N) {
    for (struct fft_plan *p = g_cache_head_fp64; p != NULL; p = p->next) {
        if (p->N == N) {
            p->refcount++;
            return p;
        }
    }
    return NULL;
}

static fft_plan *fft_plan_create_ex(int N, int windowed, int inverse) {
    if (N < 1) {
        return NULL;
    }
    if (inverse && windowed) {
        return NULL; /* no fused inverse-window contract in this topic */
    }

    pthread_mutex_lock(&g_cache_lock);
    fft_plan *hit = cache_lookup_locked(N, windowed, inverse, 0, 0, 0);
    pthread_mutex_unlock(&g_cache_lock);
    if (hit != NULL) {
        return hit;
    }

#if !defined(FFT_JIT_DISABLE)
    fft_jit_kernel *jit = NULL;
    fft_jit_fn_t jit_fn = NULL;
    if (inverse) {
        /* Inverse is family-only.  Non-family / emit / OOM → NULL.
         * MUST NOT fall back to adapt or a silent forward plan. */
        if (!fft_jit_family_member(N)) {
            return NULL;
        }
        jit = fft_jit_kernel_create(N);
        if (jit == NULL || fft_jit_kernel_fn(jit) == NULL) {
            fft_jit_kernel_destroy(jit);
            return NULL;
        }
        jit_fn = fft_jit_kernel_fn(jit);
    } else if (windowed) {
        /* Fusion is family-only.  Non-family / emit / missing G → NULL.
         * MUST NOT fall back to a silently-windowed adapt path. */
        if (!fft_jit_family_member(N)) {
            return NULL;
        }
        jit = fft_jit_kernel_create_gauss_window(N);
        if (jit == NULL || fft_jit_kernel_fn(jit) == NULL ||
            fft_jit_kernel_gauss_lut(jit) == NULL) {
            fft_jit_kernel_destroy(jit);
            return NULL;
        }
        jit_fn = fft_jit_kernel_fn(jit);
    } else if (fft_jit_family_member(N)) {
        jit = fft_jit_kernel_create(N);
        if (jit == NULL) {
            return NULL;
        }
        jit_fn = fft_jit_kernel_fn(jit);
    }
#else
    if (inverse || windowed) {
        return NULL; /* no silent inverse / window without the kernel */
    }
#endif

    fft_adapt_plan *adapt = NULL;
#if !defined(FFT_JIT_DISABLE)
    if (jit == NULL)
#endif
    {
        /* Inverse never reaches here: it returned NULL without a kernel. */
        adapt = fft_adapt_plan_create(N);
        if (adapt == NULL) {
#if !defined(FFT_JIT_DISABLE)
            fft_jit_kernel_destroy(jit);
#endif
            return NULL;
        }
    }

    struct fft_plan *plan = (struct fft_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
#if !defined(FFT_JIT_DISABLE)
        fft_jit_kernel_destroy(jit);
#endif
        fft_adapt_plan_destroy(adapt);
        return NULL;
    }
    plan->N = N;
    plan->windowed = windowed ? 1 : 0;
    plan->inverse = inverse ? 1 : 0;
    plan->conv = 0;
    plan->linear = 0;
    plan->nh = 0;
    plan->fp64 = 0;
    plan->H = NULL;
    plan->adapt = adapt;
#if !defined(FFT_JIT_DISABLE)
    plan->jit = jit;
    plan->jit_fn = jit_fn;
#endif
    plan->refcount = 1;

    pthread_mutex_lock(&g_cache_lock);
    hit = cache_lookup_locked(N, windowed, inverse, 0, 0, 0);
    if (hit != NULL) {
        pthread_mutex_unlock(&g_cache_lock);
#if !defined(FFT_JIT_DISABLE)
        fft_jit_kernel_destroy(jit);
#endif
        fft_adapt_plan_destroy(adapt);
        free(plan);
        return hit;
    }
    plan->next = g_cache_head;
    g_cache_head = plan;
    pthread_mutex_unlock(&g_cache_lock);
    return plan;
}

fft_plan *fft_plan_create(int N) {
    return fft_plan_create_ex(N, 0, 0);
}

fft_plan *fft_plan_create_gauss_window(int N) {
    return fft_plan_create_ex(N, 1, 0);
}

fft_plan *fft_plan_create_inverse(int N) {
    return fft_plan_create_ex(N, 0, 1);
}

fft_plan *fft_plan_create_fp64(int N) {
    if (N < 1 || N > 32768) {
        return NULL;
    }
#if !defined(FFT_JIT_DISABLE)
    /* topic jit-anyN-fwd-inv (draft {#API-FFT-013} N-domain revision of
     * {#API-FFT-008}): pow2 N = the ZERO-CHANGE family guard — the exact
     * promoted path (family check + family kernel), byte-for-byte
     * semantics.  Non-pow2 N: factor-coverage-set acceptance (fail-closed
     * NULL out of set — MUST NOT silently fall back to fp32 default
     * forward, adapt, pad, FFTW, or any other opt-in plan). */
    if (fft_jit_family_member(N)) {
        pthread_mutex_lock(&g_cache_lock);
        fft_plan *hit = cache_lookup_fp64_locked(N);
        pthread_mutex_unlock(&g_cache_lock);
        if (hit != NULL) {
            return hit;
        }

        fft_jit_kernel *jit = fft_jit_kernel_create_fp64(N);
        if (jit == NULL || fft_jit_kernel_fn(jit) == NULL) {
            fft_jit_kernel_destroy(jit);
            return NULL;
        }
        fft_jit_fn_t jit_fn = fft_jit_kernel_fn(jit);

        struct fft_plan *plan = (struct fft_plan *)calloc(1, sizeof(*plan));
        if (plan == NULL) {
            fft_jit_kernel_destroy(jit);
            return NULL;
        }
        plan->N = N;
        plan->fp64 = 1;
        plan->jit = jit;
        plan->jit_fn = jit_fn;
        plan->refcount = 1;

        pthread_mutex_lock(&g_cache_lock);
        hit = cache_lookup_fp64_locked(N);
        if (hit != NULL) {
            pthread_mutex_unlock(&g_cache_lock);
            fft_jit_kernel_destroy(jit);
            free(plan);
            return hit;
        }
        plan->next = g_cache_head_fp64;
        g_cache_head_fp64 = plan;
        pthread_mutex_unlock(&g_cache_lock);
        return plan;
    }

    /* ---- non-pow2: the coverage-set档 (draft {#API-FFT-013}) ---- */
    if (!anyn_covered(N)) {
        return NULL; /* fail-closed; no adapt / pad / FFTW route */
    }
    pthread_mutex_lock(&g_cache_lock);
    fft_plan *hit = cache_lookup_fp64_locked(N);
    pthread_mutex_unlock(&g_cache_lock);
    if (hit != NULL) {
        return hit; /* R0 cache decision: non-pow2 plans share the fp64
                     * forward table (key = N; the pow2 / non-pow2 key
                     * spaces are disjoint — recorded in evidence) */
    }

    fft_jit_kernel *jit = NULL;
    fft_jit_fn_t jit_fn = NULL;
    int floor = 0;
    if (anyn_is_bare_prime_or_tiny(N)) {
        floor = 1; /* wrapper floor (bare prime / tiny N <= 3) */
    } else {
        if (!fft_jit_has_avx2_fma()) {
            return NULL; /* CON-FFT-002: fail-closed without AVX2+FMA3 */
        }
        jit = fft_jit_kernel_create_anyN_forward_fp64(N);
        if (jit == NULL || fft_jit_kernel_fn(jit) == NULL) {
            fft_jit_kernel_destroy(jit);
            return NULL; /* emit failure / unviable shape: fail-closed */
        }
        jit_fn = fft_jit_kernel_fn(jit);
    }

    struct fft_plan *plan = (struct fft_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        fft_jit_kernel_destroy(jit);
        return NULL;
    }
    plan->N = N;
    plan->fp64 = 1;
    plan->anyn_floor = floor;
    plan->jit = jit;
    plan->jit_fn = jit_fn;
    plan->refcount = 1;

    pthread_mutex_lock(&g_cache_lock);
    hit = cache_lookup_fp64_locked(N);
    if (hit != NULL) {
        pthread_mutex_unlock(&g_cache_lock);
        fft_jit_kernel_destroy(jit);
        free(plan);
        return hit;
    }
    plan->next = g_cache_head_fp64;
    g_cache_head_fp64 = plan;
    pthread_mutex_unlock(&g_cache_lock);
    return plan;
#else
    /* No silent fp64 forward without the JIT kernel. */
    return NULL;
#endif
}

/*
 * topic jit-fp64-conv-fuse (draft {#API-FFT-011}): opt-in fp64 cyclic
 * convolution, arbitrary N within the family ceiling (1 <= N <= 32768).
 * Semantic-preservation route table (zero-pad + wraparound correction —
 * the candidate adjudicated against Bluestein in this topic; see
 * DESIGN §L-J and ndf/evidence):
 *
 *   N = 2^n, N >= 4 : weld DIRECTLY on io (Np = N, no copies).
 *   N = 1 / 2       : degenerate direct wrapper (h64 copy, no transform).
 *   2N-1 <= 32768   : one weld at Np = next_pow2(2N-1) on the padded plan
 *                     buffer; exact (linear support fits, no aliasing);
 *                     wrap-by-N add back out.
 *   N >= 16385      : main weld @32768 is aliased (2N-1 > 32768); the
 *                     wrapped tail y_lin[m+Np] is reconstructed EXACTLY by
 *                     a tail-correlation weld at Np2 = next_pow2(2L+1),
 *                     L = 2N-2-Np (feasible for N <= 24576); beyond that
 *                     (N <= 32768) a multi-block OLA on the same weld
 *                     (pieces of P = 32769-N) keeps the semantics exact.
 *
 * FFT_CONV_ROUTE=corr is a topic-local debug/adjudication knob forcing the
 * small-main + correction route at the smaller Np = next_pow2(N) (used by
 * the R0 route adjudication measurements; default unset = auto).
 * FFT_CONV_LOG=1 prints the chosen route to stderr.  Both are topic-local
 * experiment knobs, NOT public API.  h is NOT clobbered (H is materialized
 * on a padded scratch copy inside the JIT conv create).  Returns NULL on
 * illegal N / NULL h / emit / OOM — never a silent fallback.
 */
fft_plan *fft_plan_create_conv_fp64(int N, const double complex *h) {
    if (N < 1 || N > 32768 || h == NULL) {
        return NULL;
    }
#if !defined(FFT_JIT_DISABLE)
    pthread_mutex_lock(&g_cache_lock);
    fft_plan *hit = cache_lookup_conv64_locked(N);
    pthread_mutex_unlock(&g_cache_lock);
    if (hit != NULL) {
        return hit;
    }

    const int pow2 = ((N & (N - 1)) == 0);
    const char *routeEnv = getenv("FFT_CONV_ROUTE");
    const int forceCorr =
        (routeEnv != NULL && strcmp(routeEnv, "corr") == 0);

    /* topic jit-codelet-weld: ORIGINAL-SIZE weld first (codelet leaves x
     * JIT weld, draft {#BEH-FFT-016}); FFT_WELD=off pins the promoted
     * pad route (adjudication / regression leg).  The weld create
     * validates factorability + features internally and returns NULL
     * otherwise — the route table below then applies unchanged. */
    fft_jit_kernel *weldJit = NULL;
    if (!pow2) {
        const char *weldEnv = getenv("FFT_WELD");
        const int weldOff =
            (weldEnv != NULL && strcmp(weldEnv, "off") == 0);
        if (!weldOff) {
            weldJit = fft_jit_kernel_create_conv_weld_fp64(
                N, (const double *)h);
        }
    }

    int mode = 0; /* 0 tiny, 1 direct, 2 pad-single, 3 pad-corr, 4 OLA,
                     5 weld-original (topic jit-codelet-weld) */
    int Np = N, Np2 = 0, corrL = 0, olaPieces = 0, olaP = 0;
    if (weldJit != NULL) {
        mode = 5; /* exact N-point cyclic conv directly on io */
    } else if (pow2) {
        mode = (N >= 4) ? 1 : 0;
    } else {
        const int npLin = conv64_next_pow2(2L * N - 1); /* >= 2N-1 */
        if (npLin <= 32768 && !forceCorr) {
            mode = 2;
            Np = npLin;
        } else {
            if (forceCorr && npLin <= 32768) {
                Np = conv64_next_pow2((long)N); /* small main + corr */
            } else {
                Np = 32768; /* npLin > ceiling: biggest family weld */
            }
            corrL = 2 * N - 2 - Np;
            Np2 = conv64_next_pow2(2L * corrL + 1);
            if (corrL + 1 <= 8 || Np2 <= 32768) {
                mode = 3;
            } else {
                /* tail-correlation itself exceeds the ceiling: exact
                 * multi-block OLA on the same main weld */
                mode = 4;
                Np = 32768;
                olaP = Np - N + 1;
                olaPieces = (N + olaP - 1) / olaP;
                Np2 = 0;
                corrL = 0;
            }
        }
    }

    /* plan-time H materialization happens inside the JIT conv create
     * (promoted forward kernel on a padded scratch copy of h; h is NOT
     * clobbered; H tables separate from the twiddle LUT). */
    double complex *hPad = NULL;
    fft_jit_kernel *jit = NULL;
    fft_jit_fn_t jit_fn = NULL;
    if (mode == 5) {
        jit = weldJit;
        weldJit = NULL;
        jit_fn = fft_jit_kernel_fn(jit);
        if (jit_fn == NULL) {
            fft_jit_kernel_destroy(jit);
            return NULL;
        }
    }
    if (mode >= 1 && mode != 5) {
        hPad = (double complex *)malloc((size_t)Np * sizeof(*hPad));
        if (hPad == NULL) {
            return NULL;
        }
        for (int i = 0; i < Np; i++) {
            hPad[i] = (i < N) ? h[i] : (0.0 + 0.0 * I);
        }
        jit = fft_jit_kernel_create_conv_fused_fp64(Np, (const double *)hPad);
        if (jit == NULL) {
            free(hPad);
            return NULL;
        }
        jit_fn = fft_jit_kernel_fn(jit);
        if (jit_fn == NULL) {
            fft_jit_kernel_destroy(jit);
            free(hPad);
            return NULL;
        }
    }

    /* correction leg (mode 3): tail correlation of the reversed x-tail
     * with HrH[p] = h[N-1-p] (p <= L); tiny L is computed directly in the
     * wrapper (h64 copy), larger L gets its own conv weld at Np2. */
    fft_jit_kernel *jit2 = NULL;
    fft_jit_fn_t jit2_fn = NULL;
    double complex *h64 = NULL;
    if (mode == 3) {
        if (corrL + 1 > 8) {
            double complex *h2 =
                (double complex *)malloc((size_t)Np2 * sizeof(*h2));
            if (h2 == NULL) {
                fft_jit_kernel_destroy(jit);
                free(hPad);
                return NULL;
            }
            for (int i = 0; i < Np2; i++) {
                h2[i] = (i <= corrL) ? h[N - 1 - i] : (0.0 + 0.0 * I);
            }
            jit2 = fft_jit_kernel_create_conv_fused_fp64(
                Np2, (const double *)h2);
            free(h2);
            if (jit2 == NULL) {
                fft_jit_kernel_destroy(jit);
                free(hPad);
                return NULL;
            }
            jit2_fn = fft_jit_kernel_fn(jit2);
            if (jit2_fn == NULL) {
                fft_jit_kernel_destroy(jit2);
                fft_jit_kernel_destroy(jit);
                free(hPad);
                return NULL;
            }
        } else {
            h64 = (double complex *)malloc((size_t)(corrL + 1) *
                                           sizeof(*h64));
            if (h64 == NULL) {
                fft_jit_kernel_destroy(jit);
                free(hPad);
                return NULL;
            }
            for (int p = 0; p <= corrL; p++) {
                h64[p] = h[N - 1 - p];
            }
        }
    } else if (mode == 0) {
        h64 = (double complex *)malloc((size_t)N * sizeof(*h64));
        if (h64 == NULL) {
            free(hPad);
            return NULL;
        }
        for (int i = 0; i < N; i++) {
            h64[i] = h[i];
        }
    }

    double complex *buf64 = NULL;
    double complex *buf2 = NULL;
    if (mode == 2 || mode == 3 || mode == 4) {
        buf64 = (double complex *)malloc((size_t)Np * sizeof(*buf64));
        if (buf64 == NULL) {
            fft_jit_kernel_destroy(jit2);
            fft_jit_kernel_destroy(jit);
            free(h64);
            free(hPad);
            return NULL;
        }
    }
    if (mode == 3) {
        size_t n2 = (size_t)((corrL + 1 > 8) ? Np2 : (corrL + 1));
        buf2 = (double complex *)malloc(n2 * sizeof(*buf2));
        if (buf2 == NULL) {
            free(buf64);
            fft_jit_kernel_destroy(jit2);
            fft_jit_kernel_destroy(jit);
            free(h64);
            free(hPad);
            return NULL;
        }
    } else if (mode == 4) {
        buf2 = (double complex *)malloc((size_t)N * sizeof(*buf2));
        if (buf2 == NULL) {
            free(buf64);
            fft_jit_kernel_destroy(jit2);
            fft_jit_kernel_destroy(jit);
            free(h64);
            free(hPad);
            return NULL;
        }
    }
    free(hPad);
    fft_jit_kernel_destroy(weldJit); /* NULL unless raced away (defensive) */

    struct fft_plan *plan = (struct fft_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        free(buf2);
        free(buf64);
        fft_jit_kernel_destroy(jit2);
        fft_jit_kernel_destroy(jit);
        free(h64);
        return NULL;
    }
    plan->N = N;
    plan->fp64 = 1; /* defensive: fp32 fft_execute is a no-op here */
    plan->conv64 = 1;
    plan->c64_mode = mode;
    plan->Np = Np;
    plan->buf64 = buf64;
    plan->Np2 = Np2;
    plan->corrL = corrL;
    plan->buf2_64 = buf2;
    plan->h64 = h64;
    plan->jit = jit;
    plan->jit_fn = jit_fn;
    plan->jit2 = jit2;
    plan->jit2_fn = jit2_fn;
    plan->olaPieces = olaPieces;
    plan->olaP = olaP;
    plan->refcount = 1;

    if (getenv("FFT_CONV_LOG") != NULL) {
        static const char *modeName[6] = {"tiny", "direct", "pad-single",
                                          "pad-corr", "ola", "weld"};
        fprintf(stderr,
                "[conv64] N=%d route=%s Np=%d Np2=%d L=%d pieces=%d\n", N,
                modeName[mode], Np, Np2, corrL, olaPieces);
    }

    pthread_mutex_lock(&g_cache_lock);
    hit = cache_lookup_conv64_locked(N);
    if (hit != NULL) {
        pthread_mutex_unlock(&g_cache_lock);
        free(plan->buf2_64);
        free(plan->buf64);
        fft_jit_kernel_destroy(plan->jit2);
        fft_jit_kernel_destroy(plan->jit);
        free(plan->h64);
        free(plan);
        return hit;
    }
    plan->next = g_cache_head_conv64;
    g_cache_head_conv64 = plan;
    pthread_mutex_unlock(&g_cache_lock);
    return plan;
#else
    /* No silent fp64 conv without the JIT kernel. */
    return NULL;
#endif
}

/* Execute the fp64 cyclic convolution (draft {#API-FFT-011}).  One call
 * completes forward -> pointwise -> inverse (fused at the store/load
 * boundaries inside the weld) plus the arbitrary-N semantic layer; see
 * fft_plan_create_conv_fp64 for the route table. */
void fft_execute_conv_fp64(fft_plan *plan, double complex *io) {
    if (plan == NULL || io == NULL || !plan->conv64) {
        return;
    }
#if !defined(FFT_JIT_DISABLE)
    const int N = plan->N;
    switch (plan->c64_mode) {
    case 0: {
        /* degenerate N = 1 / 2: exact direct cyclic conv (no transform is
         * meaningful at these sizes; the weld starts at Np >= 4). */
        const double complex *hh = plan->h64;
        if (hh == NULL) {
            return; /* fail-fast, no silent fallback */
        }
        if (N == 1) {
            io[0] = io[0] * hh[0];
        } else {
            const double complex x0 = io[0], x1 = io[1];
            io[0] = x0 * hh[0] + x1 * hh[1];
            io[1] = x0 * hh[1] + x1 * hh[0];
        }
        return;
    }
    case 1:
        /* pow2 N >= 4: the weld runs DIRECTLY on the caller's io. */
        if (plan->jit_fn == NULL || plan->jit == NULL) {
            return;
        }
        plan->jit_fn(io);
        return;
    case 5:
        /* topic jit-codelet-weld: original-size weld DIRECTLY on io —
         * exact N-point cyclic semantics, no padding, no wrap add. */
        if (plan->jit_fn == NULL || plan->jit == NULL) {
            return;
        }
        plan->jit_fn(io);
        return;
    case 2: {
        /* pad-single: weld at Np >= 2N-1 (exact linear support); y[m] =
         * z[m] + z[m+N]. */
        if (plan->jit_fn == NULL || plan->jit == NULL ||
            plan->buf64 == NULL) {
            return;
        }
        double complex *buf = plan->buf64;
        const int Np = plan->Np;
        for (int i = 0; i < N; i++) {
            buf[i] = io[i];
        }
        for (int i = N; i < Np; i++) {
            buf[i] = 0.0 + 0.0 * I;
        }
        plan->jit_fn(buf);
        for (int i = 0; i < N; i++) {
            io[i] = buf[i] + buf[i + N];
        }
        return;
    }
    case 3: {
        /* pad-corr: aliased main weld @Np + exact wrapped-tail
         * reconstruction.  z = Np-cyclic conv of the padded pair;
         * w[s] = y_lin[s+Np] for s <= L = 2N-2-Np (tail correlation of
         * the reversed x-tail with h reversed head); then
         *   y[m] = (z[m] - [m<=L] w[m]) + (m+N < Np ? z[m+N]
         *                                             : [s<=L] w[s]),
         *   s = m+N-Np. */
        if (plan->jit_fn == NULL || plan->jit == NULL ||
            plan->buf64 == NULL || plan->buf2_64 == NULL) {
            return;
        }
        const int Np = plan->Np;
        const int L = plan->corrL;
        const int c = Np - N + 1;
        double complex *buf = plan->buf64;
        double complex *buf2 = plan->buf2_64;
        for (int i = 0; i < N; i++) {
            buf[i] = io[i];
        }
        for (int i = N; i < Np; i++) {
            buf[i] = 0.0 + 0.0 * I;
        }
        plan->jit_fn(buf); /* buf = z */
        if (plan->jit2_fn != NULL) {
            /* wbuf[q] = XcRev[q] = x[N-1-q] (q <= L); weld = linear conv
             * with HrH (support 2L+1 <= Np2, un-aliased); w[s] = g[L-s] */
            for (int q = 0; q <= L; q++) {
                buf2[q] = io[N - 1 - q];
            }
            for (int q = L + 1; q < plan->Np2; q++) {
                buf2[q] = 0.0 + 0.0 * I;
            }
            plan->jit2_fn(buf2); /* buf2 = g */
            for (int m = 0; m < N; m++) {
                double complex y = buf[m];
                if (m <= L) {
                    y -= buf2[L - m];
                }
                const long t = (long)m + N;
                if (t < Np) {
                    y += buf[t];
                } else {
                    const long s = t - Np;
                    if (s <= L) {
                        y += buf2[L - s];
                    }
                }
                io[m] = y;
            }
        } else {
            /* tiny L (<= 7): direct tail correlation, w[s] = sum_p
             * x[c+s+p] * h[N-1-p], p <= L-s (h64 = reversed head) */
            const double complex *hh = plan->h64;
            if (hh == NULL) {
                return;
            }
            for (int m = 0; m < N; m++) {
                double complex y = buf[m];
                if (m <= L) {
                    double complex w = 0.0 + 0.0 * I;
                    for (int p = 0; p <= L - m; p++) {
                        w += io[c + m + p] * hh[p];
                    }
                    y -= w;
                }
                const long t = (long)m + N;
                if (t < Np) {
                    y += buf[t];
                } else {
                    const long s = t - Np;
                    if (s <= L) {
                        double complex w = 0.0 + 0.0 * I;
                        for (int p = 0; p <= L - (int)s; p++) {
                            w += io[c + (int)s + p] * hh[p];
                        }
                        y += w;
                    }
                }
                io[m] = y;
            }
        }
        return;
    }
    default: {
        /* OLA multi-block (N > 24576): x = sum of pieces of length
         * P = Np-N+1; each piece's linear conv with h fits un-aliased in
         * one Np weld; scatter-add mod N.  buf2 = x copy. */
        if (plan->jit_fn == NULL || plan->jit == NULL ||
            plan->buf64 == NULL || plan->buf2_64 == NULL) {
            return;
        }
        const int Np = plan->Np;
        const int P = plan->olaP;
        double complex *buf = plan->buf64;
        double complex *xcopy = plan->buf2_64;
        for (int i = 0; i < N; i++) {
            xcopy[i] = io[i];
            io[i] = 0.0 + 0.0 * I;
        }
        for (int piece = 0; piece < plan->olaPieces; piece++) {
            const int base = piece * P;
            const int plen =
                (base + P <= N) ? P : (N - base); /* last piece remainder */
            if (plen <= 0) {
                break;
            }
            for (int q = 0; q < plen; q++) {
                buf[q] = xcopy[base + q];
            }
            for (int q = plen; q < Np; q++) {
                buf[q] = 0.0 + 0.0 * I;
            }
            plan->jit_fn(buf); /* linear conv piece*h, support plen+N-1 */
            for (int t = 0; t < plen + N - 1; t++) {
                io[(base + t) % N] += buf[t];
            }
        }
        return;
    }
    }
#endif
}

/* topic jit-fp64-inverse (amend resume 2026-09-17, draft {#API-FFT-009} /
 * {#BEH-FFT-010}): opt-in fp64 INVERSE plan.  Family-only
 * (N = 2^n, n <= 15 / N <= 32768); own plan table (never collides with
 * the fp32 inverse, the fp64 forward, or the fp64 conv plans).
 *
 *   N >= 32 : the FUSED single-kernel inverse (boundary-sunk conjugate
 *             trick — entry conj at the stage-0 gather load boundary,
 *             exit conj*(1/N) at the final store boundary; no wrapper
 *             sweeps, no inverse twiddle table).  Emit failure / OOM →
 *             NULL (fail-fast, no silent fallback to any other path).
 *   N < 32  : the Trunk small-N floor — the promoted fp64 forward kernel
 *             + the host's independent conj / conj*invN sweeps (the
 *             retired three-segment wrapper; the sched machinery starts
 *             at N >= 32 in the Trunk forward too).
 *
 * Returns NULL on illegal N / emit failure / OOM.  MUST NOT silently
 * fall back to the fp64 forward path, the fp32 inverse, adapt, or any
 * other plan. */
fft_plan *fft_plan_create_fp64_inverse(int N) {
    if (N < 1 || N > 32768) {
        return NULL;
    }
#if !defined(FFT_JIT_DISABLE)
    /* topic jit-anyN-fwd-inv (draft {#API-FFT-013} N-domain revision of
     * {#API-FFT-009}): pow2 N = the ZERO-CHANGE family guard (fused
     * kernel N >= 32 + the N < 32 host wrapper floor, byte-for-byte).
     * Non-pow2 N: the coverage-set档 — the WSK boundary-conjugate
     * single-transform inverse; fail-closed NULL out of set (MUST NOT
     * fall back to fp32 inverse, fp64 forward, adapt, pad, FFTW). */
    if (fft_jit_family_member(N)) {
        pthread_mutex_lock(&g_cache_lock);
        fft_plan *hit = cache_lookup_inv64_locked(N);
        pthread_mutex_unlock(&g_cache_lock);
        if (hit != NULL) {
            return hit;
        }

        int fused = 0;
        fft_jit_kernel *jit = NULL;
        if (N >= 32) {
            jit = fft_jit_kernel_create_inverse_fused_fp64(N);
            if (jit == NULL || fft_jit_kernel_fn(jit) == NULL) {
                fft_jit_kernel_destroy(jit);
                return NULL;
            }
            fused = 1;
        } else {
            /* small-N floor: the promoted fp64 forward kernel
             * (SmallEmitter / identity tier) wrapped by this file's conj
             * sweeps at execute. */
            jit = fft_jit_kernel_create_fp64(N);
            if (jit == NULL || fft_jit_kernel_fn(jit) == NULL) {
                fft_jit_kernel_destroy(jit);
                return NULL;
            }
        }
        fft_jit_fn_t jit_fn = fft_jit_kernel_fn(jit);

        struct fft_plan *plan = (struct fft_plan *)calloc(1, sizeof(*plan));
        if (plan == NULL) {
            fft_jit_kernel_destroy(jit);
            return NULL;
        }
        plan->N = N;
        plan->fp64 = 1; /* defensive: fp32 fft_execute is a no-op here */
        plan->inv64 = 1;
        plan->fused_inv = fused;
        plan->jit = jit;
        plan->jit_fn = jit_fn;
        plan->refcount = 1;

        pthread_mutex_lock(&g_cache_lock);
        hit = cache_lookup_inv64_locked(N);
        if (hit != NULL) {
            pthread_mutex_unlock(&g_cache_lock);
            fft_jit_kernel_destroy(jit);
            free(plan);
            return hit;
        }
        plan->next = g_cache_head_inv64;
        g_cache_head_inv64 = plan;
        pthread_mutex_unlock(&g_cache_lock);
        return plan;
    }

    /* ---- non-pow2: the coverage-set档 (inv64 table, draft
     * {#API-FFT-013}; pow2 / non-pow2 key spaces disjoint) ---- */
    if (!anyn_covered(N)) {
        return NULL; /* fail-closed; no adapt / pad / FFTW route */
    }
    pthread_mutex_lock(&g_cache_lock);
    fft_plan *hit = cache_lookup_inv64_locked(N);
    pthread_mutex_unlock(&g_cache_lock);
    if (hit != NULL) {
        return hit;
    }

    fft_jit_kernel *jit = NULL;
    fft_jit_fn_t jit_fn = NULL;
    int floor = 0;
    if (anyn_is_bare_prime_or_tiny(N)) {
        floor = 1; /* wrapper floor (bare prime / tiny N <= 3) */
    } else {
        if (!fft_jit_has_avx2_fma()) {
            return NULL; /* CON-FFT-002: fail-closed without AVX2+FMA3 */
        }
        jit = fft_jit_kernel_create_anyN_inverse_fp64(N);
        if (jit == NULL || fft_jit_kernel_fn(jit) == NULL) {
            fft_jit_kernel_destroy(jit);
            return NULL; /* emit failure / unviable shape: fail-closed */
        }
        jit_fn = fft_jit_kernel_fn(jit);
    }

    struct fft_plan *plan = (struct fft_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        fft_jit_kernel_destroy(jit);
        return NULL;
    }
    plan->N = N;
    plan->fp64 = 1; /* defensive: fp32 fft_execute is a no-op here */
    plan->inv64 = 1;
    plan->anyn_floor = floor;
    plan->fused_inv = (jit != NULL) ? 1 : 0; /* anyN inverse kernels are
                                                single-kernel fused forms */
    plan->jit = jit;
    plan->jit_fn = jit_fn;
    plan->refcount = 1;

    pthread_mutex_lock(&g_cache_lock);
    hit = cache_lookup_inv64_locked(N);
    if (hit != NULL) {
        pthread_mutex_unlock(&g_cache_lock);
        fft_jit_kernel_destroy(jit);
        free(plan);
        return hit;
    }
    plan->next = g_cache_head_inv64;
    g_cache_head_inv64 = plan;
    pthread_mutex_unlock(&g_cache_lock);
    return plan;
#else
    /* No silent fp64 inverse without the JIT kernel. */
    return NULL;
#endif
}

/* Execute the opt-in fp64 inverse (draft {#API-FFT-009}).  One call
 * computes IFFT(io) with the boundary-sunk conjugate / 1/N — the fused
 * kernel for N >= 32, the wrapper floor below.  In-place, natural order
 * (matches same-N upstream FFTW_BACKWARD / N). */
void fft_execute_fp64_inverse(fft_plan *plan, double complex *io) {
    if (plan == NULL || io == NULL || !plan->inv64) {
        return;
    }
#if !defined(FFT_JIT_DISABLE)
    /* topic jit-anyN-fwd-inv: coverage-set wrapper floor (bare prime /
     * tiny N) — host O(N^2) IDFT, oracle-tolerance exact (conv tiny
     * precedent). */
    if (plan->anyn_floor) {
        anyn_floor_idft(io, plan->N);
        return;
    }
    if (plan->jit_fn == NULL || plan->jit == NULL) {
        return; /* fail-fast, no silent fallback */
    }
    if (plan->fused_inv) {
        plan->jit_fn(io);
        return;
    }
    /* small-N floor: the retired three-segment conjugate-trick wrapper. */
    const int N = plan->N;
    const double invN = 1.0 / (double)N;
    for (int i = 0; i < N; i++) {
        io[i] = conj(io[i]);
    }
    plan->jit_fn(io);
    for (int i = 0; i < N; i++) {
        io[i] = conj(io[i]) * invN;
    }
#endif
}

/*
 * topic jit-fp64-linear-conv (draft {#API-FFT-012} / {#BEH-FFT-017}):
 * opt-in fp64 block-pinned overlap-save LINEAR convolution.  Family-only
 * block N (2^n, n <= 15 / N <= 32768), kernel length 0 < nh <= N (nh is
 * explicit — C cannot infer it from a bare const double complex *);
 * stream length nx is NOT a contract parameter (the caller streams one
 * length-N window per execute, sliding by hop = N-nh+1 over the
 * conceptually [0^{nh-1} | x | 0-tail]-padded stream — fp32
 * API-FFT-006 semantics, fp64 independent symbols).
 *
 * Plan-time: h zero-padded to N on a scratch copy, H[k] = FFT(h_pad)
 * materialized inside the OLS weld create (promoted forward machinery —
 * h NOT clobbered; hMul/hSwap tables separate from the twiddle LUT, no
 * fourth twiddle subsystem; no exp at plan time).  The weld carries the
 * OLS boundary edits (ROTATED half-A gather table + valid-prefix half-B
 * final stores; see src/jit SchedEmitterO).  The plan-held overlap
 * state (retained nh-1 input tail + stream position counter) is
 * zero-initialized here.  N = 1 / 2 keep a tiny direct wrapper (below
 * the weld floor N >= 4 — no transform is meaningful there).
 *
 * Returns NULL on illegal N / nh<=0 / nh>N / NULL h / emit / OOM.
 * MUST NOT silently fall back to the fp64 forward path, fp64 cyclic
 * convolution, any fp32 path, adapt, or route to FFTW.
 */
fft_plan *fft_plan_create_linear_conv_fp64(int N, const double complex *h,
                                           int nh) {
    if (N < 1 || N > 32768 || (N & (N - 1)) != 0) {
        return NULL;
    }
    if (h == NULL || nh <= 0 || nh > N) {
        return NULL;
    }
#if !defined(FFT_JIT_DISABLE)
    pthread_mutex_lock(&g_cache_lock);
    fft_plan *hit = cache_lookup_lin64_locked(N, nh);
    pthread_mutex_unlock(&g_cache_lock);
    if (hit != NULL) {
        return hit;
    }

    const int tiny = (N < 4);
    fft_jit_kernel *jit = NULL;
    fft_jit_fn_t jit_fn = NULL;
    fft_jit_ols_fn_t jit_ols = NULL;
    double complex *h64 = NULL;
    if (!tiny) {
        /* zero-pad h to N on a scratch copy (h NOT clobbered; the H
         * materialization happens INSIDE the weld create on its own
         * scratch — this buffer only transports the padded h). */
        double complex *hPad =
            (double complex *)malloc((size_t)N * sizeof(*hPad));
        if (hPad == NULL) {
            return NULL;
        }
        for (int i = 0; i < N; i++) {
            hPad[i] = (i < nh) ? h[i] : (0.0 + 0.0 * I);
        }
        jit = fft_jit_kernel_create_ols_fused_fp64(N, (const double *)hPad,
                                                   nh);
        free(hPad);
        if (jit == NULL) {
            return NULL; /* emit/OOM fail-fast, no silent fallback */
        }
        jit_fn = fft_jit_kernel_fn(jit);
        jit_ols = fft_jit_kernel_ols_stream_fn(jit);
        if (jit_fn == NULL || jit_ols == NULL) {
            fft_jit_kernel_destroy(jit);
            return NULL;
        }
    } else {
        /* tiny floor: time-domain padded kernel copy (direct cyclic
         * window conv, valid prefix out — no transform, no kernel). */
        h64 = (double complex *)malloc((size_t)N * sizeof(*h64));
        if (h64 == NULL) {
            return NULL;
        }
        for (int i = 0; i < N; i++) {
            h64[i] = (i < nh) ? h[i] : (0.0 + 0.0 * I);
        }
    }

    /* plan-held overlap state: retained nh-1 input tail (zero-
     * initialized head prefix) + stream position bookkeeping. */
    double complex *tail =
        (double complex *)calloc((size_t)(nh > 1 ? nh - 1 : 1),
                                 sizeof(*tail));
    if (tail == NULL) {
        fft_jit_kernel_destroy(jit);
        free(h64);
        return NULL;
    }

    struct fft_plan *plan = (struct fft_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        free(tail);
        fft_jit_kernel_destroy(jit);
        free(h64);
        return NULL;
    }
    plan->N = N;
    plan->fp64 = 1; /* defensive: fp32 fft_execute is a no-op here */
    plan->lin64 = 1;
    plan->ols_tiny = tiny ? 1 : 0;
    plan->nh = nh;
    plan->h64 = h64;
    plan->ols_tail = tail;
    plan->ols_pos = 0;
    plan->jit = jit;
    plan->jit_fn = jit_fn;
    plan->jit_ols = jit_ols;
    plan->refcount = 1;

    pthread_mutex_lock(&g_cache_lock);
    hit = cache_lookup_lin64_locked(N, nh);
    if (hit != NULL) {
        pthread_mutex_unlock(&g_cache_lock);
        free(plan->ols_tail);
        free(plan->h64);
        fft_jit_kernel_destroy(plan->jit);
        free(plan);
        return hit;
    }
    plan->next = g_cache_head_lin64;
    g_cache_head_lin64 = plan;
    pthread_mutex_unlock(&g_cache_lock);
    return plan;
#else
    /* No silent fp64 linear conv without the JIT kernel. */
    return NULL;
#endif
}

/* Execute the opt-in fp64 OLS linear convolution on ONE window
 * (draft {#API-FFT-012}): io[0..N-1] holds the length-N window on
 * entry; on return io[0..N-nh] holds the N-nh+1 valid linear-conv
 * samples (the OLS add/drop bookkeeping is INSIDE the kernel at the
 * store boundary — the rotated gather lands the valid segment at the
 * natural front) and the trailing nh-1 positions are a discard zone
 * (unpromised; not written).  Overlap state (retained tail + stream
 * position) updates inside the plan. */
void fft_execute_linear_conv_fp64(fft_plan *plan, double complex *io) {
    if (plan == NULL || io == NULL || !plan->lin64) {
        return;
    }
#if !defined(FFT_JIT_DISABLE)
    const int N = plan->N;
    const int nh = plan->nh;
    /* plan-held overlap bookkeeping (retained window tail + stream
     * position); callers MUST NOT add/drop samples themselves. */
    if (plan->ols_tail != NULL && nh > 1) {
        memcpy(plan->ols_tail, io + (N - nh + 1),
               (size_t)(nh - 1) * sizeof(*plan->ols_tail));
    }
    plan->ols_pos++;
    if (plan->ols_tiny) {
        /* N < 4 floor: direct cyclic window conv, valid prefix out
         * (identical semantics to the weld — rotated valid prefix). */
        const double complex *hh = plan->h64;
        if (hh == NULL) {
            return; /* fail-fast, no silent fallback */
        }
        double complex z[2];
        for (int m = 0; m < N; m++) {
            double complex acc = 0.0 + 0.0 * I;
            for (int j = 0; j < nh; j++) {
                acc += hh[j] * io[(m - j + N) % N];
            }
            z[m] = acc;
        }
        for (int m = nh - 1; m < N; m++) {
            io[m - (nh - 1)] = z[m];
        }
        return;
    }
    if (plan->jit_fn == NULL || plan->jit == NULL) {
        return; /* fail-fast, no silent fallback */
    }
    plan->jit_fn(io);
#endif
}

/*
 * topic jit-fp64-linear-conv INTERNAL stream face — NOT a public API
 * (bench / topic drivers only; MUST NOT leak into include/fft.h).
 * One call sweeps nblocks OLS blocks of the PADDED stream P =
 * [0^{nh-1} | x | 0-tail] (P sized >= (nblocks-1)*hop + N complex):
 * windows form inside the emitted block loop (rotated gather at
 * rdi = P + b*hop), valid segments store straight to y + b*hop at the
 * inverse final-store boundary (the O-ii form (a) subject).  y MUST be
 * sized >= nblocks*hop + (nh-1) slack (the last block's valid stores
 * stay within y[0..ylen-1] of the true output; the slack only hosts
 * write-mode discard residue).  nblocks >= 1.  Form (b) drivers call
 * this with nblocks == 1 per block (window pointer = P + b*hop).
 */
void fft_ols_stream_fp64(fft_plan *plan, const double complex *P,
                         double complex *y, long nblocks) {
    if (plan == NULL || P == NULL || y == NULL || nblocks < 1) {
        return;
    }
#if !defined(FFT_JIT_DISABLE)
    if (plan->ols_tiny || plan->jit_ols == NULL || plan->jit == NULL) {
        /* tiny floor / missing stream entry: drive the single-block
         * contract face per block (semantics identical). */
        const int N = plan->N;
        const int hop = N - plan->nh + 1;
        double complex *blk =
            (double complex *)malloc((size_t)N * sizeof(*blk));
        if (blk == NULL) {
            return;
        }
        for (long b = 0; b < nblocks; b++) {
            const double complex *win = P + (size_t)b * (size_t)hop;
            memcpy(blk, win, (size_t)N * sizeof(*blk));
            fft_execute_linear_conv_fp64(plan, blk);
            memcpy(y + (size_t)b * (size_t)hop, blk,
                   (size_t)hop * sizeof(*y));
        }
        free(blk);
        return;
    }
    plan->ols_pos += nblocks;
    plan->jit_ols((void *)P, (void *)y, nblocks);
#endif
}

/* topic jit-joint-search-batched (L-F, draft {#API-FFT-010}): opt-in
 * batched Gaussian-windowed forward plan.  One plan carries (N, B); the
 * (placement x decomposition x tile x batch form) shape is chosen by the
 * plan-time joint search.  Own table (key = (N, B)); MUST NOT collide
 * with the six legacy contracts or fp64.  Fail-fast: illegal (N, B) /
 * emit / OOM -> NULL, no silent fallback. */
fft_plan *fft_plan_create_windowed_batch(int N, int B) {
    if (N < 1 || N > 32768 || (N & (N - 1)) != 0) {
        return NULL;
    }
    if (B < 1 || B > (1 << 20)) {
        return NULL;
    }
#if !defined(FFT_JIT_DISABLE)
    pthread_mutex_lock(&g_cache_lock);
    fft_plan *hit = cache_lookup_batch_locked(N, B);
    pthread_mutex_unlock(&g_cache_lock);
    if (hit != NULL) {
        return hit;
    }

    fft_jit_kernel *jit = fft_jit_kernel_create_windowed_batch(N, B);
    if (jit == NULL) {
        return NULL;
    }
    const fft_jit_search_stats *st = fft_jit_batch_search_last();
    const int single = fft_jit_kernel_is_batch_single(jit);
    if (single ? fft_jit_kernel_fn2(jit) == NULL
               : fft_jit_kernel_fn(jit) == NULL) {
        fft_jit_kernel_destroy(jit);
        return NULL;
    }
    const int place = (st != NULL) ? st->place : FFT_PLACE_LOAD;
    const int form = (st != NULL) ? st->form
                                  : (single ? FFT_BATCH_FORM_SINGLE
                                            : FFT_BATCH_FORM_LOOP);

    /* standalone / presweep placement: the host sweeps the window (same
     * G as API-FFT-003: sigma = N/8, sum = 1) before the plain kernel
     * (topic R2: presweep = ONE batch-level streaming pass). */
    float *G2 = NULL;
    if (place == FFT_PLACE_STANDALONE || place == FFT_PLACE_PRESWEEP) {
        float *G = (float *)malloc((size_t)N * sizeof(float));
        G2 = (float *)malloc((size_t)(2 * N) * sizeof(float));
        if (G == NULL || G2 == NULL ||
            fft_gauss_window_fill(N, G) != 0) {
            free(G);
            free(G2);
            fft_jit_kernel_destroy(jit);
            return NULL;
        }
        for (int i = 0; i < N; i++) {
            G2[2 * i] = G[i];
            G2[2 * i + 1] = G[i];
        }
        free(G);
    }

    struct fft_plan *plan = (struct fft_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        free(G2);
        fft_jit_kernel_destroy(jit);
        return NULL;
    }
    plan->N = N;
    plan->batch = 1;
    plan->bcount = B;
    plan->batch_place = place;
    plan->batch_form = form;
    plan->G2 = G2;
    plan->fp64 = 0;
    plan->jit = jit;
    plan->jit_fn = single ? NULL : fft_jit_kernel_fn(jit);
    plan->jit_fn2 = single ? fft_jit_kernel_fn2(jit) : NULL;
    plan->jit_fn2c = single ? fft_jit_kernel_fn2c(jit) : NULL;
    plan->refcount = 1;

    pthread_mutex_lock(&g_cache_lock);
    hit = cache_lookup_batch_locked(N, B);
    if (hit != NULL) {
        pthread_mutex_unlock(&g_cache_lock);
        free(plan->G2);
        fft_jit_kernel_destroy(jit);
        free(plan);
        return hit;
    }
    plan->next = g_cache_head_batch;
    g_cache_head_batch = plan;
    pthread_mutex_unlock(&g_cache_lock);
    return plan;
#else
    /* No silent batched windowed forward without the JIT kernel. */
    return NULL;
#endif
}

/* topic jit-joint-search-batched: execute the batched windowed-forward
 * plan in-place over B blocks (block b at io + b*stride, stride counted
 * in float complex ELEMENTS).  One call processes ALL B blocks; blocks
 * are independent, natural order in/out; the buffer stays caller-owned.
 * NULL plan / NULL io / non-batched plan / stride < N -> no-op
 * (defensive, no partial writes). */
void fft_execute_windowed_batch(fft_plan *plan, float complex *io,
                                int stride) {
    if (plan == NULL || io == NULL || !plan->batch || plan->fp64) {
        return;
    }
    if (stride < plan->N) {
        return; /* overlapping blocks: illegal, no-op */
    }
#if !defined(FFT_JIT_DISABLE)
    const int N = plan->N;
    if (plan->batch_place == FFT_PLACE_STANDALONE) {
        /* wrapper-level independent window sweep (the search's null
         * hypothesis placement), then the unfused batch kernel. */
        const float *g2 = plan->G2;
        for (int b = 0; b < plan->bcount; b++) {
            float *p = (float *)(io + (size_t)b * (size_t)stride);
            for (int i = 0; i < 2 * N; i++) {
                p[i] *= g2[i];
            }
        }
    } else if (plan->batch_place == FFT_PLACE_PRESWEEP) {
        /* topic R2 (F5/L-G): ONE batch-level streaming pre-sweep before
         * the PLAIN batch kernel — the composed leg's window-sweep
         * shape riding a single execute.  The contiguous stride==N
         * layout takes the fused two-block-wide pass; padded stride > N
         * keeps the per-block loop. */
        const float *g2 = plan->G2;
        if (stride == N && plan->G2 != NULL) {
            float *p = (float *)io;
            const long n2 = 2L * (long)N;
            const long tot = n2 * (long)plan->bcount;
            long i = 0;
            for (; i + 2 * n2 <= tot; i += 2 * n2) {
                for (long k = 0; k < n2; k++) {
                    p[i + k] *= g2[k];
                    p[i + k + n2] *= g2[k];
                }
            }
            for (; i < tot; i += n2) {
                for (long k = 0; k < n2; k++) {
                    p[i + k] *= g2[k];
                }
            }
        } else {
            for (int b = 0; b < plan->bcount; b++) {
                float *p = (float *)(io + (size_t)b * (size_t)stride);
                for (int k = 0; k < 2 * N; k++) {
                    p[k] *= g2[k];
                }
            }
        }
    }
    if (plan->batch_form == FFT_BATCH_FORM_SLIM &&
        plan->jit_fn2 != NULL) {
        /* topic R2 (F5/L-G): the slim single-core form's dual entry —
         * the plan-time-specialized contiguous stride==N fast path, or
         * the generic runtime-stride entry for padded stride > N. */
        if (stride == N && plan->jit_fn2c != NULL) {
            plan->jit_fn2c(io);
        } else {
            plan->jit_fn2(io, (long)stride);
        }
        return;
    }
    if (plan->batch_form == FFT_BATCH_FORM_SINGLE &&
        plan->jit_fn2 != NULL) {
        plan->jit_fn2(io, (long)stride);
        return;
    }
    if (plan->batch_form == FFT_BATCH_FORM_LOOP && plan->jit_fn != NULL) {
        for (int b = 0; b < plan->bcount; b++) {
            plan->jit_fn(io + (size_t)b * (size_t)stride);
        }
    }
#endif
}

fft_plan *fft_plan_create_conv(int N, const float complex *h) {
    if (N < 1 || h == NULL) {
        return NULL;
    }
#if !defined(FFT_JIT_DISABLE)
    /* Conv is family-only.  Non-family / NULL h / emit / OOM → NULL.
     * MUST NOT fall back to adapt or a silent forward plan. */
    if (!fft_jit_family_member(N)) {
        return NULL;
    }

    pthread_mutex_lock(&g_cache_lock);
    fft_plan *hit = cache_lookup_locked(N, 0, 0, 1, 0, 0);
    pthread_mutex_unlock(&g_cache_lock);
    if (hit != NULL) {
        return hit;
    }

    fft_jit_kernel *jit = fft_jit_kernel_create(N);
    if (jit == NULL) {
        return NULL;
    }
    fft_jit_fn_t jit_fn = fft_jit_kernel_fn(jit);
    if (jit_fn == NULL) {
        fft_jit_kernel_destroy(jit);
        return NULL;
    }

    /* Plan-time H[k] = FFT(h): reuse the already-promoted forward kernel on
     * a scratch copy of h.  Do NOT clobber the caller's h; do NOT call exp;
     * do NOT emit a new twiddle / conv kernel.  H is a separate table from
     * the twiddle LUT and the gauss G table. */
    float complex *scratch =
        (float complex *)malloc((size_t)N * sizeof(*scratch));
    if (scratch == NULL) {
        fft_jit_kernel_destroy(jit);
        return NULL;
    }
    float complex *H = (float complex *)malloc((size_t)N * sizeof(*H));
    if (H == NULL) {
        free(scratch);
        fft_jit_kernel_destroy(jit);
        return NULL;
    }
    for (int i = 0; i < N; i++) {
        scratch[i] = h[i];
    }
    jit_fn(scratch); /* forward: scratch = FFT(h) */
    for (int i = 0; i < N; i++) {
        H[i] = scratch[i];
    }
    free(scratch);

    /* topic jit-ps-sched-fuse: replace the forward kernel with the FUSED
     * conv kernel (plan-time H copy, one execute welds forward*H +
     * inverse).  Emit failure / N < 32 keeps the three-segment path.
     * topic jit-joint-search-batched (L-E): the fused kernel now comes
     * from the plan-time JOINT placement search (decomposition x tile x
     * {store boundary, load boundary, standalone}); store = the base
     * fused form, load = H rows on the inverse-half gather, standalone =
     * the searched PLAIN kernel + this wrapper's three-segment path
     * (fused stays 0).  The public contract is unchanged for every
     * placement. */
    int fused = 0;
#if !defined(FFT_JIT_DISABLE)
    if (fused_conv_enabled() && N >= 32) {
        fft_jit_kernel *fj = fft_jit_kernel_create_conv_searched(
            N, (const float *)H, -1);
        if (fj != NULL && fft_jit_kernel_fn(fj) != NULL) {
            const fft_jit_search_stats *st = fft_jit_conv_search_last();
            fft_jit_kernel_destroy(jit);
            jit = fj;
            jit_fn = fft_jit_kernel_fn(fj);
            fused = (st != NULL && st->place == FFT_PLACE_STANDALONE) ? 0
                                                                      : 1;
        } else if (fj != NULL) {
            fft_jit_kernel_destroy(fj);
        }
    }
#endif

    struct fft_plan *plan = (struct fft_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        free(H);
        fft_jit_kernel_destroy(jit);
        return NULL;
    }
    plan->N = N;
    plan->windowed = 0;
    plan->inverse = 0;
    plan->conv = 1;
    plan->fused_conv = fused;
    plan->linear = 0;
    plan->nh = 0;
    plan->fp64 = 0;
    plan->H = H;
    plan->adapt = NULL;
    plan->jit = jit;
    plan->jit_fn = jit_fn;
    plan->refcount = 1;

    pthread_mutex_lock(&g_cache_lock);
    hit = cache_lookup_locked(N, 0, 0, 1, 0, 0);
    if (hit != NULL) {
        pthread_mutex_unlock(&g_cache_lock);
        free(H);
        fft_jit_kernel_destroy(jit);
        free(plan);
        return hit;
    }
    plan->next = g_cache_head;
    g_cache_head = plan;
    pthread_mutex_unlock(&g_cache_lock);
    return plan;
#else
    /* No silent conv without the JIT kernel. */
    return NULL;
#endif
}

fft_plan *fft_plan_create_linear_conv(int N, const float complex *h, int nh) {
    if (N < 1 || h == NULL || nh <= 0 || nh > N) {
        return NULL;
    }
#if !defined(FFT_JIT_DISABLE)
    /* Linear conv is family-only.  Non-family / NULL h / nh<=0 / nh>N /
     * emit / OOM → NULL.  MUST NOT fall back to adapt, a silent forward
     * plan, or cyclic convolution. */
    if (!fft_jit_family_member(N)) {
        return NULL;
    }

    pthread_mutex_lock(&g_cache_lock);
    fft_plan *hit = cache_lookup_locked(N, 0, 0, 0, 1, nh);
    pthread_mutex_unlock(&g_cache_lock);
    if (hit != NULL) {
        return hit;
    }

    fft_jit_kernel *jit = fft_jit_kernel_create(N);
    if (jit == NULL) {
        return NULL;
    }
    fft_jit_fn_t jit_fn = fft_jit_kernel_fn(jit);
    if (jit_fn == NULL) {
        fft_jit_kernel_destroy(jit);
        return NULL;
    }

    /* Plan-time H[k] = FFT(h_padded): zero-pad h (first nh entries) to
     * length N, then reuse the already-promoted forward kernel on a scratch
     * copy.  Do NOT clobber the caller's h; do NOT call exp; do NOT emit a
     * new twiddle / conv kernel.  H is a separate table from the twiddle
     * LUT and the gauss G table (no fourth twiddle subsystem). */
    float complex *scratch =
        (float complex *)malloc((size_t)N * sizeof(*scratch));
    if (scratch == NULL) {
        fft_jit_kernel_destroy(jit);
        return NULL;
    }
    float complex *H = (float complex *)malloc((size_t)N * sizeof(*H));
    if (H == NULL) {
        free(scratch);
        fft_jit_kernel_destroy(jit);
        return NULL;
    }
    for (int i = 0; i < N; i++) {
        scratch[i] = (i < nh) ? h[i] : (0.0f + 0.0f * I);
    }
    jit_fn(scratch); /* forward: scratch = FFT(h_padded) */
    for (int i = 0; i < N; i++) {
        H[i] = scratch[i];
    }
    free(scratch);

    /* topic jit-ps-sched-fuse: same fused kernel as cyclic conv; the OLS
     * valid-sample shift stays in the wrapper (contract semantics).
     * topic jit-joint-search-batched (L-E): inherits the joint placement
     * search (standalone = searched plain kernel + this wrapper's
     * three-segment path). */
    int fused = 0;
#if !defined(FFT_JIT_DISABLE)
    if (fused_conv_enabled() && N >= 32) {
        fft_jit_kernel *fj = fft_jit_kernel_create_conv_searched(
            N, (const float *)H, -1);
        if (fj != NULL && fft_jit_kernel_fn(fj) != NULL) {
            const fft_jit_search_stats *st = fft_jit_conv_search_last();
            fft_jit_kernel_destroy(jit);
            jit = fj;
            jit_fn = fft_jit_kernel_fn(fj);
            fused = (st != NULL && st->place == FFT_PLACE_STANDALONE) ? 0
                                                                      : 1;
        } else if (fj != NULL) {
            fft_jit_kernel_destroy(fj);
        }
    }
#endif

    struct fft_plan *plan = (struct fft_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        free(H);
        fft_jit_kernel_destroy(jit);
        return NULL;
    }
    plan->N = N;
    plan->windowed = 0;
    plan->inverse = 0;
    plan->conv = 0;
    plan->linear = 1;
    plan->fused_conv = fused;
    plan->nh = nh;
    plan->fp64 = 0;
    plan->H = H;
    plan->adapt = NULL;
    plan->jit = jit;
    plan->jit_fn = jit_fn;
    plan->refcount = 1;

    pthread_mutex_lock(&g_cache_lock);
    hit = cache_lookup_locked(N, 0, 0, 0, 1, nh);
    if (hit != NULL) {
        pthread_mutex_unlock(&g_cache_lock);
        free(H);
        fft_jit_kernel_destroy(jit);
        free(plan);
        return hit;
    }
    plan->next = g_cache_head;
    g_cache_head = plan;
    pthread_mutex_unlock(&g_cache_lock);
    return plan;
#else
    /* No silent linear conv without the JIT kernel. */
    return NULL;
#endif
}

/* OLS-leg block length: next power of two >= 2*nh - 1, so each block keeps
 * valid = N_block - nh + 1 ~ nh samples (minimal padding waste). */
static int conv_dispatch_block_len(int nh) {
    long n = 2L * (long)nh - 1;
    long p = 1;
    while (p < n) {
        p <<= 1;
    }
    return (int)p;
}

fft_plan *fft_plan_create_conv_dispatch(int nx, const float complex *h, int nh) {
    if (nx <= 0 || nh <= 0 || nh > nx || h == NULL) {
        return NULL;
    }
#if !defined(FFT_JIT_DISABLE)
    int direct = (nh <= FFT_CONV_DISPATCH_NH_DIRECT_MAX);

    struct fft_plan *plan = (struct fft_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        return NULL;
    }
    plan->dispatch = 1;
    plan->dispatch_direct = direct ? 1 : 0;
    plan->nx = nx;
    plan->nh = nh;
    plan->fp64 = 0;
    plan->refcount = 1;

    if (direct) {
        /* Direct scalar leg: no transform state, no kernel, no sub-plan.
         * Only a time-domain copy of h (execute reads h, not H). */
        plan->h_copy =
            (float complex *)malloc((size_t)nh * sizeof(*plan->h_copy));
        if (plan->h_copy == NULL) {
            free(plan);
            return NULL;
        }
        for (int i = 0; i < nh; i++) {
            plan->h_copy[i] = h[i];
        }
        return plan;
    }

    /* OLS block leg: reuse the promoted single-block linear_conv plan.
     * Block length pinned by nh.  Non-family / emit / OOM → NULL. */
    int N_block = conv_dispatch_block_len(nh);
    plan->sub = fft_plan_create_linear_conv(N_block, h, nh);
    if (plan->sub == NULL) {
        free(plan);
        return NULL;
    }
    long ylen = (long)nx + (long)nh - 1;
    plan->work = (float complex *)malloc((size_t)ylen * sizeof(*plan->work));
    plan->blk = (float complex *)malloc((size_t)N_block * sizeof(*plan->blk));
    if (plan->work == NULL || plan->blk == NULL) {
        fft_plan_destroy(plan->sub);
        free(plan->work);
        free(plan->blk);
        free(plan);
        return NULL;
    }
    return plan;
#else
    /* No silent dispatch without the JIT kernel (the OLS leg needs it). */
    return NULL;
#endif
}

int fft_plan_conv_dispatch_leg(const fft_plan *plan) {
    if (plan == NULL || !plan->dispatch) {
        return -1;
    }
    return plan->dispatch_direct ? 0 : 1;
}

static void fft_inverse_wrap(fft_plan *plan, float complex *io) {
    const int N = plan->N;
    const float invN = 1.0f / (float)N;
    for (int i = 0; i < N; i++) {
        io[i] = conjf(io[i]);
    }
#if !defined(FFT_JIT_DISABLE)
    plan->jit_fn(io);
#endif
    for (int i = 0; i < N; i++) {
        io[i] = conjf(io[i]) * invN;
    }
}

static void fft_conv_wrap(fft_plan *plan, float complex *io) {
    if (plan->fused_conv) {
        /* fused single-kernel execute (topic jit-ps-sched-fuse): one
         * call = forward * H + conjugate-trick inverse, store-boundary
         * fused; no wrapper sweeps. */
        plan->jit_fn(io);
        return;
    }
    const int N = plan->N;
    const float invN = 1.0f / (float)N;
#if !defined(FFT_JIT_DISABLE)
    /* forward: io = X[k] = FFT(x) */
    plan->jit_fn(io);
    /* pointwise in-register X[k]·H[k] */
    for (int i = 0; i < N; i++) {
        io[i] = io[i] * plan->H[i];
    }
    /* IFFT via conjugate trick: conj → forward → conj / N */
    for (int i = 0; i < N; i++) {
        io[i] = conjf(io[i]);
    }
    plan->jit_fn(io);
    for (int i = 0; i < N; i++) {
        io[i] = conjf(io[i]) * invN;
    }
#endif
}

static void fft_linear_wrap(fft_plan *plan, float complex *io) {
    const int N = plan->N;
    const int nh = plan->nh;
    const int valid = N - nh + 1; /* samples per block (OLS kept tail) */
    const float invN = 1.0f / (float)N;
    if (plan->fused_conv) {
        /* fused single-kernel cyclic conv + the contract OLS shift */
        plan->jit_fn(io);
        for (int i = 0; i < valid; i++) {
            io[i] = io[i + (nh - 1)];
        }
        return;
    }
#if !defined(FFT_JIT_DISABLE)
    /* forward: io = X[k] = FFT(x block) */
    plan->jit_fn(io);
    /* pointwise in-register X[k]·H[k] */
    for (int i = 0; i < N; i++) {
        io[i] = io[i] * plan->H[i];
    }
    /* IFFT via conjugate trick: conj → forward → conj / N */
    for (int i = 0; i < N; i++) {
        io[i] = conjf(io[i]);
    }
    plan->jit_fn(io);
    for (int i = 0; i < N; i++) {
        io[i] = conjf(io[i]) * invN;
    }
    /* OLS: drop the first nh-1 wrap-around-polluted samples; shift the
     * N-nh+1 valid linear-conv samples (io[nh-1 .. N-1]) into the front
     * (io[0 .. N-nh]).  Forward leftward shift is safe: each write target
     * io[i] is read (as io[i+nh-1] source) before it is written.  The
     * trailing nh-1 positions are a discard zone (content unpromised). */
    for (int i = 0; i < valid; i++) {
        io[i] = io[i + (nh - 1)];
    }
#endif
}

/* Direct scalar linear-convolution leg (draft API-FFT-007):
 * y[m] = Σ_j h[j]·x[m−j], O(nx·nh).  In-place high→low: y[m] reads x[j]
 * for j ≤ min(m, nx−1), all of which are still intact because we write in
 * decreasing m order (x[j] for j ≤ m is never needed by any y[m'] with
 * m' < m).  No FFT, no padding, no kernel, no transform state. */
static void fft_dispatch_direct_wrap(fft_plan *plan, float complex *io) {
    const int nx = plan->nx;
    const int nh = plan->nh;
    const float complex *h = plan->h_copy;
    const long ylen = (long)nx + (long)nh - 1;
    for (long m = ylen - 1; m >= 0; m--) {
        float complex acc = 0.0f + 0.0f * I;
        long j0 = (m >= nh) ? (m - nh + 1) : 0;
        long j1 = (m < nx) ? m : (nx - 1);
        for (long j = j0; j <= j1; j++) {
            acc += io[j] * h[m - j];
        }
        io[m] = acc;
    }
}

/* OLS block linear-convolution leg (draft API-FFT-007): snapshot x into a
 * padded scratch ([0^{nh-1}, x]), then process length-N_block blocks with
 * hop = valid = N_block - nh + 1 via the promoted single-block linear_conv
 * sub-plan, concatenating valid samples back into io.  The block loop runs
 * INSIDE fft_execute (one call welds the whole y = x * h). */
static void fft_dispatch_ols_wrap(fft_plan *plan, float complex *io) {
    fft_plan *sub = plan->sub;
    const int N_block = sub->N;
    const int nh = plan->nh;
    const int valid = N_block - nh + 1;
    const long ylen = (long)plan->nx + (long)nh - 1;
    float complex *work = plan->work;
    float complex *blk = plan->blk;

    for (long i = 0; i < ylen; i++) {
        work[i] = 0.0f + 0.0f * I;
    }
    for (long i = 0; i < plan->nx; i++) {
        work[(long)(nh - 1) + i] = io[i];
    }

    long nblocks = (ylen + valid - 1) / valid;
    for (long b = 0; b < nblocks; b++) {
        long base = b * (long)valid;
        for (int i = 0; i < N_block; i++) {
            long idx = base + i;
            blk[i] = (idx < ylen) ? work[idx] : (0.0f + 0.0f * I);
        }
        fft_execute(sub, blk);
        for (int i = 0; i < valid; i++) {
            long oidx = base + i;
            if (oidx < ylen) {
                io[oidx] = blk[i];
            }
        }
    }
}

void fft_execute(fft_plan *plan, float complex *io) {
    if (plan == NULL || io == NULL) {
        return;
    }
    if (plan->batch) {
        /* Batched windowed plans MUST use fft_execute_windowed_batch
         * (topic jit-joint-search-batched, draft {#API-FFT-010}); the
         * default execute never processes them. */
        return;
    }
    if (plan->fp64) {
        /* fp64 plans MUST use fft_execute_fp64 (double complex). */
        return;
    }
#if !defined(FFT_JIT_DISABLE)
    if (plan->dispatch) {
        if (plan->dispatch_direct) {
            /* Missing time-domain h copy: fail-fast.  MUST NOT fall through
             * to forward/adapt/conv/linear. */
            if (plan->h_copy == NULL) {
                return;
            }
            fft_dispatch_direct_wrap(plan, io);
        } else {
            /* Missing sub-plan / scratch: fail-fast.  MUST NOT fall
             * through to forward/adapt/conv/linear. */
            if (plan->sub == NULL || plan->work == NULL || plan->blk == NULL) {
                return;
            }
            fft_dispatch_ols_wrap(plan, io);
        }
        return;
    }
    if (plan->conv) {
        /* Missing kernel / missing H: fail-fast.  MUST NOT fall through to
         * forward/adapt.  MUST NOT exp. */
        if (plan->jit_fn == NULL || plan->jit == NULL || plan->H == NULL) {
            return;
        }
        fft_conv_wrap(plan, io);
        return;
    }
    if (plan->linear) {
        /* Missing kernel / missing H: fail-fast.  MUST NOT fall through to
         * forward/adapt/conv.  MUST NOT exp. */
        if (plan->jit_fn == NULL || plan->jit == NULL || plan->H == NULL) {
            return;
        }
        fft_linear_wrap(plan, io);
        return;
    }
    if (plan->inverse) {
        /* Missing kernel: fail-fast.  MUST NOT fall through to forward. */
        if (plan->jit_fn == NULL || plan->jit == NULL) {
            return;
        }
        fft_inverse_wrap(plan, io);
        return;
    }
    if (plan->windowed) {
        /* Missing G / missing fused fn: fail-fast.  MUST NOT exp, MUST
         * NOT fall through to the unwindowed adapt path. */
        if (plan->jit_fn == NULL || plan->jit == NULL ||
            fft_jit_kernel_gauss_lut(plan->jit) == NULL) {
            return;
        }
        plan->jit_fn(io);
        return;
    }
    if (plan->jit_fn != NULL) {
        plan->jit_fn(io);
        return;
    }
#endif
    fft_adapt_execute(plan->adapt, io);
}

void fft_execute_fp64(fft_plan *plan, double complex *io) {
    if (plan == NULL || io == NULL) {
        return;
    }
    if (!plan->fp64) {
        /* Non-fp64 plan: no-op (defensive; never run fp32 kernel on a
         * double buffer). */
        return;
    }
    if (plan->conv64) {
        /* topic jit-fp64-conv-fuse: a conv plan MUST use
         * fft_execute_conv_fp64 (the weld runs at Np, not N — running it
         * through the forward execute symbol would be a buffer overrun).
         * No-op here (defensive). */
        return;
    }
    if (plan->inv64) {
        /* topic jit-fp64-inverse: an inverse plan MUST use
         * fft_execute_fp64_inverse — running the fused inverse kernel
         * through the forward execute symbol would be a contract
         * violation.  No-op here (defensive). */
        return;
    }
    if (plan->lin64) {
        /* topic jit-fp64-linear-conv: an OLS linear-conv plan MUST use
         * fft_execute_linear_conv_fp64 (draft {#API-FFT-012} — C has no
         * overloading; the OLS weld's valid-prefix store semantics are
         * not the forward contract).  No-op here (defensive). */
        return;
    }
#if !defined(FFT_JIT_DISABLE)
    /* topic jit-anyN-fwd-inv: coverage-set wrapper floor (bare prime /
     * tiny N) — host O(N^2) DFT, oracle-tolerance exact (conv tiny
     * precedent). */
    if (plan->anyn_floor) {
        anyn_floor_dft(io, plan->N);
        return;
    }
    if (plan->jit_fn == NULL || plan->jit == NULL) {
        return;
    }
    plan->jit_fn(io);
#endif
}

void fft_plan_destroy(fft_plan *plan) {
    if (plan == NULL) {
        return;
    }

    pthread_mutex_lock(&g_cache_lock);
    plan->refcount--;
    int drop = (plan->refcount <= 0);
    pthread_mutex_unlock(&g_cache_lock);

    if (!drop) {
        return;
    }

    pthread_mutex_lock(&g_cache_lock);
    struct fft_plan **head =
        plan->batch    ? &g_cache_head_batch
        : plan->conv64 ? &g_cache_head_conv64
        : plan->inv64  ? &g_cache_head_inv64
        : plan->lin64  ? &g_cache_head_lin64
        : (plan->fp64 ? &g_cache_head_fp64 : &g_cache_head);
    struct fft_plan **link = head;
    while (*link != NULL && *link != plan) {
        link = &(*link)->next;
    }
    if (*link == plan) {
        *link = plan->next;
    }
    pthread_mutex_unlock(&g_cache_lock);

    free(plan->H);
    free(plan->G2);
    if (plan->dispatch) {
        fft_plan_destroy(plan->sub);
        free(plan->h_copy);
        free(plan->work);
        free(plan->blk);
    }
/* topic jit-fp64-conv-fuse: conv plan state (padded buffers, tail-
     * correlation weld, tiny h copies).  calloc-zeroed, so freeing NULL
     * is a no-op for every non-conv plan. */
    free(plan->buf64);
    free(plan->buf2_64);
    free(plan->h64);
    /* topic jit-fp64-linear-conv: OLS plan state (retained input tail;
     * the tiny floor's h copy rides h64 above).  calloc-zeroed. */
    free(plan->ols_tail);
#if !defined(FFT_JIT_DISABLE)
    fft_jit_kernel_destroy(plan->jit2);
    fft_jit_kernel_destroy(plan->jit);
#endif
    fft_adapt_plan_destroy(plan->adapt);
    free(plan);
}
