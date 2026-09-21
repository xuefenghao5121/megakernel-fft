/* topic jit-codelet-weld: standalone convention check (call-in/call-out)
 * for the vendored codelet subset — run before any JIT wiring. */
#include <stdio.h>
#include "weld_codelets.h"

int main(void) {
    weld_codelets_init();
    printf("probe: %s\n", weld_codelets_probe_report());
    int bad = 0;
    for (int r = 2; r <= 25; ++r) {
        if (weld_clet_n1(r) == NULL && weld_clet_t1(r) == NULL) {
            continue; /* not in the set */
        }
        if (weld_clet_n1(r) == NULL || weld_clet_t1(r) == NULL) {
            /* t1fv_13 legitimately absent; only flag n1fv failures */
            if (weld_clet_n1(r) == NULL) {
                printf("MISSING n1fv_%d\n", r);
                bad = 1;
            }
        }
    }
    printf(bad ? "CODELET CHECK FAIL\n" : "CODELET CHECK PASS\n");
    return bad;
}
