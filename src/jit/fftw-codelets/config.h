/* topic jit-codelet-weld: DOUBLE-precision config shim.
 *
 * The vendored third_party/fftw/config.h belongs to the fp32 build
 * (#define FFTW_SINGLE 1) and has NO include guard, so every inclusion
 * re-defines the precision.  This shim (found first on the -I chain)
 * layers on top of it read-only and undefines FFTW_SINGLE after each
 * inclusion, selecting the double-precision compile for the topic codelet
 * subset.  The vendored file is never modified.
 */
#include_next "config.h"
#undef FFTW_SINGLE
