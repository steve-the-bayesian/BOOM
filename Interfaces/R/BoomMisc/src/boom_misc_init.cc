#include <Rinternals.h>        // for SEXP
#include <R_ext/Rdynload.h>    // for R_registerRoutines and R_CallMethodDef

#define CALLDEF(name, n)  {#name, (DL_FUNC) &name, n}

extern "C" {

  SEXP boom_misc_slice_sampler_wrapper(
      SEXP r_vector_function,
      SEXP r_initial_value,
      SEXP r_niter,
      SEXP r_ping,
      SEXP r_seed);

  static R_CallMethodDef boom_misc_arg_description[] = {
    CALLDEF(boom_misc_slice_sampler_wrapper, 5),
    {NULL, NULL, 0}
  };

  void R_init_BoomMisc(DllInfo *info) {
    R_registerRoutines(info,
                       NULL,
                       boom_misc_arg_description,
                       NULL,
                       NULL);
    R_useDynamicSymbols(info, FALSE);
  }
  
}  // extern "C"
