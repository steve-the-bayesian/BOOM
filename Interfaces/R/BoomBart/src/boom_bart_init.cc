#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#define CALLDEF(name, n)  {#name, (DL_FUNC) &name, n}

extern "C" {

  SEXP boom_bart_wrapper_(
      SEXP r_number_of_trees,
      SEXP r_design_matrix,
      SEXP r_response,
      SEXP r_family,
      SEXP r_tree_prior,
      SEXP r_discrete_distribution_limit,
      SEXP r_continuous_distribution_strategy,
      SEXP r_niter,
      SEXP r_ping,
      SEXP r_seed);

  static R_CallMethodDef boom_bart_arg_description[] = {
    CALLDEF(boom_bart_wrapper_, 10),
    {NULL, NULL, 0}
  };

  void R_init_BoomMix(DllInfo *info) {
    R_registerRoutines(info,
                       NULL,
                       boom_bart_arg_description,
                       NULL,
                       NULL);
    R_useDynamicSymbols(info, FALSE);
  }

}  // extern "C"
