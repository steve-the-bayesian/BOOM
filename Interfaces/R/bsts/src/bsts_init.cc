#include "Rinternals.h"        // for SEXP

// for R_registerRoutines and R_CallMethodDef
#include "R_ext/Rdynload.h"

#define CALLDEF(name, n)  {#name, (DL_FUNC) &name, n}

extern "C" {

SEXP analysis_common_r_fit_bsts_model_(
    SEXP r_data_list,
    SEXP r_state_specification,
    SEXP r_prior,
    SEXP r_options,
    SEXP r_family,
    SEXP r_save_state_contribution_flag,
    SEXP r_save_prediction_errors_flag,
    SEXP r_niter,
    SEXP r_ping,
    SEXP r_timeout_in_seconds,
    SEXP r_seed);

SEXP analysis_common_r_predict_bsts_model_(
    SEXP r_bsts_object,
    SEXP r_prediction_data,
    SEXP r_burn,
    SEXP r_observed_data);

SEXP analysis_common_r_bsts_one_step_prediction_errors_(
    SEXP r_bsts_object,
    SEXP r_cutpoints);

SEXP analysis_common_r_bsts_aggregate_time_series_(
    SEXP r_fine_series,
    SEXP r_contains_end,
    SEXP r_membership_fraction);

SEXP analysis_common_r_bsts_fit_mixed_frequency_model_(
    SEXP r_target_series,
    SEXP r_predictors,
    SEXP r_which_coarse_interval,
    SEXP r_membership_fraction,
    SEXP r_contains_end,
    SEXP r_state_specification,
    SEXP r_regression_prior,
    SEXP r_niter,
    SEXP r_ping,
    SEXP r_seed,
    SEXP r_truth);

static R_CallMethodDef bsts_arg_description[] = {
  CALLDEF(analysis_common_r_fit_bsts_model_, 11),
  CALLDEF(analysis_common_r_predict_bsts_model_, 4),
  CALLDEF(analysis_common_r_bsts_one_step_prediction_errors_, 2),
  CALLDEF(analysis_common_r_bsts_aggregate_time_series_, 3),
  CALLDEF(analysis_common_r_bsts_fit_mixed_frequency_model_, 11),
  {NULL, NULL, 0}  // NOLINT
};

void R_init_bsts(DllInfo *info) {
  R_registerRoutines(info, NULL, bsts_arg_description, NULL, NULL);  // NOLINT
  R_useDynamicSymbols(info, FALSE);
}

}  // extern "C"
