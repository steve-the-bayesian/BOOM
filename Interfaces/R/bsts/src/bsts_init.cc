// Copyright 2018 Google Inc. All Rights Reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

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
      SEXP r_niter,
      SEXP r_ping,
      SEXP r_timeout_in_seconds,
      SEXP r_seed);

  SEXP analysis_common_r_predict_bsts_model_(
      SEXP r_bsts_object,
      SEXP r_prediction_data,
      SEXP r_burn,
      SEXP r_observed_data,
      SEXP r_seed);

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

  SEXP analysis_common_r_get_date_ranges_(
      SEXP r_holiday,
      SEXP r_timestamps);
  
  static R_CallMethodDef bsts_arg_description[] = {
    CALLDEF(analysis_common_r_fit_bsts_model_, 9),
    CALLDEF(analysis_common_r_predict_bsts_model_, 5),
    CALLDEF(analysis_common_r_bsts_one_step_prediction_errors_, 2),
    CALLDEF(analysis_common_r_bsts_aggregate_time_series_, 3),
    CALLDEF(analysis_common_r_bsts_fit_mixed_frequency_model_, 11),
    CALLDEF(analysis_common_r_get_date_ranges_, 2),
    {NULL, NULL, 0}  // NOLINT
  };

  void R_init_bsts(DllInfo *info) {
    R_registerRoutines(info, NULL, bsts_arg_description, NULL, NULL);  // NOLINT
    R_useDynamicSymbols(info, FALSE);
  }

}  // extern "C"
