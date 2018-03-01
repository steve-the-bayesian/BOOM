#include <Rinternals.h>        // for SEXP
#include <R_ext/Rdynload.h>    // for R_registerRoutines and R_CallMethodDef

#define CALLDEF(name, n)  {#name, (DL_FUNC) &name, n}

extern "C" {

SEXP analysis_common_r_do_spike_slab(
    SEXP r_design_matrix,
    SEXP r_response_vector,
    SEXP r_spike_slab_prior,
    SEXP r_error_distribution,
    SEXP r_niter,
    SEXP r_ping,
    SEXP r_bma_method,
    SEXP r_oda_options,
    SEXP r_seed);

SEXP logit_spike_slab_wrapper(
    SEXP r_x,              // design matrix
    SEXP r_y,              // vector of success counts
    SEXP r_ny,             // vector of trial counts
    SEXP r_prior,          // SpikeSlabPrior
    SEXP r_niter,          // number of mcmc iterations
    SEXP r_ping,           // frequency of desired status updates
    SEXP r_nthreads,       // number of imputation threads
    SEXP r_beta0,          // initial value in the MCMC simulation
    SEXP r_clt_threshold,  // see comments in ../R/logit.spike.R
    SEXP r_mh_chunk_size,  // see comments in ../R/logit.spike.R
    SEXP r_sampling_weights,  // see comments in ../R/logit.spike.R
    SEXP r_seed);

SEXP analysis_common_r_multinomial_logit_spike_slab(
    SEXP r_response_factor,
    SEXP r_subject_predictor_matrix,
    SEXP r_choice_predictor_matrix,
    SEXP r_choice_predictor_subject_id,
    SEXP r_choice_predictor_choice_id,
    SEXP r_multinomial_logit_spike_slab_prior,
    SEXP r_niter,
    SEXP r_ping,
    SEXP r_proposal_df,
    SEXP r_rwm_scale_factor,
    SEXP r_nthreads,
    SEXP r_mh_chunk_size,
    SEXP r_proposal_weights,
    SEXP r_seed);

SEXP analysis_common_r_poisson_regression_spike_slab(
    SEXP r_design_matrix,
    SEXP r_integer_response_vector,
    SEXP r_exposure_vector,
    SEXP r_spike_slab_prior,
    SEXP r_niter,
    SEXP r_ping,
    SEXP r_nthreads,
    SEXP r_initial_beta,
    SEXP r_seed);

SEXP probit_spike_slab_wrapper(
    SEXP r_x,              // design matrix
    SEXP r_y,              // vector of success counts
    SEXP r_ny,             // vector of trial counts
    SEXP r_prior,          // SpikeSlabPrior
    SEXP r_niter,          // number of MCMC iterations
    SEXP r_ping,           // frequency of desired status updates
    SEXP r_beta0,          // initial value in the MCMC simulation
    SEXP r_clt_threshold,  // see comments in ../R/probit.spike.R
    SEXP r_proposal_df,
    SEXP r_sampling_weights,
    SEXP r_seed);

SEXP analysis_common_r_quantile_regression_spike_slab(
    SEXP r_design_matrix,
    SEXP r_response_vector,
    SEXP r_quantile,
    SEXP r_spike_slab_prior,
    SEXP r_niter,
    SEXP r_ping,
    SEXP r_nthreads,
    SEXP r_initial_beta,
    SEXP r_seed);

SEXP boom_shrinkage_regression_wrapper(
    SEXP r_regression_suf,
    SEXP r_coefficient_groups,
    SEXP r_residual_precision_prior,
    SEXP r_niter,
    SEXP r_ping,
    SEXP r_seed);

SEXP boom_nested_regression_wrapper(
    SEXP r_regression_suf_list,
    SEXP r_coefficient_prior,
    SEXP r_coefficient_mean_hyperprior,
    SEXP r_coefficient_variance_hyperprior,
    SEXP r_residual_precision_prior,
    SEXP r_niter,
    SEXP r_ping,
    SEXP r_sampling_method,
    SEXP r_seed);

static R_CallMethodDef spike_slab_arg_description[] = {
  CALLDEF(analysis_common_r_do_spike_slab, 9),
  CALLDEF(logit_spike_slab_wrapper, 12),
  CALLDEF(analysis_common_r_multinomial_logit_spike_slab, 14),
  CALLDEF(analysis_common_r_poisson_regression_spike_slab, 9),
  CALLDEF(probit_spike_slab_wrapper, 11),
  CALLDEF(analysis_common_r_quantile_regression_spike_slab, 9),
  CALLDEF(boom_shrinkage_regression_wrapper, 6),
  CALLDEF(boom_nested_regression_wrapper, 9),
  {NULL, NULL, 0}
};

void R_init_BoomSpikeSlab(DllInfo *info) {
  R_registerRoutines(info,
                     NULL,
                     spike_slab_arg_description,
                     NULL,
                     NULL);
  R_useDynamicSymbols(info, FALSE);
}

}  // extern "C"
