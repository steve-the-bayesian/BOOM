#include <exception>

#include "Models/Glm/QuantileRegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/QuantileRegressionPosteriorSampler.hpp"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/handle_exception.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/prior_specification.hpp"
#include "r_interface/seed_rng_from_R.hpp"

#include "utils.h"

#include "cpputil/Ptr.hpp"

namespace {
  using namespace BOOM;  // NOLINT
  using namespace BOOM::RInterface;  // NOLINT
  Ptr<QuantileRegressionModel> SpecifyQuantileRegression(
      SEXP r_design_matrix,
      SEXP r_response_vector,
      SEXP r_quantile,
      SEXP r_spike_slab_prior,
      SEXP r_nthreads,
      SEXP r_initial_beta,
      RListIoManager *io_manager) {
    Matrix design_matrix(ToBoomMatrix(r_design_matrix));
    Vector response(ToBoomVector(r_response_vector));
    double quantile = Rf_asReal(r_quantile);
    NEW(QuantileRegressionModel, model)(design_matrix.ncol(), quantile);
    int n = response.size();
    for (int i = 0; i < n; ++i) {
      NEW(RegressionData, data_point)(response[i], design_matrix.row(i));
      model->add_data(data_point);
    }
    SpikeSlabGlmPrior prior_spec(r_spike_slab_prior);

    NEW(QuantileRegressionSpikeSlabSampler, sampler)(
        model.get(),
        prior_spec.slab(),
        prior_spec.spike());
    int nthreads = std::max<int>(1, Rf_asInteger(r_nthreads));
    sampler->set_number_of_workers(nthreads);

    if (prior_spec.max_flips() > 0) {
      sampler->limit_model_selection(prior_spec.max_flips());
    }
    model->set_method(sampler);
    BOOM::spikeslab::InitializeCoefficients(
        ToBoomVector(r_initial_beta),
        prior_spec.spike()->prior_inclusion_probabilities(),
        model,
        sampler);
    io_manager->add_list_element(new GlmCoefsListElement(
        model->coef_prm(),
        "beta"));
    return model;
  }
}  // namespace

extern "C" {
  using namespace BOOM;  // NOLINT

  SEXP analysis_common_r_quantile_regression_spike_slab(
      SEXP r_design_matrix,
      SEXP r_response_vector,
      SEXP r_quantile,
      SEXP r_spike_slab_prior,
      SEXP r_niter,
      SEXP r_ping,
      SEXP r_nthreads,
      SEXP r_initial_beta,
      SEXP r_seed) {
    RErrorReporter error_reporter;
    RMemoryProtector protector;
    try {
      seed_rng_from_R(r_seed);
      RListIoManager io_manager;
      Ptr<QuantileRegressionModel> model = SpecifyQuantileRegression(
          r_design_matrix,
          r_response_vector,
          r_quantile,
          r_spike_slab_prior,
          r_nthreads,
          r_initial_beta,
          &io_manager);
      int niter = Rf_asInteger(r_niter);
      int ping = Rf_asInteger(r_ping);
      SEXP ans = protector.protect(io_manager.prepare_to_write(niter));
      for (int i = 0; i < niter; ++i) {
        if (RCheckInterrupt()) {
          error_reporter.SetError("Canceled by user.");
          return R_NilValue;
        }
        print_R_timestamp(i, ping);
        model->sample_posterior();
        io_manager.write();
      }
      return ans;
    } catch (std::exception &e) {
      handle_exception(e);
    } catch (...) {
      handle_unknown_exception();
    }
    return R_NilValue;
  }

}  // extern "C"
