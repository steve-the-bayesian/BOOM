#include <exception>

#include "Models/Glm/PoissonRegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/PoissonRegressionSpikeSlabSampler.hpp"

#include "Models/ChisqModel.hpp"
#include "Models/IndependentMvnModelGivenScalarSigma.hpp"

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
  Ptr<PoissonRegressionModel> SpecifyPoissonRegression(
      SEXP r_design_matrix,
      SEXP r_integer_response_vector,
      SEXP r_exposure_vector,
      SEXP r_spike_slab_prior,
      SEXP r_nthreads,
      SEXP r_initial_beta,
      RListIoManager *io_manager) {
    Matrix design_matrix(ToBoomMatrix(r_design_matrix));
    std::vector<int> response(ToIntVector(r_integer_response_vector));
    Vector exposure(ToBoomVector(r_exposure_vector));
    NEW(PoissonRegressionModel, model)(design_matrix.ncol());
    int n = response.size();
    for (int i = 0; i < n; ++i) {
      NEW(PoissonRegressionData, data_point)(
          response[i],
          design_matrix.row(i),
          exposure[i]);
      model->add_data(data_point);
    }
    SpikeSlabGlmPrior prior_spec(r_spike_slab_prior);

    int nthreads = std::max<int>(1, Rf_asInteger(r_nthreads));
    NEW(PoissonRegressionSpikeSlabSampler, sampler)(
        model.get(),
        prior_spec.slab(),
        prior_spec.spike(),
        nthreads);
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

  SEXP analysis_common_r_poisson_regression_spike_slab(
      SEXP r_design_matrix,
      SEXP r_integer_response_vector,
      SEXP r_exposure_vector,
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
      Ptr<PoissonRegressionModel> model = SpecifyPoissonRegression(
          r_design_matrix,
          r_integer_response_vector,
          r_exposure_vector,
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
