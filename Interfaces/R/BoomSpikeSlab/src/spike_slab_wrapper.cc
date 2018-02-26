// Copyright 2010 Google Inc. All Rights Reserved.
// Author: stevescott@google.com (Steve Scott)

#include <exception>

#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/PosteriorSamplers/SpikeSlabDaRegressionSampler.hpp"
#include "Models/Glm/PosteriorSamplers/TRegressionSpikeSlabSampler.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/TRegression.hpp"

#include "Models/ChisqModel.hpp"
#include "Models/IndependentMvnModelGivenScalarSigma.hpp"
#include "Models/MvnGivenScalarSigma.hpp"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/handle_exception.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/prior_specification.hpp"
#include "r_interface/seed_rng_from_R.hpp"
#include "utils.h"

#include "cpputil/Ptr.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::RInterface;

  class SseCallback
      : public BOOM::ScalarIoCallback {
   public:
    SseCallback(RegressionModel *model)
        : model_(model) {}
    double get_value() const override {
      return model_->suf()->relative_sse(model_->coef());
    }
   private:
    RegressionModel *model_;
  };

  template <class MODEL>
  void configure_io_manager(Ptr<MODEL> model,
                            BOOM::RListIoManager *io_manager) {
    io_manager->add_list_element(
        new GlmCoefsListElement(model->coef_prm(), "beta"));
    io_manager->add_list_element(
        new StandardDeviationListElement(model->Sigsq_prm(), "sigma"));
  }

  template <class MODEL>
  void initialize_model(Ptr<MODEL> model,
                        const Matrix &design_matrix,
                        const Vector &response_vector) {
    size_t n = design_matrix.nrow();
    for (size_t i = 0; i < n; ++i) {
      model->add_data(Ptr<RegressionData>(
          new RegressionData(response_vector[i], design_matrix.row(i))));
    }
    model->coef().drop_all();
    model->coef().add(0);  // start with the intercept
  }

  template <class SAMPLER, class PRIOR>
  void initialize_sampler(Ptr<SAMPLER> sampler, const PRIOR &prior) {
    sampler->set_sigma_upper_limit(prior.sigma_upper_limit());
    if (prior.max_flips() > 0) {
      sampler->limit_model_selection(prior.max_flips());
    }
  }

  Ptr<GlmModel> SpecifyRegressionModel(
      SEXP r_design_matrix,
      SEXP r_response_vector,
      SEXP r_spike_slab_prior,
      SEXP r_bma_method,
      SEXP r_oda_options,
      BOOM::RListIoManager *io_manager) {
    Matrix design_matrix(ToBoomMatrix(r_design_matrix));
    Ptr<RegressionModel> model(new RegressionModel(design_matrix.ncol()));
    initialize_model(model, design_matrix, ToBoomVector(r_response_vector));

    std::string bma_method = ToString(r_bma_method);
    if (bma_method == "SSVS") {
      BOOM::RInterface::RegressionConjugateSpikeSlabPrior prior(
          r_spike_slab_prior, model->Sigsq_prm());
      NEW(BregVsSampler, ssvs_sampler)(
          model.get(),
          prior.slab(),
          prior.siginv_prior(),
          prior.spike());
      initialize_sampler(ssvs_sampler, prior);
      model->set_method(ssvs_sampler);
      BOOM::spikeslab::InitializeCoefficients(
          model->Beta(),
          prior.spike()->prior_inclusion_probabilities(),
          model,
          ssvs_sampler);
    } else if (bma_method == "ODA") {
      BOOM::RInterface::IndependentRegressionSpikeSlabPrior prior(
          r_spike_slab_prior, model->Sigsq_prm());
      double eigenvalue_fudge_factor = .01;
      double fallback_probability = 0.0;
      if (!Rf_isNull(r_oda_options)) {
        eigenvalue_fudge_factor = Rf_asReal(getListElement(
            r_oda_options,
            "eigenvalue.fudge.factor"));
        fallback_probability = Rf_asReal(getListElement(
            r_oda_options,
            "fallback.probability"));
      }
      NEW(SpikeSlabDaRegressionSampler, oda_sampler)(
              model.get(),
              prior.slab(),
              prior.siginv_prior(),
              prior.prior_inclusion_probabilities(),
              eigenvalue_fudge_factor,
              fallback_probability);
      initialize_sampler(oda_sampler, prior);
      model->set_method(oda_sampler);
      BOOM::spikeslab::InitializeCoefficients(
          model->Beta(),
          prior.spike()->prior_inclusion_probabilities(),
          model,
          oda_sampler);
    }
    configure_io_manager(model, io_manager);
    io_manager->add_list_element(
        new NativeUnivariateListElement(
            new SseCallback(model.get()), "sse", nullptr));
    return(model);
  }

  Ptr<TRegressionModel> SpecifyStudentRegressionModel(
      SEXP r_design_matrix,
      SEXP r_response_vector,
      SEXP r_spike_slab_prior,
      BOOM::RListIoManager *io_manager) {
    Matrix design_matrix(ToBoomMatrix(r_design_matrix));
    NEW(TRegressionModel, model)(design_matrix.ncol());
    initialize_model(model, design_matrix, ToBoomVector(r_response_vector));
    RInterface::StudentRegressionConjugateSpikeSlabPrior prior(
        r_spike_slab_prior, model->Sigsq_prm());
    NEW(TRegressionSpikeSlabSampler, sampler)(
        model.get(),
        prior.slab(),
        prior.spike(),
        prior.siginv_prior(),
        prior.degrees_of_freedom_prior());
    initialize_sampler(sampler, prior);
    model->set_method(sampler);
    configure_io_manager(model, io_manager);
    io_manager->add_list_element(
        new UnivariateListElement(model->Nu_prm(), "tail.thickness"));
    return model;
  }

}  // namespace

extern "C" {
  using namespace BOOM;  // NOLINT

  SEXP analysis_common_r_do_spike_slab(SEXP r_design_matrix,
                                       SEXP r_response_vector,
                                       SEXP r_spike_slab_prior,
                                       SEXP r_error_distribution,
                                       SEXP r_niter,
                                       SEXP r_ping,
                                       SEXP r_bma_method,
                                       SEXP r_oda_options,
                                       SEXP r_seed) {
    RErrorReporter error_reporter;
    RMemoryProtector protector;
    try {
      seed_rng_from_R(r_seed);
      RListIoManager io_manager;
      Ptr<Model> model;
      std::string error_distribution = ToString(r_error_distribution);
      if (error_distribution == "gaussian") {
        model = SpecifyRegressionModel(
            r_design_matrix,
            r_response_vector,
            r_spike_slab_prior,
            r_bma_method,
            r_oda_options,
            &io_manager);
      } else if (error_distribution == "student") {
        model = SpecifyStudentRegressionModel(
            r_design_matrix,
            r_response_vector,
            r_spike_slab_prior,
            &io_manager);
      }

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
    } catch(std::exception &e) {
      handle_exception(e);
    } catch (...) {
      handle_unknown_exception();
    }
    return R_NilValue;
  }
}
