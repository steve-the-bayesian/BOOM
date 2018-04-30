#include "Models/WishartModel.hpp"
#include "Models/GammaModel.hpp"
#include "Models/MvnModel.hpp"
#include "Models/PosteriorSamplers/MvnVarSampler.hpp"
#include "Models/PosteriorSamplers/MvnMeanSampler.hpp"
#include "Models/Glm/PosteriorSamplers/RegressionShrinkageSampler.hpp"
#include "Models/Hierarchical/HierarchicalGaussianRegressionModel.hpp"
#include "Models/Hierarchical/PosteriorSamplers/HierarchicalGaussianRegressionSampler.hpp"
#include "Models/Hierarchical/PosteriorSamplers/HierGaussianRegressionAsisSampler.hpp"

#include "cpputil/math_utils.hpp"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/handle_exception.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/prior_specification.hpp"
#include "r_interface/sufstats.hpp"
#include "r_interface/seed_rng_from_R.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::RInterface;

  // Extracts the prior distribution for a nested regression model.  The
  // hyperpriors for the mean and variance (if non-NULL) are set, along with the
  // corresponding PosteriorSamplers.
  class PriorExtractor {
    public:
    // Args:
    //   r_coefficient_prior: An object of class MvnPrior giving the initial
    //     value of the prior parameters.
    //   r_coefficient_mean_hyperprior: An object of class MvnPrior, or NULL.
    //     If non-NULL, the hyperprior is used to set a posterior sampler for
    //     the mean of r_coefficient_prior.
    //   r_coefficient_variance_hyperprior: An object of class
    //     InverseWishartPrior, or NULL.  If non-NULL, the hyperprior is used to
    //     set a posterior sampler for the variance of r_coefficient_prior.
    PriorExtractor(SEXP r_coefficient_prior,
                   SEXP r_coefficient_mean_hyperprior,
                   SEXP r_coefficient_variance_hyperprior,
                   SEXP r_sampling_method)
        : r_coefficient_prior_(r_coefficient_prior),
          r_coefficient_mean_hyperprior_(r_coefficient_mean_hyperprior),
          r_coefficient_variance_hyperprior_(r_coefficient_variance_hyperprior),
          use_ASIS_(ToString(r_sampling_method) == "ASIS") {}

    // Returns the prior distribution, as specified by the constructor
    // arguments.  Posterior samplers are set for parameters where a non-NULL
    // hyperprior was specified.
    Ptr<MvnModel> extract() {
      extract_prior();
      if (!use_ASIS_) {
        mean_hyperprior();
        variance_hyperprior();
      }
      return prior_;
    }

    // Must call extract_prior before calling this.
    Ptr<MvnModel> mean_hyperprior() {
      if (!Rf_isNull(r_coefficient_mean_hyperprior_)) {
        RInterface::MvnPrior spec(r_coefficient_mean_hyperprior_);
        NEW(MvnModel, mean_hyperprior)(spec.mu(), spec.Sigma());
        if (!use_ASIS_) {
          NEW(MvnMeanSampler, mean_sampler)(prior_.get(), mean_hyperprior);
          prior_->set_method(mean_sampler);
        }
        return mean_hyperprior;
      } else {
        return nullptr;
      }
    }

    // Must call extract_prior before calling this.
    Ptr<WishartModel> variance_hyperprior() {
      if (!Rf_isNull(r_coefficient_variance_hyperprior_)) {
        RInterface::InverseWishartPrior spec(
            r_coefficient_variance_hyperprior_);
        NEW(WishartModel, variance_hyperprior)(
            spec.variance_guess_weight(),
            spec.variance_guess());
        if (!use_ASIS_) {
          NEW(MvnVarSampler, variance_sampler)(
              prior_.get(), variance_hyperprior);
          prior_->set_method(variance_sampler);
        }
        return variance_hyperprior;
      } else {
        return nullptr;
      }
    }

    bool ASIS() const {return use_ASIS_;}

   private:
    void extract_prior() {
      if (Rf_isNull(r_coefficient_prior_)) {
        RInterface::MvnPrior hyperprior_spec(r_coefficient_mean_hyperprior_);
        prior_ = new MvnModel(hyperprior_spec.mu().size());
      } else {
        RInterface::MvnPrior spec(r_coefficient_prior_);
        prior_ = new MvnModel(spec.mu(), spec.Sigma());
      }
    }

    SEXP r_coefficient_prior_;
    SEXP r_coefficient_mean_hyperprior_;
    SEXP r_coefficient_variance_hyperprior_;
    bool use_ASIS_;
    Ptr<MvnModel> prior_;
  };

  // Args:
  //   r_regression_suf_list: An R list of R's RegressionSuf objects.  Each list
  //     element contains the sufficient statistics for a particular group in
  //     the hierarchical model.
  //   r_coefficient_prior: An object of class MvnPrior giving the initial
  //     values of the prior distribution describing the distribution of
  //     regression coefficients across groups.
  //   r_coefficient_mean_hyperprior: An object of class MvnPrior, or NULL.  If
  //     NULL then the mean of r_coefficient_prior will not be updated as part
  //     of the MCMC.  Otherwise a PosteriorSampler using this prior will be
  //     used to update the mean of r_coefficient_prior in the MCMC algorithm.
  //   r_coefficient_variance_hyperprior_: An object of class
  //     InverseWishartPrior, or NULL.  If NULL then the variance of
  //     r_coefficient_prior will not be updated as part of the MCMC.  Otherwise
  //     a PosteriorSampler using this prior will be used to update the variance
  //     of r_coefficient_prior in the MCMC algorithm.
  //   r_residual_precision_prior: An object of class SdPrior giving the prior
  //     distribution for the residual precision.
  //   io_manager: The io manager responsible for recording the MCMC output and
  //     returning it to R.
  //
  // Returns:
  //   A HierarchicalGaussianRegressionModel, with data and posterior sampler
  //   set.  The model is ready for MCMC.
  Ptr<HierarchicalGaussianRegressionModel> SpecifyModel(
      SEXP r_regression_suf_list,
      SEXP r_coefficient_prior,
      SEXP r_coefficient_mean_hyperprior,
      SEXP r_coefficient_variance_hyperprior,
      SEXP r_residual_precision_prior,
      SEXP r_sampling_method,
      RListIoManager *io_manager) {
    PriorExtractor prior_extractor(r_coefficient_prior,
                                   r_coefficient_mean_hyperprior,
                                   r_coefficient_variance_hyperprior,
                                   r_sampling_method);
    Ptr<MvnModel> prior = prior_extractor.extract();
    NEW(UnivParams, residual_variance)(1.0);
    NEW(HierarchicalGaussianRegressionModel, model)(prior,
                                                    residual_variance);

    int number_of_groups = Rf_length(r_regression_suf_list);
    std::vector<Ptr<VectorParams>> coefficient_parameters;
    for (int i = 0; i < number_of_groups; ++i) {
      Ptr<RegSuf> suf = CreateRegSuf(VECTOR_ELT(r_regression_suf_list, i));
      model->add_data(suf);
      coefficient_parameters.push_back(model->data_model(i)->coef_prm());
    }

    RInterface::SdPrior sd_prior(r_residual_precision_prior);
    NEW(ChisqModel, residual_precision_prior)(sd_prior.prior_df(),
                                              sd_prior.prior_guess());

    if (prior_extractor.ASIS()) {
      NEW(HierGaussianRegressionAsisSampler, sampler)(
          model.get(),
          prior_extractor.mean_hyperprior(),
          prior_extractor.variance_hyperprior(),
          residual_precision_prior);
      model->set_method(sampler);
    } else {
      NEW(HierarchicalGaussianRegressionSampler, sampler)(
          model.get(),
          residual_precision_prior);
      model->set_method(sampler);
    }

    io_manager->add_list_element(new VectorListElement(
        model->prior()->Mu_prm(), "prior.mean"));
    io_manager->add_list_element(new SpdListElement(
        model->prior()->Sigma_prm(), "prior.variance"));
    io_manager->add_list_element(new StandardDeviationListElement(
        residual_variance, "residual.sd"));
    io_manager->add_list_element(new HierarchicalVectorListElement(
        coefficient_parameters, "coefficients"));
    return model;
  }
}  // namespace

extern "C" {
  // Args:
  //   r_regression_suf_list: An R list of R's RegressionSuf objects.  Each list
  //     element contains the sufficient statistics for a particular group in
  //     the hierarchical model.
  //   r_coefficient_prior: An object of class MvnPrior giving the initial
  //     values of the prior distribution describing the distribution of
  //     regression coefficients across groups.
  //   r_coefficient_mean_hyperprior: An object of class MvnPrior, or NULL.  If
  //     NULL then the mean of r_coefficient_prior will not be updated as part
  //     of the MCMC.  Otherwise a PosteriorSampler using this prior will be
  //     used to update the mean of r_coefficient_prior in the MCMC algorithm.
  //   r_coefficient_variance_hyperprior_: An object of class
  //     InverseWishartPrior, or NULL.  If NULL then the variance of
  //     r_coefficient_prior will not be updated as part of the MCMC.  Otherwise
  //     a PosteriorSampler using this prior will be used to update the variance
  //     of r_coefficient_prior in the MCMC algorithm.
  //   r_residual_precision_prior: An object of class SdPrior giving the prior
  //     distribution for the residual precision.
  //   r_niter:  The desired number of MCMC iterations.
  //   r_ping: The frequency with which to print status updates to the terminal
  //     when the MCMC algorithm is running.
  //   r_seed: An integer to use as a seed for the C++ random number generator
  //     (or NULL).
  SEXP boom_nested_regression_wrapper(
      SEXP r_regression_suf_list,
      SEXP r_coefficient_prior,
      SEXP r_coefficient_mean_hyperprior,
      SEXP r_coefficient_variance_hyperprior,
      SEXP r_residual_precision_prior,
      SEXP r_niter,
      SEXP r_ping,
      SEXP r_sampling_method,
      SEXP r_seed) {
    RErrorReporter error_reporter;
    RMemoryProtector protector;
    try {
      seed_rng_from_R(r_seed);
      RListIoManager io_manager;
      Ptr<Model> model = SpecifyModel(
          r_regression_suf_list,
          r_coefficient_prior,
          r_coefficient_mean_hyperprior,
          r_coefficient_variance_hyperprior,
          r_residual_precision_prior,
          r_sampling_method,
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
    } catch(std::exception &e) {
      RInterface::handle_exception(e);
    } catch(...) {
      RInterface::handle_unknown_exception();
    }
    return R_NilValue;
  }

}  // extern "C"
