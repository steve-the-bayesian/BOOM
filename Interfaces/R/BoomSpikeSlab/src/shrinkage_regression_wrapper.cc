#include "Models/GammaModel.hpp"
#include "Models/GaussianModel.hpp"
#include "Models/GaussianModelGivenSigma.hpp"
#include "Models/PosteriorSamplers/GaussianVarSampler.hpp"
#include "Models/PosteriorSamplers/GaussianMeanSampler.hpp"
#include "Models/Glm/PosteriorSamplers/RegressionShrinkageSampler.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/PosteriorSamplers/GaussianGivenSigmaSampler.hpp"

#include "cpputil/math_utils.hpp"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/handle_exception.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/prior_specification.hpp"
#include "r_interface/seed_rng_from_R.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::RInterface;

  typedef RegressionShrinkageSampler::CoefficientGroup CoefficientGroup;
  class CoefficientGroupExtractor {
   public:
    CoefficientGroup Extract(SEXP r_coefficient_group,
                             std::vector<Ptr<UnivParams>> *prior_means,
                             std::vector<Ptr<UnivParams>> *prior_variances) {
      r_coefficient_group_ = r_coefficient_group;
      Ptr<GaussianModel> prior = ExtractPrior();
      prior_means->push_back(prior->Mu_prm());
      prior_variances->push_back(prior->Sigsq_prm());
      return CoefficientGroup(prior, ExtractIndices());
    }

   private:
    std::vector<int> ExtractIndices() {
      return ToIntVector(getListElement(r_coefficient_group_, "indices"), true);
    }

    Ptr<GaussianModel> ExtractPrior() {
      SEXP r_prior = getListElement(r_coefficient_group_, "prior");
      Ptr<GaussianModel> prior;
      if (Rf_isNull(r_prior)) {
        prior = new GaussianModel(0, 100);
      } else {
        RInterface::NormalPrior prior_spec(r_prior);
        prior = new GaussianModel(prior_spec.mu(), square(prior_spec.sigma()));
      }
      ExtractMeanHyperprior(prior);
      ExtractVarianceHyperprior(prior);
      return prior;
    }

    void ExtractMeanHyperprior(const Ptr<GaussianModel> &prior) {
      SEXP r_mean_hyperprior = getListElement(
          r_coefficient_group_, "mean.hyperprior");
      if (!Rf_isNull(r_mean_hyperprior)) {
        RInterface::NormalPrior spec(r_mean_hyperprior);
        NEW(GaussianModel, mean_hyperprior)(spec.mu(), spec.sigsq());
        NEW(GaussianMeanSampler, mean_sampler)(prior.get(), mean_hyperprior);
        prior->set_method(mean_sampler);
      }
    }

    void ExtractVarianceHyperprior(const Ptr<GaussianModel> &prior) {
      SEXP r_sd_hyperprior = getListElement(
          r_coefficient_group_, "sd.hyperprior");
      if (!Rf_isNull(r_sd_hyperprior)) {
        RInterface::SdPrior spec(r_sd_hyperprior);
        NEW(ChisqModel, precision_prior)(spec.prior_df(), spec.prior_guess());
        NEW(GaussianVarSampler, var_sampler)(prior.get(), precision_prior);
        if (std::isfinite(spec.upper_limit())) {
          var_sampler->set_sigma_upper_limit(spec.upper_limit());
        }
        prior->set_method(var_sampler);
      }
    }

    SEXP r_coefficient_group_;
  };

  //----------------------------------------------------------------------
  Ptr<RegressionModel> SpecifyModel(SEXP r_gaussian_suf,
                                    SEXP r_coefficient_groups,
                                    SEXP r_residual_prior,
                                    RListIoManager *io_manager) {
    NEW(NeRegSuf, suf)(ToBoomMatrix(getListElement(r_gaussian_suf, "xtx")),
                       ToBoomVector(getListElement(r_gaussian_suf, "xty")),
                       Rf_asReal(getListElement(r_gaussian_suf, "yty")),
                       Rf_asReal(getListElement(r_gaussian_suf, "n")),
                       ToBoomVector(getListElement(r_gaussian_suf, "xbar")));

    NEW(RegressionModel, model)(suf->size());
    model->set_suf(suf);
    Ptr<UnivParams> sigsq_prm(model->Sigsq_prm());

    std::vector<CoefficientGroup> groups;
    int number_of_groups = Rf_length(r_coefficient_groups);
    std::vector<Ptr<UnivParams>> prior_means;
    std::vector<Ptr<UnivParams>> prior_variances;
    CoefficientGroupExtractor cg_extractor;
    for (int i = 0; i < number_of_groups; ++i) {
      groups.push_back(cg_extractor.Extract(
          VECTOR_ELT(r_coefficient_groups, i),
          &prior_means,
          &prior_variances));
    }

    RInterface::SdPrior residual_precision_prior_spec(r_residual_prior);
    NEW(ChisqModel, residual_precision_prior)(
        residual_precision_prior_spec.prior_df(),
        residual_precision_prior_spec.prior_guess());

    NEW(RegressionShrinkageSampler, sampler)(
        model.get(),
        residual_precision_prior,
        groups);
    model->set_method(sampler);

    io_manager->add_list_element(
        new GlmCoefsListElement(model->coef_prm(), "coefficients"));
    io_manager->add_list_element(
        new StandardDeviationListElement(model->Sigsq_prm(), "residual.sd"));
    io_manager->add_list_element(
        new UnivariateCollectionListElement(prior_means, "group.means"));
    io_manager->add_list_element(
        new SdCollectionListElement(prior_variances, "group.sds"));
    return model;
}

}  // namespace

extern "C" {

  SEXP boom_shrinkage_regression_wrapper(
      SEXP r_regression_suf,
      SEXP r_coefficient_groups,
      SEXP r_residual_precision_prior,
      SEXP r_niter,
      SEXP r_ping,
      SEXP r_seed) {
    RErrorReporter error_reporter;
    RMemoryProtector protector;
    try {
      seed_rng_from_R(r_seed);
      RListIoManager io_manager;
      Ptr<Model> model = SpecifyModel(
          r_regression_suf, r_coefficient_groups, r_residual_precision_prior,
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
