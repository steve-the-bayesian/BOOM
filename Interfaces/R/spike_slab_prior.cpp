#include "r_interface/prior_specification.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/MvnBase.hpp"
#include "Models/MvnModel.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {
  namespace RInterface {

    namespace {
      int GetMaxFlips(SEXP r_prior) {
        SEXP r_max_flips = getListElement(r_prior, "max.flips");
        if (Rf_isNull(r_max_flips)) {
          return -1;
        } else {
          return Rf_asInteger(r_max_flips);
        }
      }

      double GetSigmaUpperLimit(SEXP r_prior) {
        SEXP r_sigma_upper_limit = getListElement(r_prior, "sigma.upper.limit");
        double inf = BOOM::infinity();
        if (Rf_isNull(r_sigma_upper_limit)) {
          return inf;
        } else {
          double sigma_upper_limit = Rf_asReal(r_sigma_upper_limit);
          if (sigma_upper_limit > 0 && sigma_upper_limit < inf) {
            return sigma_upper_limit;
          } else {
            // Being careful to use BOOM::infinity here, in case R
            // does something stupid vis-a-vis infinity.
            return inf;
          }
        }
      }
    }  // namespace

    SpikeSlabGlmPrior::SpikeSlabGlmPrior(SEXP r_prior)
        : spike_(new VariableSelectionPrior(ToBoomVector(getListElement(
              r_prior, "prior.inclusion.probabilities", true)))),
          max_flips_(GetMaxFlips(r_prior))
    {
      Vector prior_mean = ToBoomVector(getListElement(r_prior, "mu"));
      if (Rf_inherits(r_prior, "SpikeSlabPrior")
          || Rf_inherits(r_prior, "LogitZellnerPrior")
          || Rf_inherits(r_prior, "PoissonZellnerPrior")
          || Rf_inherits(r_prior, "SpikeSlabGlmPriorDirect")) {
        SpdMatrix prior_precision = ToBoomSpdMatrix(getListElement(
            r_prior, "siginv"));
        slab_.reset(new MvnModel(prior_mean, prior_precision, true));
      } else if (Rf_inherits(r_prior, "IndependentSpikeSlabPrior")) {
        Vector prior_variance_diagonal(ToBoomVector(getListElement(
            r_prior, "prior.variance.diagonal")));
        slab_.reset(new IndependentMvnModel(
            prior_mean,
            prior_variance_diagonal));
      } else {
        report_error("Unknown R object passed to SpikeSlabPrior");
      }
    }

    ConditionalZellnerPrior::ConditionalZellnerPrior(
        SEXP r_prior) {
      if (!Rf_inherits(r_prior, "ConditionalZellnerPrior")) {
        report_error("Expected a ConditionalZellnerPrior.");
      }
      spike_.reset(new VariableSelectionPrior(ToBoomVector(getListElement(
          r_prior, "prior.inclusion.probabilities", true))));
      prior_mean_ = ToBoomVector(getListElement(r_prior, "prior.mean", true));
      diagonal_shrinkage_ = Rf_asReal(getListElement(
          r_prior, "diagonal.shrinkage", true));
      prior_information_weight_ = Rf_asReal(getListElement(
          r_prior, "prior.information.weight", true));
      max_flips_ = GetMaxFlips(r_prior);
    }
    
    RegressionConjugateSpikeSlabPrior::RegressionConjugateSpikeSlabPrior(
        SEXP r_spike_slab_prior,
        const Ptr<UnivParams> &residual_variance)
        : spike_(new VariableSelectionPrior(ToBoomVector(getListElement(
              r_spike_slab_prior, "prior.inclusion.probabilities")))),
          siginv_prior_(new ChisqModel(
              Rf_asReal(getListElement(r_spike_slab_prior, "prior.df")),
              Rf_asReal(getListElement(r_spike_slab_prior, "sigma.guess")))),
          max_flips_(GetMaxFlips(r_spike_slab_prior)),
          sigma_upper_limit_(GetSigmaUpperLimit(r_spike_slab_prior))
    {
      Vector mu = ToBoomVector(getListElement(
          r_spike_slab_prior, "mu"));
      if (Rf_inherits(r_spike_slab_prior, "SpikeSlabPrior")
          || Rf_inherits(r_spike_slab_prior, "SpikeSlabPriorDirect")) {
        slab_.reset(new MvnGivenScalarSigma(
            mu,
            ToBoomSpdMatrix(getListElement(r_spike_slab_prior, "siginv")),
            residual_variance));
      } else if (Rf_inherits(r_spike_slab_prior, "IndependentSpikeSlabPrior")) {
        slab_.reset(new IndependentMvnModelGivenScalarSigma(
            mu,
            ToBoomVector(getListElement(
                r_spike_slab_prior, "prior.variance.diagonal")),
            residual_variance));
      }
    }

    namespace {
      typedef StudentRegressionConjugateSpikeSlabPrior SRCSSP;
      typedef StudentRegressionNonconjugateSpikeSlabPrior SRNSSP;
      typedef StudentIndependentSpikeSlabPrior SISSP;
    }

    SRCSSP::StudentRegressionConjugateSpikeSlabPrior(
        SEXP r_prior, const Ptr<UnivParams> &residual_variance)
        : RegressionConjugateSpikeSlabPrior(r_prior, residual_variance),
          df_prior_(create_double_model(getListElement(
              r_prior, "degrees.of.freedom.prior")))
    {}

    RegressionNonconjugateSpikeSlabPrior::RegressionNonconjugateSpikeSlabPrior(
        SEXP r_spike_slab_prior)
        : SpikeSlabGlmPrior(r_spike_slab_prior),
          sigma_upper_limit_(GetSigmaUpperLimit(r_spike_slab_prior))
    {
      double prior_df = Rf_asReal(getListElement(
          r_spike_slab_prior, "prior.df"));
      double sigma_guess = Rf_asReal(getListElement(
          r_spike_slab_prior, "sigma.guess"));
      siginv_prior_.reset(new ChisqModel(prior_df, sigma_guess));
    }

    ArSpikeSlabPrior::ArSpikeSlabPrior(SEXP r_prior)
        : RegressionNonconjugateSpikeSlabPrior(r_prior),
          truncate_(false)
    {
      truncate_ = Rf_asLogical(getListElement(r_prior, "truncate"));
    }

    SRNSSP::StudentRegressionNonconjugateSpikeSlabPrior(SEXP r_prior)
        : RegressionNonconjugateSpikeSlabPrior(r_prior),
          df_prior_(create_double_model(getListElement(
              r_prior, "degrees.of.freedom.prior")))
    {}

    IndependentRegressionSpikeSlabPrior::IndependentRegressionSpikeSlabPrior(
        SEXP r_prior, const Ptr<UnivParams> &sigsq)
        : spike_(new VariableSelectionPrior(ToBoomVector(getListElement(
              r_prior, "prior.inclusion.probabilities", true)))),
          slab_(new IndependentMvnModelGivenScalarSigma(
              ToBoomVector(getListElement(r_prior, "mu")),
              ToBoomVector(getListElement(r_prior, "prior.variance.diagonal")),
              sigsq)),
          siginv_prior_(new ChisqModel(
              Rf_asReal(getListElement(r_prior, "prior.df")),
              Rf_asReal(getListElement(r_prior, "sigma.guess")))),
          max_flips_(GetMaxFlips(r_prior)),
          sigma_upper_limit_(GetSigmaUpperLimit(r_prior))
    {}

    SISSP::StudentIndependentSpikeSlabPrior(
        SEXP r_prior, const Ptr<UnivParams> &sigsq)
        : IndependentRegressionSpikeSlabPrior(r_prior, sigsq),
          df_prior_(create_double_model(getListElement(
              r_prior, "degrees.of.freedom.prior")))
    {}

  }  // namespace RInterface
}  // namespace BOOM
