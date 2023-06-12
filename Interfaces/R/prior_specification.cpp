/*
  Copyright (C) 2005-2011 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/prior_specification.hpp"
#include "Models/BetaModel.hpp"
#include "Models/DiscreteUniformModel.hpp"
#include "Models/GammaModel.hpp"
#include "Models/TruncatedGammaModel.hpp"
#include "Models/GaussianModel.hpp"
#include "Models/LognormalModel.hpp"
#include "Models/MarkovModel.hpp"
#include "Models/PoissonModel.hpp"
#include "Models/PosteriorSamplers/MarkovConjSampler.hpp"
#include "Models/UniformModel.hpp"
#include "Models/MvnGivenSigma.hpp"
#include "Models/WishartModel.hpp"


#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"

#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace RInterface {
    SdPrior::SdPrior(SEXP prior)
        : prior_guess_(Rf_asReal(getListElement(prior, "prior.guess"))),
          prior_df_(Rf_asReal(getListElement(prior, "prior.df"))),
          initial_value_(Rf_asReal(getListElement(prior, "initial.value"))),
          fixed_(Rf_asLogical(getListElement(prior, "fixed"))),
          upper_limit_(Rf_asReal(getListElement(prior, "upper.limit")))
    {
      if (upper_limit_ < 0 || !R_FINITE(upper_limit_)) {
        upper_limit_ = BOOM::infinity();
      }
    }

    std::ostream & SdPrior::print(std::ostream &out) const {
      out << "prior_guess_   = " << prior_guess_ << std::endl
          << "prior_df_      = " << prior_df_ << std::endl
          << "initial_value_ = " << initial_value_ << std::endl
          << "fixed          = " << fixed_ << std::endl
          << "upper_limit_   = " << upper_limit_ << std::endl;
      return out;
    }

    NormalPrior::NormalPrior(SEXP prior)
        : mu_(Rf_asReal(getListElement(prior, "mu"))),
          sigma_(Rf_asReal(getListElement(prior, "sigma"))),
          initial_value_(Rf_asReal(getListElement(prior, "initial.value"))) {
      int is_fixed = Rf_asLogical(getListElement(prior, "fixed"));
      if (is_fixed == 1) {
        fixed_ = true;
      } else if (is_fixed == 0) {
        fixed_ = false;
      } else {
        report_error("Strange value of 'fixed' in NormalPrior constructor.");
      }
    }

    std::ostream & NormalPrior::print(std::ostream &out) const {
      out << "mu =     " << mu_ << std::endl
          << "sigma_ = " << sigma_ << std::endl
          << "init   = " << initial_value_ << std::endl;
      return out;
    }

    BetaPrior::BetaPrior(SEXP prior)
        : a_(Rf_asReal(getListElement(prior, "a"))),
          b_(Rf_asReal(getListElement(prior, "b"))),
          initial_value_(Rf_asReal(getListElement(prior, "initial.value")))
    {}

    std::ostream & BetaPrior::print(std::ostream &out) const {
      out << "a = " << a_ << "b = " << b_;
      return out;
    }

    GammaPrior::GammaPrior(SEXP prior)
        : a_(Rf_asReal(getListElement(prior, "a"))),
          b_(Rf_asReal(getListElement(prior, "b")))
    {
      RMemoryProtector protector;
      SEXP rinitial_value = protector.protect(
          getListElement(prior, "initial.value"));
      if (rinitial_value != R_NilValue) {
        initial_value_ = Rf_asReal(rinitial_value);
      } else {
        initial_value_ = a_ / b_;
      }
    }

    std::ostream & GammaPrior::print(std::ostream &out) const {
      out << "a = " << a_ << "b = " << b_;
      return out;
    }

    TruncatedGammaPrior::TruncatedGammaPrior(SEXP prior)
        : GammaPrior(prior),
          lower_truncation_point_(Rf_asReal(getListElement(
              prior, "lower.truncation.point"))),
          upper_truncation_point_(Rf_asReal(getListElement(
              prior, "upper.truncation.point")))
    {}

    std::ostream & TruncatedGammaPrior::print(std::ostream &out) const {
      GammaPrior::print(out) << " (" << lower_truncation_point_
          << ", " << upper_truncation_point_ << ") ";
      return out;
    }

    MvnPrior::MvnPrior(SEXP prior)
        : mu_(ToBoomVector(getListElement(prior, "mean"))),
          Sigma_(ToBoomSpdMatrix(getListElement(prior, "variance")))
    {}

    std::ostream & MvnPrior::print(std::ostream &out) const {
      out << "mu: " << mu_ << std::endl
          << "Sigma:" << std::endl
          << Sigma_;
      return out;
    }

    MvnGivenSigmaMatrixPrior::MvnGivenSigmaMatrixPrior(SEXP r_prior)
        : mu_(ToBoomVector(getListElement(r_prior, "mean"))),
          sample_size_(Rf_asReal(getListElement(r_prior, "sample.size")))
    {}

    MvnGivenSigma *MvnGivenSigmaMatrixPrior::boom() const {
      return new MvnGivenSigma(mu(), kappa());
    }

    Ar1CoefficientPrior::Ar1CoefficientPrior(SEXP prior)
        : NormalPrior(prior),
          force_stationary_(Rf_asLogical(getListElement(
              prior, "force.stationary"))),
          force_positive_(Rf_asLogical(getListElement(
              prior, "force.positive"))) {}

    std::ostream & Ar1CoefficientPrior::print(std::ostream &out) const {
      NormalPrior::print(out) << "force_stationary_ = "
                              << force_stationary_ << std::endl;
      return out;
    }

    ConditionalNormalPrior::ConditionalNormalPrior(SEXP prior)
        : mu_(Rf_asReal(getListElement(prior, "mu"))),
          sample_size_(Rf_asReal(getListElement(prior, "sample.size")))
    {}

    std::ostream & ConditionalNormalPrior::print(std::ostream & out) const {
      out << "prior mean: " << mu_ << std::endl
          << "prior sample size for prior mean:" << sample_size_;
      return out;
    }

    NormalInverseGammaPrior::NormalInverseGammaPrior(SEXP prior)
        : prior_mean_guess_(Rf_asReal(getListElement(
            prior, "mu.guess"))),
          prior_mean_sample_size_(Rf_asReal(getListElement(
              prior, "mu.guess.weight"))),
          sd_prior_(getListElement(prior, "sigma.prior"))
    {}

    std::ostream & NormalInverseGammaPrior::print(std::ostream &out) const {
      out << "prior_mean_guess        = " << prior_mean_guess_ << std::endl
          << "prior_mean_sample_size: = " << prior_mean_sample_size_
          << std::endl
          << "prior for sigma: " << std::endl
          << sd_prior_;
      return out;
    }

    DirichletPrior::DirichletPrior(SEXP prior)
        : prior_counts_(ToBoomVector(
            getListElement(prior, "prior.counts")))
    {}

    const Vector & DirichletPrior::prior_counts() const {
      return prior_counts_;
    }

    int DirichletPrior::dim() const {
      return prior_counts_.size();
    }

    MarkovPrior::MarkovPrior(SEXP prior)
        : transition_counts_(ToBoomMatrix(getListElement(
            prior, "prior.transition.counts"))),
          initial_state_counts_(ToBoomVector(getListElement(
              prior, "prior.initial.state.counts")))
    {}

    std::ostream & MarkovPrior::print(std::ostream &out) const {
      out << "prior transition counts: " << std::endl
          << transition_counts_ << std::endl
          << "prior initial state counts:" << std::endl
          << initial_state_counts_;
      return out;
    }

    MarkovModel * MarkovPrior::create_markov_model() const {
      MarkovModel * ans(new MarkovModel(transition_counts_.nrow()));
      Ptr<MarkovConjSampler> sampler(new MarkovConjSampler(
          ans, transition_counts_, initial_state_counts_));
      ans->set_method(sampler);
      return ans;
    }

    InverseWishartPrior::InverseWishartPrior(SEXP prior)
        : variance_guess_weight_(Rf_asReal(getListElement(
              prior, "variance.guess.weight"))),
          variance_guess_(ToBoomSpdMatrix(getListElement(
              prior, "variance.guess")))
    {}

    WishartModel * InverseWishartPrior::boom() const {
      return new WishartModel(variance_guess_weight_, variance_guess_);
    }

    NormalInverseWishartPrior::NormalInverseWishartPrior(SEXP prior)
        : mu_guess_(ToBoomVector(getListElement(prior, "mean.guess"))),
          mu_guess_weight_(Rf_asReal(getListElement(
              prior, "mean.guess.weight"))),
          sigma_guess_(ToBoomSpdMatrix(getListElement(
              prior, "variance.guess"))),
          sigma_guess_weight_(Rf_asReal(getListElement(
              prior, "variance.guess.weight")))
    {}

    std::ostream & NormalInverseWishartPrior::print(std::ostream &out) const {
      out << "the prior mean for mu:" << std::endl
          << mu_guess_ << std::endl
          << "prior sample size for mu0: " << mu_guess_weight_ << std::endl
          << "prior sample size for Sigma_guess: " << sigma_guess_weight_
          << std::endl
          << "prior guess at Sigma: " << std::endl
          << sigma_guess_ << std::endl;
      return out;
    }

    MvnIndependentSigmaPrior::MvnIndependentSigmaPrior(SEXP prior)
        : mu_prior_(getListElement(prior, "mu.prior"))
    {
      int n = mu_prior_.mu().size();
      sigma_priors_.reserve(n);
      SEXP sigma_prior_list = getListElement(prior, "sigma.prior");
      for (int i = 0; i < n; ++i) {
        SdPrior sigma_prior(VECTOR_ELT(sigma_prior_list, i));
        sigma_priors_.push_back(sigma_prior);
      }
    }

    MvnDiagonalPrior::MvnDiagonalPrior(SEXP prior)
        : mean_(ToBoomVector(getListElement(prior, "mean"))),
          sd_(ToBoomVector(getListElement(prior, "sd")))
    {}

    //==========================================================================
    ScaledMatrixNormalPrior::ScaledMatrixNormalPrior(SEXP r_prior)
        : mean_(ToBoomMatrix(getListElement(r_prior, "mean", true))),
          sample_size_(Rf_asReal(getListElement(r_prior, "nu", true)))
    {}

    //==========================================================================
    DiscreteUniformPrior::DiscreteUniformPrior(SEXP prior)
        :lo_(Rf_asInteger(getListElement(prior, "lower.limit"))),
         hi_(Rf_asInteger(getListElement(prior, "upper.limit")))
    {
      if (hi_ < lo_) {
        report_error("hi < lo in DiscreteUniformPrior.");
      }
      log_normalizing_constant_ = -log1p(hi_ - lo_);
    }

    double DiscreteUniformPrior::logp(int value) const {
      if (value < lo_ || value > hi_) {
        return negative_infinity();
      }
      return log_normalizing_constant_;
    }

    PoissonPrior::PoissonPrior(SEXP prior)
        : lambda_(Rf_asReal(getListElement(prior, "mean"))),
          lo_(Rf_asReal(getListElement(prior, "lower.limit"))),
          hi_(Rf_asReal(getListElement(prior, "upper.limit")))
    {
      if (lambda_ <= 0) {
        report_error("lambda must be positive in PoissonPrior");
      }
      if (hi_ < lo_) {
        report_error("upper.limit < lower.limit in PoissonPrior.");
      }
      log_normalizing_constant_ = log(ppois(hi_, lambda_)
                                      - ppois(lo_ - 1, lambda_));
    }

    double PoissonPrior::logp(int value) const {
      return dpois(value, lambda_, true) - log_normalizing_constant_;
    }

    PointMassPrior::PointMassPrior(SEXP prior)
        : location_(Rf_asInteger(getListElement(prior, "location")))
    {}

    double PointMassPrior::logp(int value) const {
      return value == location_ ? 0 : negative_infinity();
    }

    RegressionCoefficientConjugatePrior::RegressionCoefficientConjugatePrior(
        SEXP r_prior)
        : mean_(ToBoomVector(getListElement(r_prior, "mean"))),
          sample_size_(Rf_asReal(getListElement(r_prior, "sample.size"))),
          additional_prior_precision_(ToBoomVector(getListElement(
              r_prior, "additional.prior.precision"))),
          diagonal_weight_(Rf_asReal(getListElement(
              r_prior, "diagonal.weight")))
    {}

    UniformPrior::UniformPrior(SEXP r_prior)
        : lo_(Rf_asReal(getListElement(r_prior, "lo"))),
          hi_(Rf_asReal(getListElement(r_prior, "hi"))),
          initial_value_(Rf_asReal(getListElement(r_prior, "initial.value")))
    {}

    Ptr<LocationScaleDoubleModel> create_location_scale_double_model(
        SEXP r_spec, bool throw_on_failure) {
      if (Rf_inherits(r_spec, "GammaPrior")) {
        GammaPrior spec(r_spec);
        return new GammaModel(spec.a(), spec.b());
      } else if (Rf_inherits(r_spec, "BetaPrior")) {
        BetaPrior spec(r_spec);
        return new BetaModel(spec.a(), spec.b());
      } else if (Rf_inherits(r_spec, "NormalPrior")) {
        NormalPrior spec(r_spec);
        return new GaussianModel(spec.mu(), spec.sigma() * spec.sigma());
      } else if (Rf_inherits(r_spec, "UniformPrior")) {
        double lo = Rf_asReal(getListElement(r_spec, "lo"));
        double hi = Rf_asReal(getListElement(r_spec, "hi"));
        return new UniformModel(lo, hi);
      } else if (Rf_inherits(r_spec, "LognormalPrior")) {
        double mu = Rf_asReal(getListElement(r_spec, "mu"));
        double sigma = Rf_asReal(getListElement(r_spec, "sigma"));
        return new LognormalModel(mu, sigma);
      }
      if (throw_on_failure) {
        report_error("Could not convert specification into a "
                     "LocationScaleDoubleModel");
      }
      return nullptr;
    }

    Ptr<DoubleModel> create_double_model(SEXP r_spec) {
      Ptr<LocationScaleDoubleModel> ans =
          create_location_scale_double_model(r_spec, false);
      if (!!ans) {
        return ans;
      } else if (Rf_inherits(r_spec, "TruncatedGammaPrior")) {
        TruncatedGammaPrior spec(r_spec);
        return new TruncatedGammaModel(
            spec.a(), spec.b(), spec.lower_truncation_point(),
            spec.upper_truncation_point());
      }
      report_error("Could not convert specification into a DoubleModel");
      return nullptr;
    }

    Ptr<DiffDoubleModel> create_diff_double_model(SEXP r_spec) {
      if (Rf_inherits(r_spec, "GammaPrior")) {
        GammaPrior spec(r_spec);
        return new GammaModel(spec.a(), spec.b());
      } else if (Rf_inherits(r_spec, "TruncatedGammaPrior")) {
        TruncatedGammaPrior spec(r_spec);
        return new TruncatedGammaModel(
            spec.a(), spec.b(), spec.lower_truncation_point(),
            spec.upper_truncation_point());
      } else if (Rf_inherits(r_spec, "BetaPrior")) {
        BetaPrior spec(r_spec);
        return new BetaModel(spec.a(), spec.b());
      } else if (Rf_inherits(r_spec, "NormalPrior")) {
        NormalPrior spec(r_spec);
        return new GaussianModel(spec.mu(), spec.sigma() * spec.sigma());
      } else if (Rf_inherits(r_spec, "SdPrior")) {
        SdPrior spec(r_spec);
        double shape = spec.prior_df() / 2;
        double sum_of_squares = square(spec.prior_guess()) * spec.prior_df();
        double scale = sum_of_squares / 2;
        if (spec.upper_limit() < infinity()) {
          double lower_limit = 1.0 / square(spec.upper_limit());
          double upper_limit = infinity();
          return new TruncatedGammaModel(shape, scale, lower_limit,
                                         upper_limit);
        } else {
          return new GammaModel(shape, scale);
        }
      } else if (Rf_inherits(r_spec, "UniformPrior")) {
        UniformPrior spec(r_spec);
        return new UniformModel(spec.lo(), spec.hi());
      }
      report_error("Could not convert specification into a DiffDoubleModel");
      return nullptr;
    }

    Ptr<IntModel> create_int_model(SEXP r_spec) {
      if (Rf_inherits(r_spec, "DiscreteUniformPrior")) {
        DiscreteUniformPrior spec(r_spec);
        return new DiscreteUniformModel(spec.lo(), spec.hi());
      } else if (Rf_inherits(r_spec, "PoissonPrior")) {
        PoissonPrior spec(r_spec);
        return new PoissonModel(spec.lambda());
      } else if (Rf_inherits(r_spec, "PointMassPrior")) {
        PointMassPrior spec(r_spec);
        return new DiscreteUniformModel(spec.location(), spec.location());
      } else {
        report_error("Could not convert specification into an IntModel.");
        return nullptr;
      }
    }


    //===========================================================================
    // Setting priors for regression models.
    //===========================================================================


    // Set the coefficients equal to their initial values, and determine
    // which coefficients are initially excluded (i.e. forced to zero).
    // Args:
    //   initial_beta:  Vector containing initial coefficients.
    //   prior_inclusion_probabilities: Prior probabilities that each
    //     coefficient is nonzero.
    //   model:  The model that owns the coefficients.
    //   sampler:  The sampler that will make posterior draws for the model.
    template<class SAMPLER>
    void InitializeSpikeSlabCoefficients(
        const BOOM::Vector &initial_beta,
        const BOOM::Vector &prior_inclusion_probabilities,
        Ptr<GlmModel>  model,
        BOOM::Ptr<SAMPLER> sampler) {
      model->set_Beta(initial_beta);
      if (min(prior_inclusion_probabilities) >= 1.0) {
        // Ensure all coefficients are included if you're not going to
        // do model averaging.
        sampler->allow_model_selection(false);
        model->coef().add_all();
      } else {
        // Model averaging is desired.  "Small" coefficients start off
        // excluded from the model.  Large ones start off included.
        // Adding or dropping is idempotent, so no need to worry about
        // dropping an already excluded coefficient.
        for (int i = 0; i < initial_beta.size(); ++i) {
          if (fabs(initial_beta[i]) < 1e-8) {
            model->coef().drop(i);
          } else {
            model->coef().add(i);
          }

          // Respect absolute prior opinions about coefficients,
          // regardless of whether the initial coefficient is large or
          // small,
          if (prior_inclusion_probabilities[i] >= 1.0) {
            model->add(i);
          } else if (prior_inclusion_probabilities[i] <= 0.0) {
            model->drop(i);
          }
        }
      }
    }

    inline void SetIndependentSpikeSlabPrior(RegressionModel *model,
                                             SEXP r_prior) {
      IndependentRegressionSpikeSlabPrior prior(r_prior, model->Sigsq_prm());
      NEW(BregVsSampler, sampler)(model, prior.slab(), prior.siginv_prior(),
                                  prior.spike());
      if (prior.sigma_upper_limit() > 0 && prior.sigma_upper_limit() < infinity()) {
        sampler->set_sigma_upper_limit(prior.sigma_upper_limit());
      }
      model->set_method(sampler);
      InitializeSpikeSlabCoefficients(
          model->Beta(), prior.spike()->prior_inclusion_probabilities(),
          model, sampler);
    }

    inline void SetSpikeSlabPrior(RegressionModel *model, SEXP r_prior) {
      RegressionConjugateSpikeSlabPrior prior(r_prior, model->Sigsq_prm());
      NEW(BregVsSampler, sampler)(
          model, prior.slab(), prior.siginv_prior(), prior.spike());
      if (prior.sigma_upper_limit() > 0 && prior.sigma_upper_limit() < infinity()) {
        sampler->set_sigma_upper_limit(prior.sigma_upper_limit());
      }
      model->set_method(sampler);
      InitializeSpikeSlabCoefficients(
          model->Beta(),
          prior.spike()->prior_inclusion_probabilities(),
          model,
          sampler);
    }

    void SetRegressionSampler(RegressionModel *model, SEXP r_prior) {
      if (Rf_inherits(r_prior, "RegressionCoefficientConjugatePrior")) {
        report_error("TODO");
      } else if (Rf_inherits(r_prior, "RegressionConjugatePrior")) {
        report_error("TODO");
      } else if (Rf_inherits(r_prior, "SpikeSlabPrior")) {
        SetSpikeSlabPrior(model, r_prior);
      } else if (Rf_inherits(r_prior, "IndependentSpikeSlabPrior")) {
        report_error("TODO");
      } else {
        ReportBadClass("Unsupported object passed to SetRegressionSampler.",
                       r_prior);
      }
    }

  }  // namespace RInterface
}  // namespace BOOM
