// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#ifndef BOOM_BREG_VS_SAMPLER_HPP
#define BOOM_BREG_VS_SAMPLER_HPP
#include "Models/ChisqModel.hpp"
#include "Models/GammaModel.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/Glm/PosteriorSamplers/CorrelationMap.hpp"
#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/MvnGivenSigma.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {
  struct ZellnerPriorParameters {
    Vector prior_inclusion_probabilities;
    Vector prior_beta_guess;
    double prior_beta_guess_weight;
    SpdMatrix prior_beta_information;  // this is Omega^{-1} in the math below
    double prior_sigma_guess;
    double prior_sigma_guess_weight;
  };

  // A posterior sampler for doing "spike and slab" stochastic search
  // variable selection using Zellner's g-prior.
  //
  // prior:  beta | gamma, sigma ~ Normal(b, sigma^2 * Omega)
  //                   1/sigma^2 ~ Gamma(sigma.guess, df)
  //                       gamma ~ VsPrior (product of Bernoulli)
  //
  // A good choice for Omega^{-1} is kappa * XTX/n, which is kappa
  // 'typical' observations.
  //
  // Note that with this prior it is possible for a really poor guess
  // at the prior mean 'b' to inflate the "sum of squares" statistic
  // used to draw the variance.  A reasonable value for b is to set
  // the intercept to the sample mean of the responses, and set the
  // slopes to zero.
  class BregVsSampler : public PosteriorSampler {
   public:
    // Omega inverse is 'prior_nobs' * XTX/n. The intercept term in 'b' is ybar
    // (sample mean of the responses).  The slope terms in b are all zero.  The
    // prior for 1/sigsq is Gamma(prior_nobs/2, prior_ss/2), with prior_ss =
    // prior_nobs*sigma_guess^2, and sigma_guess =
    // sample_variance*(1-expected_rsq)
    BregVsSampler(RegressionModel *model,
                  double prior_nobs,           // Omega is prior_nobs * XTX/n
                  double expected_rsq,         // sigsq_guess = sample var*this
                  double expected_model_size,  // prior inc probs = this/dim
                  bool first_term_is_intercept = true,
                  RNG &seeding_rng = GlobalRng::rng);

    // Omega inverse is kappa*[(1-alpha)*XTX/n + alpha*(XTX/n)].  kappa is
    // 'prior_beta_nobs', and alpha is 'diagonal_shrinkage'.  The prior on
    // 1/sigsq is Gamma(prior_sigma_nobs/2, priors_ss/2) with prior_ss =
    // prior_sigma_guess^2 * prior_sigma_nobs.  b = [ybar, 0, 0, ...]
    BregVsSampler(RegressionModel *model, double prior_sigma_nobs,
                  double prior_sigma_guess, double prior_beta_nobs,
                  double diagonal_shrinkage, double prior_inclusion_probability,
                  bool force_intercept = true,
                  RNG &seeding_rng = GlobalRng::rng);

    // Use this constructor if you want full control over the parameters of the
    // prior distribution, but you don't want to supply actual model objects.
    // You won't be able to modify the values of the prior parameters
    // afterwards.
    BregVsSampler(RegressionModel *model, const Vector &prior_mean,
                  const SpdMatrix &unscaled_prior_precision, double sigma_guess,
                  double df, const Vector &prior_inclusion_probs,
                  RNG &seeding_rng = GlobalRng::rng);

    // Equivalent to the preceding constructor, but the prior parameters are
    // specified in a struct.
    BregVsSampler(RegressionModel *model, const ZellnerPriorParameters &prior,
                  RNG &seeding_rng = GlobalRng::rng);

    // This constructor offers full control.  If external copies of the pointers
    // supplied to the constructor are kept then the values of the prior
    // parameters can be modified.  This would be useful in a hierarchical
    // model, for example.
    BregVsSampler(RegressionModel *model,
                  const Ptr<MvnGivenScalarSigmaBase> &slab,
                  const Ptr<GammaModelBase> &residual_precision_prior,
                  const Ptr<VariableSelectionPrior> &spike,
                  RNG &seeding_rng = GlobalRng::rng);

    // Set the slab portion of the prior after construction.  The slab argument
    // must contain a pointer to the residual variance parameter in the
    // regression model managed by this object.
    void set_slab(const Ptr<MvnGivenScalarSigmaBase> &new_slab) {
      slab_ = check_slab_dimension(new_slab);
    }

    // Set the spike portion of the prior to something else after construction.
    void set_spike(const Ptr<VariableSelectionPrior> &new_spike) {
      spike_ = check_spike_dimension(new_spike);
    }

    // Set the residual precision prior to something else after construction.
    void set_residual_precision_prior(
        const Ptr<GammaModelBase> &residual_precision_prior) {
      residual_precision_prior_ = residual_precision_prior;
      sigsq_sampler_.set_prior(residual_precision_prior_);
    }

    void draw() override;
    double logpri() const override;
    double log_model_prob(const Selector &inclusion_indicators) const;

    // Model selection can be turned on and off altogether, or if very large
    // sets of predictors are being considered then the number of exploration
    // steps can be limited to a specified number.
    void suppress_model_selection();

    // allow_model_selection takes an argument so its signature will match other
    // similar spike and slab sampler classes.
    void allow_model_selection(bool allow = true);

    // Restrict model selection to be no more than 'nflips' locations per sweep.
    void limit_model_selection(uint nflips);

    // For testing purposes, the draw of beta and/or sigma can be suppressed.
    // This is also useful in cases where sigma is known.
    void suppress_beta_draw();
    void suppress_sigma_draw();
    void allow_sigma_draw();
    void allow_beta_draw();

    double prior_df() const;
    double prior_ss() const;

    bool model_is_empty() const;

    void set_sigma_upper_limit(double sigma_upper_limit);

    // The smallest value that an absolute correlation between variables must
    // have before the variables can be considered for a swap move.
    void set_correlation_swap_threshold(double threshold) {
      correlation_map_.set_threshold(threshold);
    }
    
    // Sets the model parameters to their posterior mode, conditional on the
    // current include / exclude status of the regression coefficients.  Any
    // coefficient that is included will be optimized.  Any coefficient that is
    // excluded will continue to be set to zero.
    void find_posterior_mode(double epsilon = 1e-5) override;

    bool can_find_posterior_mode() const override { return true; }

    bool posterior_mode_found() const { return true; }

    // Propose a Metropolis-Hastings move in which a variable currently in the
    // model is swapped for another variable currently out of the model.
    //
    // The proposal distribution chooses a coefficient uniformly at random from
    // among the set of included coefficients.  It chooses a coefficient from
    // among the excluded coefficients with probability proportional to the
    // square of the correlation between the included and excluded variables.
    void attempt_swap();
    
   protected:
    double draw_sigsq_given_sufficient_statistics(double df, double ss) {
      return sigsq_sampler_.draw(rng(), df, ss);
    }

    // Does one MCMC draw on a specific element of a vector of inclusion
    // indicators, given all others.
    //
    // Args:
    //   inclusion_indicators: The vector of inclusion indicators to be sampled.
    //   which_var: The position (element) in inclusion_indicators that might be
    //     changed.
    //   current_logp: The current log posterior evaluated at
    //     inclusion_indicators.
    //
    // Returns:
    //   inclusion_indicators[which_var] will be sampled from its full
    //   conditional distribution.  The return value is the (unnormalized) log
    //   posterior of the current inclusion_indicators at function exit.
    double mcmc_one_flip(Selector &inclusion_indicators, uint which_var,
                         double current_logp);

   private:
    // The model whose paramaters are to be drawn.
    RegressionModel *model_;

    // A conditionally (given sigma) Gaussian prior distribution for the
    // coefficients of the full model (with all variables included).
    Ptr<MvnGivenScalarSigmaBase> slab_;

    // A marginal prior distribution for 1/sigma^2.
    Ptr<GammaModelBase> residual_precision_prior_;

    // A marginal prior for the set of 0's and 1's indicating which variables
    // are in/out of the model.
    Ptr<VariableSelectionPrior> spike_;

    std::vector<uint> indx;
    uint max_nflips_;
    bool draw_beta_;
    bool draw_sigma_;

    // The normal inverse gamma distribution for regression models is:
    //
    //     beta | sigsq ~ N(b, Omega * sigsq)
    //        1 / sigsq ~ Gamma(df / 2, ss / 2)
    //
    // Upon observing data, the parameters b, Omega, df, and ss are updated to
    // become
    //      b -> posterior_mean_
    // V^{-1} -> unscaled_posterior_precision_
    //     df -> DF_
    //     ss -> SS_
    //
    // These quantities are mutable because they are modified when computing
    // posterior model probabilities.
    mutable Vector posterior_mean_;
    mutable SpdMatrix unscaled_posterior_precision_;
    mutable double DF_, SS_;

    GenericGaussianVarianceSampler sigsq_sampler_;

    CorrelationMap correlation_map_;

    // Keeps track of the number of times within the current draw that algorithm
    // had to restart because of a degenerate information matrix.
    int failure_count_;
    
    double set_reg_post_params(const Selector &inclusion_indicators,
                               bool do_ldoi) const;

    void draw_beta();
    void draw_model_indicators();
    void draw_sigma();

    const Ptr<MvnGivenScalarSigmaBase> &check_slab_dimension(
        const Ptr<MvnGivenScalarSigmaBase> &slab);
    const Ptr<VariableSelectionPrior> &check_spike_dimension(
        const Ptr<VariableSelectionPrior> &spike);
  };
  
}  // namespace BOOM

#endif  // BOOM_BREG_VS_SAMPLER_HPP
