#ifndef BOOM_DYNAMIC_REGRESSION_DIRECT_GIBBS_SAMPLER_HPP_
#define BOOM_DYNAMIC_REGRESSION_DIRECT_GIBBS_SAMPLER_HPP_
/*
  Copyright (C) 2005-2020 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/StateSpace/DynamicRegression.hpp"
#include "Models/PosteriorSamplers/MarkovConjSampler.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"

namespace BOOM {

  // A "direct Gibbs" sampler for sparse dynamic regression models in the
  // spirirt of Nakajima and West.  This sampler could be improved (as shown in
  // Scott(2002)) using FB sampling for the inclusion indicators after
  // integrating out the coefficients.
  //
  // The dynamic regression model is y[t] = X[t] * beta[t] + epsilon[t], with
  // X[t] fully observed, epsilon[t] ~ MVN(0, sigma^2 I), independent across t,
  // where the dimension of I is n[t] X n[t], and with state dynamics on beta as
  // follows.
  //
  // Deonte the scalar elements of beta[t] as beta[j, t], and let gamma[j, t] =
  // 1 if beta[j, t] is nonzero and gamma[j, t] = 0 if beta[j, t] == 0.  The
  // model assumes gamma[j, t] evolves independently across j with gamma[j, t] ~
  // Bernoulli(pi_j[gamma[j, t-1]]).  That is, gamma[j, t] evolves according to
  // a 2-state discrete time Markov chain.
  //
  // The prior parameters of the Markov chain are supplied by the user in the
  // form of three quantities:
  //
  //   - An expected duration (number of time periods) of a visit to state 1.
  //   - The stationary probability of being in state 1.
  //   - A sample size.
  //
  // When these three numbers are passed through a function infer_Markov_prior
  // they are converted into a 2x2 matrix of positive counts for a product
  // Dirichlet prior.
  class DynamicRegressionDirectGibbsSampler
      : public PosteriorSampler {
   public:

    // Args:
    //   model:  The model whose unknowns are to be sampled.
    //   residual_sd_prior_guess: A prior estimate at the residual standard
    //     deviation.
    //   residual_sd_prior_sample_size: A prior sample size.  The number of
    //     observations worth of weight given to 'residual_sd_prior_guess'.
    //   innovation_sd_prior_guess: A prior guess at the typical amount by which
    //     an active coefficient's value might change in a typical time period.
    //     This is a Vector with one entry per coefficient.
    //   innovation_sd_prior_sample_size: A prior sample size associated with
    //     innovation_sd_prior_guess.  It is the number of observations worth of
    //     weight to place on innovation_sd_prior_guess.
    //   prior_inclusion_probabilities: A Vector of probabilities, one per each
    //     predictor, describing a prior guess at the stationary probability of
    //     that predictor being included.
    //   expected_inclusion_duration: A Vector with one element per predictor
    //     representing the expected durations (number of time periods).  This
    //     is a guess at how many time periods an "inclusion event" would
    //     typically last.
    //   transition_probability_prior_sample_size: The prior inclusion
    //     probability and expected inclusion duration combine to give an
    //     estimated transition probability matrix.  This argument is the number
    //     of observations worth of weight to assign to that matrix.  This is a
    //     Vector with one element per predictor.
    //   seeding_rng:  The RNG used to set the seed for the RNG in this sampler.
    DynamicRegressionDirectGibbsSampler(
        DynamicRegressionModel *model,
        double residual_sd_prior_guess,
        double residual_sd_prior_sample_size,
        const Vector &innovation_sd_prior_guess,
        const Vector &innovation_sd_prior_sample_size,
        const Vector &prior_inclusion_probabilities,
        const Vector &expected_inclusion_duration,
        const Vector &transition_probability_prior_sample_size,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    // Draw each inclusion indicator by a direct Gibbs sampler, integrating over
    // the model coefficients, but conditioning on everything else.
    void draw_inclusion_indicators();

    // Sample the residual variance, given all else, from its full conditional.
    void draw_residual_variance();

    // Sample the state innovation variances, given all else, from their full
    // conditional distributions.
    void draw_unscaled_state_innovation_variance();

    // Sample the transition probabilities for the inclusion indicators from
    // their full conditional.
    void draw_transition_probabilities();

    // The unnormalized log posterior proportional to p(gamma[t] | *), where *
    // is everything but the dynamic regression coefficients, which are assumed
    // integrated out.
    //
    // The prior distribution on beta is simpler than the general g-prior,
    // because betas, conditional in inclusion, are independent across
    // predictors.  The only aspect of the prior that needs to be evaluated is
    // element j.
    double log_model_prob(const Selector &inclusion_indicators,
                          int time_index,
                          int predictor_index) const;

    // The log of the prior inclusion probability at a given time/predictor
    // index, conditional on neighboring values.
    double log_inclusion_prior(const Selector &inclusion_indicators,
                               int time_index,
                               int predictor_index) const;

    // A single MCMC draw of the inclusion indicator at a given time index and
    // predictor index.  This draw integrates out the regression coefficients,
    // but conditions on everything else.
    void mcmc_one_flip(Selector &inclusion_indicators,
                       int time_index,
                       int predictor_index);

    // The diagonal of the prior variance matrix of the regression coefficients
    // at time t, conditional on inclusion indicators and hyperparameters.  Only
    // the diagonal is returned because the variables are independent.
    //
    // This function returns the _unscaled_ variance.  The actual variance is
    // the unscaled variance times the residual variance (sigsq).
    Vector compute_unscaled_prior_variance(
        const Selector &inc, int time_index) const;

    // The transition probability matrix of a 2-state Markov chain can be
    // deduced from the stationary distribution and the expected duration of a
    // visit to state 1.  The transition probability matrix times a prior sample
    // size gives the "prior counts" parameter of a Markov conjugate prior.
    //
    // Args:
    //   prior_success_prob:  The stationary probability of being in state 1.
    //   expected_time: The expected length of a visit to state 1.  This must be
    //     >= 1.
    //   sample_size: The number of observations worth of weight to assign the
    //     prior guess.
    //
    // Returns:
    //   A matrix of prior counts.
    static Matrix infer_Markov_prior(double prior_success_prob,
                                     double expected_time,
                                     double sample_size);

   private:
    DynamicRegressionModel *model_;

    Ptr<ChisqModel> residual_precision_prior_;
    GenericGaussianVarianceSampler residual_variance_sampler_;

  };


}  // namespace BOOM


#endif   // BOOM_DYNAMIC_REGRESSION_DIRECT_GIBBS_SAMPLER_HPP_
