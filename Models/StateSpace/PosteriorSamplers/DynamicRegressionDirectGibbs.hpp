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

namespace BOOM {

  // A "direct Gibbs" sampler for sparse dynamic regression models in the
  // spirirt of Nakajima and West.  This sampler could be improved (as shown in
  // Scott(2002)) using FB sampling for the inclusion indicators after
  // integrating out the coefficients.
  class DynamicRegressionDirectGibbsSampler
      : public PosteriorSampler {
   public:
    DynamicRegressionDirectGibbsSampler(
        DynamicRegressionModel *model,
        const Matrix &prior_transition_counts,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    // Draw each inclusion indicator by a direct Gibbs sampler, integrating over
    // the model coefficients, but conditioning on everything else.
    void draw_inclusion_indicators();

    // Sample the coefficients at each time point, conditional on the inclusion
    // indicators, using forward-filtering backward-sampling.
    void draw_coefficients_given_inclusion();

    // Sample the residual variance, given all else, from its full conditional.
    void draw_residual_variance();

    // Sample the state innovation variances, given all else, from their full
    // conditional distributions.
    void draw_state_innovation_variance();

    // Sample the transition probabilities for the inclusion indicators from
    // their full conditional.
    void draw_transition_probabilities();

    // The unnormalized log posterior proportional to p(gamma[t] | *), where *
    // is everything but the dynamic regression coefficients, which are assumed
    // integrated out.
    //
    // The prior distribution on beta is simpler than the general g-prior, because
    // betas, conditional in inclusion, are independent across predictors.  The
    // only aspect of the prior that needs to be evaluated is element j.
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
    Vector compute_prior_variance(const Selector &inc, int time_index) const;

   private:
    DynamicRegressionModel *model_;
    Ptr<MarkovConjSampler> transition_probability_sampler_;
  };


}  // namespace BOOM


#endif   // BOOM_DYNAMIC_REGRESSION_DIRECT_GIBBS_SAMPLER_HPP_
