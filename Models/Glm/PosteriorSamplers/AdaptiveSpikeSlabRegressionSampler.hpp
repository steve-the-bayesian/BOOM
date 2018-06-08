#ifndef BOOM_GLM_ADAPTIVE_SPIKE_SLAB_REGRESSION_SAMPLER_HPP_
#define BOOM_GLM_ADAPTIVE_SPIKE_SLAB_REGRESSION_SAMPLER_HPP_
/*
  Copyright (C) 2005-2018 Steven L. Scott

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
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/GammaModel.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "LinAlg/Selector.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  // A spike and slab sampler for regression models that assumes a conjugate
  // prior.  The sampler uses the method of Benson and Friel (2017) [Adaptive
  // MCMC for multiple changepoint analysis with applications to large datasets]
  // to find variables that are likely candidates for inclusion or deletion in a
  // more effective way than random selection.
  class AdaptiveSpikeSlabRegressionSampler
      : public PosteriorSampler {
   public:

    // Args:
    //   model:  The model to sample.
    //   slab: The Gaussian prior model (given gamma and sigma) for the included
    //     regression coefficients.
    //   residual_precision_prior:  The prior distribution for 1 / sigma^2.
    //   spike:  The prior model for which coefficients are included.
    //   rng:  The random number generator used to seed the RNG for this sampler.
    AdaptiveSpikeSlabRegressionSampler(
        RegressionModel *model,
        const Ptr<MvnGivenScalarSigmaBase> &slab,
        const Ptr<GammaModelBase> &residual_precision_prior,
        const Ptr<VariableSelectionPrior> &spike,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    // By default, model selection is turned on.  To turn it off call
    // allow_model_selection(false).  This will freeze the coefficients in the
    // included/excluded state they were in at the time of the call.
    void allow_model_selection(bool allow = true) {
      allow_model_selection_ = allow;
    }

    void set_sigma_upper_limit(double sigma_upper_limit) {
      sigsq_sampler_.set_sigma_max(sigma_upper_limit);
    }

    // Set the number of attempted birth/death moves for each iteration.  The
    // default is min(100, model_->xdim());
    void limit_model_selection(int flips) {
      max_flips_ = flips;
    }

    void store_model_probs(bool store) {
      store_model_probs_ = store;
      if (!store) {
        log_model_probabilities_.clear();
      }
    }
    
    void birth_move(Selector &included_coefficients);
    void death_move(Selector &included_coefficients);
    
    double log_model_prob(const Selector &inclusion_indicators);
    double prior_df() const {
      return 2 * residual_precision_prior_->alpha();
    }
    double prior_ss() const {
      return 2 * residual_precision_prior_->beta();
    }

    void set_step_size(double step_size);
    void set_target_acceptance_rate(double rate);
    
   private:
    RegressionModel *model_;
    Ptr<MvnGivenScalarSigmaBase> slab_;
    Ptr<GammaModelBase> residual_precision_prior_;
    Ptr<VariableSelectionPrior> spike_;
    GenericGaussianVarianceSampler sigsq_sampler_;

    bool allow_model_selection_;
    
    // Compute the moments of the posterior distribution conditional on the set
    // of included coefficients.
    void set_posterior_moments(const Selector &inclusion_indicators);
    void draw_coefficients();
    void draw_residual_variance();

    // Adjust the birth rate for a variable as a result of an attempted birth
    // move.
    void adjust_birth_rate(int which_variable, double MH_alpha);

    // Adjust the death rate for a variable as a result of an attempted death
    // move.
    void adjust_death_rate(int which_variable, double MH_alpha);

    size_t max_flips_;
    
    // The adjustment to birth and death and death rates diminishes with MCMC
    // iteration.
    size_t iteration_count_;

    // The amount by which the birth or death rate would be incremented at step
    // 1.
    double step_size_;

    // Pretty much what it says.
    double target_acceptance_rate_;

    Vector birth_rates_;
    Vector death_rates_;

    // The log model probability 
    double current_log_model_prob_;
    
    // The remaining data members are parameters of the conditional posterior
    // distribution given the inclusion vector.

    // Posterior df is the prior degrees of freedom plus the sample size.
    double posterior_df_;

    // Posterior sum of squares is the prior sum of squares, plus the sum of
    // squared errors when the model is evaluated at the posterior mean of the
    // coefficients, plus the Mahalanobis distance from the prior to the
    // posterior mean of the coefficients (with respect to the unscaled prior
    // precision).
    double posterior_sum_of_squares_;

    // The posterior mean of the regression coefficients.
    Vector posterior_mean_;

    // The unscaled prior precision of the coefficients plus X'X.  'Unscaled'
    // means you need to divide by the residual variance to get the actual
    // precision.
    SpdMatrix unscaled_posterior_precision_;

    // The log determinant of the prior precision.
    double logdet_omega_inverse_;

    bool store_model_probs_;
    std::map<Selector, double> log_model_probabilities_;
  };
  
}  // namespace BOOM 

#endif  // BOOM_GLM_ADAPTIVE_SPIKE_SLAB_REGRESSION_SAMPLER_HPP_

