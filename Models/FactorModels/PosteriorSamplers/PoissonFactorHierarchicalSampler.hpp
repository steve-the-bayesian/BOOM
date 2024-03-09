#ifndef BOOM_POISSON_FACTOR_HIERARCHICAL_SAMPLER_HPP_
#define BOOM_POISSON_FACTOR_HIERARCHICAL_SAMPLER_HPP_

/*
  Copyright (C) 2005-2024 Steven L. Scott

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

#include "Models/FactorModels/PosteriorSamplers/PoissonFactorPosteriorSamplerBase.hpp"
#include "Models/FactorModels/PoissonFactorModel.hpp"
#include "Models/MvnModel.hpp"

namespace BOOM {

  // Posterior sampler for a PoissonFactorModel based on a hierarchical prior
  // that borrows strength across sites for estimating a site's visitation
  // profile.
  //
  // The PoissonFactorModel describes a collection of visitors as belonging to
  // one of K latent categories.  Those visitors visit sites according to a
  // Poisson process with category dependent rates lambda[j, 0], ...,
  // lambda[j, K-1] (for site j).
  //
  // This sampler views a site's "lambda" (category-specific intensity)
  // parameters in terms of the total visitation rate alpha = sum(lambda) and
  // a visitation profile pi = lambda / alpha.  In other words, pi is a
  // discrete probability distribution.  This sampler assumes a flat prior on
  // alpha, but assumes MultinomialLogit(pi[j]) ~ N(mu, Sigma).  The reference
  // category for the multinomial logit transformation is pi[0].
  //
  // The prior parameters then are the parameters of a normal-invserse-Wishart
  // prior on mu and Sigma.
  //
  // - Sigma_guess is a guess at the variance matrix describing variation in
  //     the multinomial logits across sites.  The scale of the multinomial
  //     logit tranformation is such that 1 unit is a fairly large shift, so a
  //     prior guess at Sigma with unit diagonal emphasizes heterogeneity.
  //
  // - prior_df: A scalar "prior sample size" indicating how much weight
  //     should be assigned to Sigma_guess.  The relevant sample size here is
  //     the number of sites in the PoissonFactorModel.  prior_df should be
  //     larger than the dimension of Sigma.
  //
  // - prior_mean: The prior guess at the multinomial logit values.  Absent
  //     strong prior knowledge about the rates at which various groups
  //     generally visit sites, a vector of all 0's is an appropriate central
  //     value.
  //
  // - kappa: A prior sample size indicating how many observations worth of
  //     weight should be given to mu. The relevant sample size for comparison
  //     is the number of sites in the PoissonFactorModel.
  class PoissonFactorHierarchicalSampler
      : public PoissonFactorPosteriorSamplerBase {
   public:
    // Args:
    //   model:  The model to be posterior sampled.
    //   default_prior_class_probabilities: Default prior probability
    //     distribution for class membership.  This prior can be replaced for
    //     specific individuals.  See PoissonFactorPosteriorSamplerBase.
    //   prior_mean, kappa, Sigma_guess, prior_df: See above.
    //   MH_threshold: Sites with at least this many observations in each
    //     category will be drawn by MH sampling.  Sites that do not satisfy the
    //     threshold will update using slice sampling.
    //   seeding_rng: The random number generator used to seed this sampler's
    //     RNG.
    PoissonFactorHierarchicalSampler(
        PoissonFactorModel *model,
        const Vector &default_prior_class_probabilities,
        const Vector &prior_mean,
        double kappa,
        const SpdMatrix &Sigma_guess,
        double prior_df,
        int MH_threshold = 10,
        RNG &seeding_rng = GlobalRng::rng);

    double logpri() const override;
    void draw() override;

    // The lambda site parameters are decomponsed into lambda = alpha * pi,
    // where alpha is a scalar and pi is a discrete probability distribution.
    void draw_site_parameters();
    void draw_hyperparameters();

    // Different implementations for draw_site_parameters depending on how much
    // data was observed.
    void draw_site_parameters_MH(Ptr<FactorModels::PoissonSite> &site);
    void draw_site_parameters_slice(Ptr<FactorModels::PoissonSite> &site);

    const MvnModel *hyperprior() const {
      return profile_hyperprior_.get();
    }

    // Sites with at least 'threshold' observations in each category will be
    // sampled using MH sampling.
    void set_MH_threshold(int threshold);

    // Return a report indicating how many moves of each type were tried, and
    // how many MH attempts succeeded.
    std::string sampling_report() const;
    
   private:
    void check_dimension(const Ptr<MvnModel> &profile_hyperprior) const;

    PoissonFactorModel *model_;

    // Prior distribution for mulinomial logits of intensity parameters, with
    // category zero as the reference class.
    Ptr<MvnModel> profile_hyperprior_;

    // Posterior sampler for profile_hyperprior_.
    Ptr<MvnConjSampler> hyperprior_sampler_;

    // The number of accepted MH proposals.
    Int MH_acceptance_;

    // The number of rejected MH proposals.
    Int MH_failure_;

    // The number of slice sampling draws that have been attempted.
    Int slice_sample_draws_;
    
    // The min number of observations in each category needed to use MH sampling
    // instead of slice sampling.
    int MH_threshold_;
  };

}  // namespace BOOM


#endif  //  BOOM_POISSON_FACTOR_HIERARCHICAL_SAMPLER_HPP_
