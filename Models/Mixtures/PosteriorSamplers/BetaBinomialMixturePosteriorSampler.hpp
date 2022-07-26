#ifndef BOOM_MIXTURES_BETA_BINOMIAL_MIXTURE_POSTERIOR_SAMPLER_HPP_
#define BOOM_MIXTURES_BETA_BINOMIAL_MIXTURE_POSTERIOR_SAMPLER_HPP_

/*
  Copyright (C) 2005-2022 Steven L. Scott

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
#include "Models/Mixtures/BetaBinomialMixture.hpp"
#include "Models/DoubleModel.hpp"
#include "Models/BetaModel.hpp"
#include "Models/DirichletModel.hpp"

#include "Samplers/UnivariateSliceSampler.hpp"

namespace BOOM {

  // A PosteriorSampler for the BetaBinomialMixtureModel.  The sampling
  // algorithm uses data augmentation to impute the missing mixture indicators.
  class BetaBinomialMixturePosteriorSampler
      : public PosteriorSampler {
   public:
    // Args:
    //   model: The model to be managed.  The mixing distribution and the
    //     mixture components managed by the model must be assigned their own
    //     PosteriorSampler's.
    //   seeding_rng:  The random number generator used to seed this object.
    explicit BetaBinomialMixturePosteriorSampler(
        BetaBinomialMixtureModel *model,
        RNG &seeding_rng = GlobalRng::rng);

    double logpri() const override;
    void draw() override;

   protected:
    BetaBinomialMixtureModel *model() {return model_;}

   private:
    BetaBinomialMixtureModel *model_;
  };

  //===========================================================================
  // A posterior sampler for the BetaBinomialMixtureModel that implements the
  // 'draw' method without imputing the mixture indicators.  This should have
  // better mixing behavior than the traditional data augmentation approach.
  class BetaBinomialMixtureDirectPosteriorSampler
      : public PosteriorSampler {
   public:
    // Args:
    //   model: The model to be managed.  Any PosteriorSamplers assigned to the
    //     mixture components or mixing distribution in 'model' are ignored by
    //     this object.
    //
    BetaBinomialMixtureDirectPosteriorSampler(
        BetaBinomialMixtureModel *model,
        const Ptr<DirichletModel> &mixing_weight_prior,
        const std::vector<Ptr<BetaModel>> &component_mean_priors,
        const std::vector<Ptr<DoubleModel>> &sample_size_priors,
        RNG &seeding_rng = GlobalRng::rng);

    double logpri() const override;
    void draw() override;

    // Package model parameters in the form and scale expected by the slice sampler.
    Vector pack_theta() const;

    // Extract model parameters from the form used by the slice sampler.
    //
    // Args:
    //   theta:  The packed, transformed model parameters.
    //   mixing_weights, component_means, sample_sizes: These are outputs that
    //     will be resized and filled with the parameter values taken from theta.
    void unpack_theta(const Vector &theta, Vector &mixing_weights,
                      Vector &component_means, Vector &sample_sizes) const;

   private:
    // Theta contains the packed, transformed collection of model parameters.
    // If there are K mixture components, the first K-1 elements of theta are
    // the multinomial logit transform of the mixing weights, relative to
    // component zero (which is omitted).  The next 2K elements are the
    // parameters of the K mixing components, alternating between the logit of
    // the component's mean and the log of the component's sample size
    // parameter.
    double log_posterior(const Vector &theta) const;


    BetaBinomialMixtureModel *model_;
    Ptr<DirichletModel> mixing_weight_prior_;
    std::vector<Ptr<BetaModel>> component_mean_priors_;
    std::vector<Ptr<DoubleModel>> sample_size_priors_;

    UnivariateSliceSampler sampler_;
  };

}  // namespace BOOM



#endif  //  BOOM_MIXTURES_BETA_BINOMIAL_MIXTURE_POSTERIOR_SAMPLER_HPP_
