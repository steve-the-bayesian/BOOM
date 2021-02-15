#ifndef BOOM_DYNAMIC_INTERCEPT_REGRESSION_POSTERIOR_SAMPLER_HPP_
#define BOOM_DYNAMIC_INTERCEPT_REGRESSION_POSTERIOR_SAMPLER_HPP_
/*
  Copyright (C) 2019 Steven L. Scott

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
#include "Models/StateSpace/DynamicInterceptRegression.hpp"

namespace BOOM {

  class DynamicInterceptRegressionPosteriorSampler
      : public PosteriorSampler {
   public:
    explicit DynamicInterceptRegressionPosteriorSampler(
        DynamicInterceptRegressionModel *model,
        RNG &seeding_rng = GlobalRng::rng)
        : PosteriorSampler(seeding_rng),
          model_(model),
          latent_data_initialized_(false)
    {}

    void draw() override {
      if (!latent_data_initialized_) {
        model_->impute_state(rng());
        latent_data_initialized_ = true;
        impute_nonstate_latent_data();
      }
      model_->observation_model()->sample_posterior();
      for (int s = 0; s < model_->number_of_state_models(); ++s) {
        model_->state_model(s)->sample_posterior();
      }
      impute_nonstate_latent_data();
      model_->impute_state(rng());
    }

    double logpri() const override {
      double ans = model_->observation_model()->logpri();
      for (int s = 0; s < model_->number_of_state_models(); ++s) {
        ans += model_->state_model(s)->logpri();
      }
      return ans;
    }

    // For Gaussian models this is a no-op.
    virtual void impute_nonstate_latent_data() {}

   private:
    DynamicInterceptRegressionModel *model_;
    bool latent_data_initialized_;
  };

}  // namespace BOOM

#endif   //  BOOM_DYNAMIC_INTERCEPT_REGRESSION_POSTERIOR_SAMPLER_HPP_
