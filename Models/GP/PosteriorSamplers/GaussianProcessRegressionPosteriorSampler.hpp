#ifndef BOOM_MODELS_GP_POSTERIOR_SAMPLERS_GAUSSIAN_PROCESS_POSTERIOR_SAMPLER_HPP
#define BOOM_MODELS_GP_POSTERIOR_SAMPLERS_GAUSSIAN_PROCESS_POSTERIOR_SAMPLER_HPP
/*
  Copyright (C) 2005-2023 Steven L. Scott

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

#include "Models/GP/GaussianProcessRegressionModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"

namespace BOOM {

  namespace GP {
    class ParameterSampler
        : private RefCounted {
     public:
      virtual void draw(RNG &rng) = 0;
      virtual double logpri() const = 0;

      friend void intrusive_ptr_add_ref(ParameterSampler *sam) {
        sam->up_count();
      }
      friend void intrusive_ptr_release(ParameterSampler *sam) {
        sam->down_count();
        if (sam->ref_count() == 0) {
          delete sam;
        }
      }
    };

    // For use with mean/kernel functions that contain no parameters.
    class NullSampler : public ParameterSampler {
     public:
      void draw(RNG &) override {}
      double logpri() const override {return 0.0;}
    };
  }  // namespace GP


  class GaussianProcessRegressionPosteriorSampler
      : public PosteriorSampler {
   public:
    GaussianProcessRegressionPosteriorSampler(
        GaussianProcessRegressionModel *model,
        const Ptr<GP::ParameterSampler> &mean_function_sampler,
        const Ptr<GP::ParameterSampler> &kernel_sampler,
        const Ptr<GammaModelBase> &residual_variance_prior,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    void draw_residual_variance();
    void set_sigma_max(double sigma_max) {
      residual_variance_sampler_.set_sigma_max(sigma_max);
    }

   private:
    GaussianProcessRegressionModel *model_;
    Ptr<GP::ParameterSampler> mean_function_sampler_;
    Ptr<GP::ParameterSampler> kernel_sampler_;

    GenericGaussianVarianceSampler residual_variance_sampler_;
  };

}  // namespace BOOM

#endif // BOOM_MODELS_GP_POSTERIOR_SAMPLERS_GAUSSIAN_PROCESS_POSTERIOR_SAMPLER
