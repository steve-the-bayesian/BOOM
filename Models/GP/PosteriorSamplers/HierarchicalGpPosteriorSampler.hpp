#ifndef BOOM_MODELS_GP_HIERARCHICAL_GP_POSTERIOR_SAMPLER_HPP_
#define BOOM_MODELS_GP_HIERARCHICAL_GP_POSTERIOR_SAMPLER_HPP_

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

#include "Models/GP/HierarchicalGpRegressionModel.hpp"
#include "Models/GP/PosteriorSamplers/GaussianProcessRegressionPosteriorSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  class HierarchicalGpPosteriorSampler : public PosteriorSampler {
   public:
    // The prior and all the 'data model' subcomponents of 'model' should have
    // posterior samplers assigned to them.
    explicit HierarchicalGpPosteriorSampler(
        HierarchicalGpRegressionModel *model,
        RNG &seeding_rng = GlobalRng::rng);

    double logpri() const override;
    void draw() override;

    // For each data point in the training data for 'model', simulate the
    // model's posterior function values, and adjust the data point by
    // subtracting the imputed function value from the observed y value.
    void adjust_function_values(GaussianProcessRegressionModel *model);

    // Reset all data adjustments to 0.
    void clear_data_adjustments();

   private:
    HierarchicalGpRegressionModel *model_;

  };

}  // namespace BOOM

#endif  //  BOOM_MODELS_GP_HIERARCHICAL_GP_POSTERIOR_SAMPLER_HPP_
