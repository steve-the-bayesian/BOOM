#ifndef BOOM_MODELS_GP_LINEAR_MEAN_FUNCTION_POSTERIOR_SAMPLER_HPP_
#define BOOM_MODELS_GP_LINEAR_MEAN_FUNCTION_POSTERIOR_SAMPLER_HPP_

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

#include "Models/GP/PosteriorSamplers/GaussianProcessRegressionPosteriorSampler.hpp"
#include "Models/MvnBase.hpp"

namespace BOOM {

  // TODO:  Do we need to draw the residual variance parameter?
  class LinearMeanFunctionSampler : public GP::ParameterSampler {
   public:

    // Args:
    //   mean_function:  The mean function to be sampled.
    //   model: The model that owns the mean function.
    //   prior: The prior distribution over the coefficients of the mean
    //     function.
    LinearMeanFunctionSampler(LinearMeanFunction *mean_function,
                              GaussianProcessRegressionModel *model,
                              const Ptr<MvnBase> &prior);
    double logpri() const override;
    void draw(RNG &rng) override;

   private:
    LinearMeanFunction *mean_function_;
    GaussianProcessRegressionModel *model_;
    Ptr<MvnBase> prior_;
  };

}  // namespace BOOM

#endif  //  BOOM_MODELS_GP_LINEAR_MEAN_FUNCTION_POSTERIOR_SAMPLER_HPP_
