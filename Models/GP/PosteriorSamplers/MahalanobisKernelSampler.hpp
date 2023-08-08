#ifndef BOOM_GP_MAHALANOBIS_KERNEL_SAMPLER_HPP_
#define BOOM_GP_MAHALANOBIS_KERNEL_SAMPLER_HPP_

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
#include "Samplers/ScalarSliceSampler.hpp"

namespace BOOM {

  class MahalanobisKernelSampler : GP::ParameterSampler {
   public:
    MahalanobisKernelSampler(
        MahalanobisKernel *kernel,
        GaussianProcessRegressionModel *model,
        const Ptr<DoubleModel> &prior);

    void draw(RNG &rng) override;

   private:
    MahalanobisKernel *kernel_;
    Ptr<ScalarSliceSampler> slice_;
  };

}  // namespace BOOM


#endif  //  BOOM_GP_MAHALANOBIS_KERNEL_SAMPLER_HPP_
