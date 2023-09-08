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

#include "Models/GP/PosteriorSamplers/MahalanobisKernelSampler.hpp"

namespace BOOM {

  MahalanobisKernelSampler::MahalanobisKernelSampler(
      MahalanobisKernel *kernel,
      GaussianProcessRegressionModel *model,
      const Ptr<DoubleModel> &prior)
      : kernel_(kernel),
        model_(model),
        prior_(prior),
        slice_(nullptr)
  {

    std::function<double(double)> target = [kernel, model, prior](
        double scale) {
      double ans = prior->logp(scale);
      if (!std::isfinite(ans)) {
        return ans;
      }

      double original_scale = kernel->scale();
      kernel->set_scale(scale);
      ans += model->log_likelihood();
      kernel->set_scale(original_scale);
      return ans;
    };

    slice_.reset(new ScalarSliceSampler(target));
    slice_->set_lower_limit(0.0);
  }

  double MahalanobisKernelSampler::logpri() const {
    return prior_->logp(kernel_->scale());
  }

  double MahalanobisKernelSampler::logpost() const {
    return logpri() + model_->log_likelihood();
  }

  void MahalanobisKernelSampler::draw(RNG &rng) {
    slice_->set_rng(&rng, false);
    double new_scale = slice_->draw(kernel_->scale());
    kernel_->set_scale(new_scale);
  }


}  // namespace BOOM
