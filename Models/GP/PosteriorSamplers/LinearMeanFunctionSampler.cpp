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

#include "Models/GP/PosteriorSamplers/LinearMeanFunctionSampler.hpp"
#include "distributions.hpp"
#include "Models/Glm/RegressionModel.hpp"

namespace BOOM {

  LinearMeanFunctionSampler::LinearMeanFunctionSampler(
      LinearMeanFunction *mean_function,
      GaussianProcessRegressionModel *model,
      const Ptr<MvnBase> &prior)
      : mean_function_(mean_function),
        model_(model),
        prior_(prior)
  {}

  double LinearMeanFunctionSampler::logpri() const {
    return prior_->logp(mean_function_->coef()->Beta());
  }

  // The model is Y ~ N(mu(X), K(X)) where mu(X) = beta'X.
  void LinearMeanFunctionSampler::draw(RNG &rng) {
    // beta | Y ~ N(B, V)
    //
    // (Y - XB)' Kinv (Y - XB)
    // B'X' Kinv X B - 2 B'X'Kinv Y

    int sample_size = model_->dat().size();
    int dim = model_->xdim();
    Matrix X(sample_size, dim);
    Vector y(sample_size);
    for (int i = 0; i < sample_size; ++i) {
      const Ptr<RegressionData> &data_point(model_->dat()[i]);
      X.row(i) = data_point->x();
      y[i] = data_point->y();
    }

    const SpdMatrix &Kinv(model_->inverse_kernel_matrix());

    SpdMatrix posterior_precision =
        prior_->precision() + sandwich_transpose(X, Kinv);

    Vector unscaled_posterior_mean =
        prior_->precision() * prior_->mean() + X.Tmult(Kinv * y);
    Vector posterior_mean = posterior_precision.solve(unscaled_posterior_mean);
    Vector beta = rmvn_ivar_mt(rng, posterior_mean, posterior_precision);
    mean_function_->coef()->set_Beta(beta);
  }

}  // namespace BOOM
