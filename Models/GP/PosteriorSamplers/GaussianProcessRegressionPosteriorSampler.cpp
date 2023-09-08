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
#include "distributions.hpp"
#include "cpputil/report_error.hpp"

#include <sstream>

namespace BOOM {

  namespace {
    using GPRPS = GaussianProcessRegressionPosteriorSampler;
  }  // namespace

  GPRPS::GaussianProcessRegressionPosteriorSampler(
      GaussianProcessRegressionModel *model,
      const Ptr<GP::ParameterSampler> &mean_function_sampler,
      const Ptr<GP::ParameterSampler> &kernel_sampler,
      const Ptr<GammaModelBase> &residual_variance_prior,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        mean_function_sampler_(mean_function_sampler),
        kernel_sampler_(kernel_sampler),
        residual_variance_sampler_(residual_variance_prior)
  {}

  double GPRPS::logpri() const {
    return mean_function_sampler_->logpri()
        + kernel_sampler_->logpri()
        + residual_variance_sampler_.log_prior(model_->sigsq());
  }

  void GPRPS::draw() {
    draw_residual_variance();
    kernel_sampler_->draw(rng());
    mean_function_sampler_->draw(rng());
  }

  void GPRPS::draw_residual_variance() {
    double data_sum_of_squares = 0;
    size_t sample_size = model_->dat().size();

    Vector posterior_residuals = model_->posterior_residuals();
    for (double resid : posterior_residuals) {
      data_sum_of_squares += square(resid);
    }

    double sigsq = residual_variance_sampler_.draw(
        rng(), sample_size, data_sum_of_squares);
    model_->set_sigsq(sigsq);
  }

}  // namespace BOOM
