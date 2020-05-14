#ifndef BOOM_LOGLINEAR_MODEL_BIPF_SAMPLER_HPP_
#define BOOM_LOGLINEAR_MODEL_BIPF_SAMPLER_HPP_

/*
  Copyright (C) 2005-2020 Steven L. Scott

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

#include "Models/Glm/LoglinearModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  // Simulate the parameters of a log linear model using the "Bayesian iterative
  // proportional fitting" algorithm described in Gelman (BDA) and Schafer
  // (1999, incomplete data).
  class LoglinearModelBipfSampler : public PosteriorSampler {
   public:
    // Add a spot for the prior.
    explicit LoglinearModelBipfSampler(LoglinearModel *model,
                                       double prior_count = 1.0,
                                       double min_scale = 1e-10,
                                       RNG &seeding_rng = GlobalRng::rng);

    void draw() override;

    double logpri() const override;

    void draw_effect_parameters(int effect_index);
    void draw_intercept();

   private:
    LoglinearModel *model_;
    double prior_count_;
    double min_scale_;
  };

}  // namespace BOOM

#endif  // BOOM_LOGLINEAR_MODEL_BIPF_SAMPLER_HPP_
