#ifndef BOOM_NAKAJIMA_WEST_SAMPLER_HPP_
#define BOOM_NAKAJIMA_WEST_SAMPLER_HPP_
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

#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/StateSpace/DynamicRegression.hpp"

namespace BOOM {

  // A "direct Gibbs" sampler for sparse dynamic regression models in the
  // spirirt of Nakajima and West.  This sampler could be improved (as shown in
  // Scott(2002)) using FB sampling for the inclusion indicators after
  // integrating out the coefficients.
  class DynamicRegressionDirectGibbsSampler
      : public PosteriorSampler {
   public:
    DynamicRegressionDirectGibbsSampler(
        DynamicRegressionModel *model,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    void logpri() const override;

    void draw_inclusion_indicators();
    void draw_coefficients_given_inclusion();
    void draw_residual_variance();
    void draw_state_innovation_variance();
    void draw_transition_probabilities();

    void log_model_prob(const Selector &inc, int t) const;

   private:
    DynamicRegressionModel *model_;
  };


}  // namespace BOOM


#endif   // BOOM_NAKAJIMA_WEST_SAMPLER_HPP_
