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

#include "Models/Glm/IndependentRegressionModels.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  // A posterior sampler for IndependentRegressionModels.  All work is deferred
  // to the posterior samplers assigned to the subordinate models.
  class IndependentRegressionModelsPosteriorSampler
      : public PosteriorSampler {
   public:
    // Args:
    //   model: The model to be managed.  Each sub-regression in model must have
    //     its own posterior samplers assigned.
    //   seeding_rng: The random number generator used to seed the RNG owned by
    //     this posterior sampler.
    explicit IndependentRegressionModelsPosteriorSampler(
        IndependentRegressionModels *model,
        RNG &seeding_rng = GlobalRng::rng)
        : PosteriorSampler(seeding_rng),
          model_(model)
    {}

    void draw() override {
      for (int i = 0; i < model_->ydim(); ++i) {
        model_->model(i)->sample_posterior();
      }
    }

    double logpri() const override {
      double ans = 0;
      for (int i = 0; i < model_->ydim(); ++i) {
        ans += model_->model(i)->logpri();
      }
      return ans;
    }

   private:
    IndependentRegressionModels *model_;
  };


}  // namespace BOOM
