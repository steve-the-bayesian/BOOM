// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#ifndef BOOM_DRAW_GAUSSIAN_MEAN_HPP
#define BOOM_DRAW_GAUSSIAN_MEAN_HPP
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "cpputil/Ptr.hpp"

namespace BOOM {

  class GaussianModel;
  class UnivParams;

  class GaussianMeanSampler : public PosteriorSampler {
   public:
    // mu ~ N(mu_bar, tausq), independent of sigma^2
    GaussianMeanSampler(GaussianModel *model,
                        double expected_mu,
                        double prior_sd_mu,
                        RNG &seeding_rng = GlobalRng::rng);
    GaussianMeanSampler(GaussianModel *model,
                        const Ptr<GaussianModel> &Pri,
                        RNG &seeding_rng = GlobalRng::rng);

    GaussianMeanSampler *clone_to_new_host(Model *new_host) const override;

    double logpri() const override;
    void draw() override;

   private:
    GaussianModel *model_;
    Ptr<GaussianModel> prior_;
  };
}  // namespace BOOM
#endif  // BOOM_DRAW_GAUSSIAN_MEAN_HPP
