// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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

#ifndef BOOM_ZERO_MEAN_MVN_CONJ_SAMPLER_HPP_
#define BOOM_ZERO_MEAN_MVN_CONJ_SAMPLER_HPP_
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/WishartModel.hpp"

namespace BOOM {
  class ZeroMeanMvnModel;
  class ZeroMeanMvnConjSampler : public PosteriorSampler {
   public:
    ZeroMeanMvnConjSampler(ZeroMeanMvnModel *m,
                           const Ptr<WishartModel> &prior,
                           RNG &seeding_rng = GlobalRng::rng);

    // creates a WishartModel with nu = prior_df and a diagonal scale
    // matrix with prior_df * sigma_guess^2
    ZeroMeanMvnConjSampler(ZeroMeanMvnModel *m,
                           double prior_df,
                           double sigma_guess,
                           RNG &seeding_rng = GlobalRng::rng);
    ZeroMeanMvnConjSampler *clone_to_new_host(
        Model *new_host) const override;

    void draw() override;
    double logpri() const override;
    void find_posterior_mode(double epsilon = 1e-5) override;

   private:
    ZeroMeanMvnModel *m_;
    Ptr<WishartModel> siginv_prior_;
  };
}  // namespace BOOM

#endif  // BOOM_ZERO_MEAN_MVN_CONJ_SAMPLER_HPP_
