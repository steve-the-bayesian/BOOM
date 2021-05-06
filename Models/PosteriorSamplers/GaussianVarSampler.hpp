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
#ifndef BOOM_GAUSSIAN_VARIANCE_METHOD_HPP
#define BOOM_GAUSSIAN_VARIANCE_METHOD_HPP

#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {
  class GaussianModel;
  class GammaModelBase;

  // draws sigma given mu
  class GaussianVarSampler : public PosteriorSampler {
   public:
    GaussianVarSampler(GaussianModel *model, double prior_df,
                       double prior_sigma_guess,
                       RNG &seeding_rng = GlobalRng::rng);
    GaussianVarSampler(GaussianModel *model,
                       const Ptr<GammaModelBase> &precision_prior,
                       RNG &seeding_rng = GlobalRng::rng);

    GaussianVarSampler *clone_to_new_host(Model *new_host) const override;

    void draw() override;
    double logpri() const override;
    // Call to ensure that sigma (standard deviation) remains below
    // the specified upper_truncation_point
    void set_sigma_upper_limit(double max_sigma);

    // Sets mod->sigsq() to the posterior mode given the prior and mod->suf();
    void find_posterior_mode(double epsilon = 1e-5) override;
    bool can_find_posterior_mode() const override { return true; }

   protected:
    const Ptr<GammaModelBase> ivar() const;

   private:
    Ptr<GammaModelBase> prior_;
    GaussianModel *model_;
    GenericGaussianVarianceSampler sampler_;
  };

}  // namespace BOOM
#endif  // BOOM_GAUSSIAN_VARIANCE_METHOD_HPP
