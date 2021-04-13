// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#ifndef BOOM_GAUSSIAN_MODEL_CONJUGATE_SAMPLER_HPP
#define BOOM_GAUSSIAN_MODEL_CONJUGATE_SAMPLER_HPP

#include "Models/GammaModel.hpp"
#include "Models/GaussianModelGivenSigma.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {
  class GaussianModel;

  class GaussianConjSampler : public PosteriorSampler {
   public:
    GaussianConjSampler(GaussianModel *m,
                        const Ptr<GaussianModelGivenSigma> &mu,
                        const Ptr<GammaModelBase> &sig,
                        RNG &seeding_rng = GlobalRng::rng);

    GaussianConjSampler *clone_to_new_host(Model *host) const override;

    void draw() override;
    double logpri() const override;

    double mu() const;
    double kappa() const;
    double df() const;
    double ss() const;

    void find_posterior_mode(double epsilon = 1e-5) override;
    bool can_find_posterior_mode() const override { return true; }

   private:
    GaussianModel *mod_;
    Ptr<GaussianModelGivenSigma> mu_;
    Ptr<GammaModelBase> siginv_;
    GenericGaussianVarianceSampler sigsq_sampler_;
  };
}  // namespace BOOM
#endif  // BOOM_GAUSSIAN_MODEL_CONJUGATE_SAMPLER_HPP
