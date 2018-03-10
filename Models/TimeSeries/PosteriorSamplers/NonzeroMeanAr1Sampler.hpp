// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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
#ifndef BOOM_NONZERO_MEAN_AR1_SAMPLER_HPP_
#define BOOM_NONZERO_MEAN_AR1_SAMPLER_HPP_

#include "Models/GammaModel.hpp"
#include "Models/GaussianModelBase.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/TimeSeries/NonzeroMeanAr1Model.hpp"

namespace BOOM {

  class NonzeroMeanAr1Sampler : public PosteriorSampler {
   public:
    NonzeroMeanAr1Sampler(NonzeroMeanAr1Model *model,
                          const Ptr<GaussianModelBase> &mean_prior,
                          const Ptr<GaussianModelBase> &phi_prior,
                          const Ptr<GammaModelBase> &siginv_prior,
                          RNG &seeding_rng = GlobalRng::rng);

    // Truncate the support of phi (the autoregression coefficient) to
    // (-1, 1) to ensure stationarity
    void force_stationary();

    // Truncate the suport of the autoregression coefficient to
    // disallow negative values.
    void force_ar1_positive();
    void set_sigma_upper_limit(double sigma_hi_);

    void draw() override;
    double logpri() const override;

    void draw_mu();
    void draw_phi();
    void draw_sigma();

   private:
    NonzeroMeanAr1Model *m_;
    Ptr<GaussianModelBase> mean_prior_;
    Ptr<GaussianModelBase> phi_prior_;
    Ptr<GammaModelBase> siginv_prior_;
    bool truncate_phi_;  // truncate the support of phi to (-1, 1) to
                         // ensure stationarity.
    bool force_ar1_positive_;
    GenericGaussianVarianceSampler sigsq_sampler_;
  };

}  // namespace BOOM
#endif  // BOOM_NONZERO_MEAN_AR1_SAMPLER_HPP_
