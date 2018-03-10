// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#ifndef BOOM_STUDENT_LOCAL_LINEAR_TREND_POSTERIOR_SAMPLER_HPP_
#define BOOM_STUDENT_LOCAL_LINEAR_TREND_POSTERIOR_SAMPLER_HPP_

#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/StateSpace/StateModels/StudentLocalLinearTrend.hpp"

namespace BOOM {

  class StudentLocalLinearTrendPosteriorSampler : public PosteriorSampler {
   public:
    StudentLocalLinearTrendPosteriorSampler(
        StudentLocalLinearTrendStateModel *model,
        const Ptr<GammaModelBase> &sigsq_level_prior,
        const Ptr<DoubleModel> &nu_level_prior,
        const Ptr<GammaModelBase> &sigsq_slope_prior,
        const Ptr<DoubleModel> &nu_slope_prior,
        RNG &seeding_rng = GlobalRng::rng);

    double logpri() const override;
    void draw() override;

    void set_sigma_level_upper_limit(double upper_limit);
    void set_sigma_slope_upper_limit(double upper_limit);

    void draw_sigsq_level();
    void draw_nu_level();
    void draw_sigsq_slope();
    void draw_nu_slope();

   private:
    StudentLocalLinearTrendStateModel *model_;

    Ptr<GammaModelBase> sigsq_level_prior_;
    Ptr<DoubleModel> nu_level_prior_;
    Ptr<GammaModelBase> sigsq_slope_prior_;
    Ptr<DoubleModel> nu_slope_prior_;

    GenericGaussianVarianceSampler sigsq_level_sampler_;
    GenericGaussianVarianceSampler sigsq_slope_sampler_;
  };

}  // namespace BOOM

#endif  //  BOOM_STUDENT_LOCAL_LINEAR_TREND_POSTERIOR_SAMPLER_HPP_
