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

#ifndef BOOM_WEEKLY_CYCLE_POISSON_PROCESS_POSTERIOR_SAMPLER_HPP_
#define BOOM_WEEKLY_CYCLE_POISSON_PROCESS_POSTERIOR_SAMPLER_HPP_

#include "Models/DirichletModel.hpp"
#include "Models/GammaModel.hpp"
#include "Models/PointProcess/WeeklyCyclePoissonProcess.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  class WeeklyCyclePoissonProcessSampler : public PosteriorSampler {
   public:
    WeeklyCyclePoissonProcessSampler(
        WeeklyCyclePoissonProcess *model,
        const Ptr<GammaModelBase> &average_daily_rate_prior,
        const Ptr<DirichletModel> &day_of_week_prior,
        const Ptr<DirichletModel> &weekday_hourly_prior,
        const Ptr<DirichletModel> &weekend_hourly_prior,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    void draw_average_daily_rate();
    void draw_daily_pattern();
    void draw_weekday_hourly_pattern();
    void draw_weekend_hourly_pattern();

    // The acceptance rates for MH proposals
    double daily_pattern_accept_rate();
    double weekday_hourly_accept_rate();
    double weekend_hourly_accept_rate();

   private:
    WeeklyCyclePoissonProcess *model_;
    Ptr<GammaModelBase> average_daily_rate_prior_;
    Ptr<DirichletModel> day_of_week_prior_;
    Ptr<DirichletModel> weekday_hourly_prior_;
    Ptr<DirichletModel> weekend_hourly_prior_;

    int daily_pattern_attempts_;
    int daily_pattern_successes_;

    int weekday_hourly_attempts_;
    int weekday_hourly_successes_;

    int weekend_hourly_attempts_;
    int weekend_hourly_successes_;
  };

}  // namespace BOOM

#endif  // BOOM_WEEKLY_CYCLE_POISSON_PROCESS_POSTERIOR_SAMPLER_HPP_
