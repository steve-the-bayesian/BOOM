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

#include "Models/PointProcess/PosteriorSamplers/WeeklyCyclePoissonProcessSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef WeeklyCyclePoissonProcessSampler SAM;
    typedef WeeklyCyclePoissonProcess WP;
  }  // namespace

  SAM::WeeklyCyclePoissonProcessSampler(
      WeeklyCyclePoissonProcess *model,
      const Ptr<GammaModelBase> &average_daily_rate_prior,
      const Ptr<DirichletModel> &day_of_week_prior,
      const Ptr<DirichletModel> &weekday_hourly_prior,
      const Ptr<DirichletModel> &weekend_hourly_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        average_daily_rate_prior_(average_daily_rate_prior),
        day_of_week_prior_(day_of_week_prior),
        weekday_hourly_prior_(weekday_hourly_prior),
        weekend_hourly_prior_(weekend_hourly_prior) {}

  void SAM::draw() {
    draw_average_daily_rate();
    draw_daily_pattern();
    draw_weekday_hourly_pattern();
    draw_weekend_hourly_pattern();
  }

  double SAM::logpri() const {
    double ans = average_daily_rate_prior_->logp(model_->average_daily_rate());
    ans += day_of_week_prior_->logp(model_->day_of_week_pattern());
    ans += weekday_hourly_prior_->logp(model_->weekday_hourly_pattern());
    ans += weekend_hourly_prior_->logp(model_->weekend_hourly_pattern());
    return ans;
  }

  void SAM::draw_average_daily_rate() {
    double a = sum(model_->suf()->count()) + average_daily_rate_prior_->alpha();
    double b = average_daily_rate_prior_->beta();
    const Vector &daily(model_->day_of_week_pattern());
    const Vector &weekend(model_->weekend_hourly_pattern());
    const Vector &weekday(model_->weekday_hourly_pattern());
    const Matrix &exposure(model_->suf()->exposure());
    for (int d = 0; d < 6; ++d) {
      const Vector &hourly((d == Sat || d == Sun) ? weekend : weekday);
      for (int hour = 0; hour < 24; ++hour) {
        b += daily[d] * hourly[hour] * exposure(d, hour);
      }
    }
    double lambda = rgamma_mt(rng(), a, b);
    model_->set_average_daily_rate(lambda);
  }

  //----------------------------------------------------------------------
  void SAM::draw_daily_pattern() {
    Vector nu = model_->suf()->daily_event_count() + day_of_week_prior_->nu();
    Vector cand = rdirichlet_mt(rng(), nu);
    Vector orig = model_->day_of_week_pattern() / 7;

    double num = model_->loglike(WP::concatenate_params(
                     model_->average_daily_rate(), cand * 7,
                     model_->weekday_hourly_pattern(),
                     model_->weekend_hourly_pattern())) -
                 ddirichlet(cand, nu, true);

    double denom =
        model_->loglike(WP::concatenate_params(
            model_->average_daily_rate(), model_->day_of_week_pattern(),
            model_->weekday_hourly_pattern(),
            model_->weekend_hourly_pattern())) -
        ddirichlet(orig, nu, true);

    ++daily_pattern_attempts_;
    double logu = log(runif_mt(rng()));
    if (logu > num - denom) {
      // MH step failed
    } else {
      model_->set_day_of_week_pattern(cand * 7);
      ++daily_pattern_successes_;
    }
  }

  //----------------------------------------------------------------------
  void SAM::draw_weekend_hourly_pattern() {
    Vector nu = model_->suf()->weekend_hourly_event_count() +
                weekend_hourly_prior_->nu();

    Vector cand = rdirichlet_mt(rng(), nu);
    Vector orig = model_->weekend_hourly_pattern() / 24;

    double num =
        model_->loglike(WP::concatenate_params(
            model_->average_daily_rate(), model_->day_of_week_pattern(),
            model_->weekday_hourly_pattern(), cand * 24)) -
        ddirichlet(cand, nu, true);

    double denom =
        model_->loglike(WP::concatenate_params(
            model_->average_daily_rate(), model_->day_of_week_pattern(),
            model_->weekday_hourly_pattern(),
            model_->weekend_hourly_pattern())) -
        ddirichlet(orig, nu, true);

    ++weekend_hourly_attempts_;
    double logu = log(runif_mt(rng()));
    if (logu > num - denom) {
      // Do nothing.. MH failed
    } else {
      ++weekend_hourly_successes_;
      model_->set_weekend_hourly_pattern(cand * 24);
    }
  }

  //----------------------------------------------------------------------
  void SAM::draw_weekday_hourly_pattern() {
    Vector nu = model_->suf()->weekday_hourly_event_count() +
                weekday_hourly_prior_->nu();
    Vector cand = rdirichlet_mt(rng(), nu);
    Vector orig = model_->weekday_hourly_pattern() / 24;

    double num =
        model_->loglike(WP::concatenate_params(
            model_->average_daily_rate(), model_->day_of_week_pattern(),
            cand * 24, model_->weekend_hourly_pattern())) -
        ddirichlet(cand, nu, true);

    double denom =
        model_->loglike(WP::concatenate_params(
            model_->average_daily_rate(), model_->day_of_week_pattern(),
            model_->weekday_hourly_pattern(),
            model_->weekend_hourly_pattern())) -
        ddirichlet(orig, nu, true);

    ++weekday_hourly_attempts_;
    double logu = log(runif_mt(rng()));
    if (logu > num - denom) {
      // Do nothing
    } else {
      ++weekday_hourly_successes_;
      model_->set_weekday_hourly_pattern(cand * 24);
    }
  }

}  // namespace BOOM
