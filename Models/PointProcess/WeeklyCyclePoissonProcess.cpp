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

#include "Models/PointProcess/WeeklyCyclePoissonProcess.hpp"
#include <iomanip>
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef WeeklyCyclePoissonSuf WS;
    typedef WeeklyCyclePoissonProcess WP;
  }  // namespace

  const Vector WS::one_7(7, 1.0);
  const Vector WS::one_24(24, 1.0);

  WS::WeeklyCyclePoissonSuf() : count_(7, 24, 0.0), exposure_(7, 24, 0.0) {}

  WS *WS::clone() const { return new WS(*this); }

  void WS::clear() {
    count_ = 0;
    exposure_ = 0;
  }

  //----------------------------------------------------------------------
  void WS::Update(const PointProcess &data) {
    // Incrementing event counts is easy...
    for (int i = 0; i < data.number_of_events(); ++i) {
      const DateTime &event(data.event(i).timestamp());
      DayNames day = event.date().day_of_week();
      int hour = event.hour();
      ++count_(day, hour);
    }

    // Increment exposure by integrating over the observation window.
    const DateTime &window_begin(data.window_begin());
    const DateTime &window_end(data.window_end());
    add_exposure_window(window_begin, window_end);
  }
  //----------------------------------------------------------------------
  void WS::add_event(const DateTime &event) {
    ++count_(event.date().day_of_week(), event.hour());
  }
  //----------------------------------------------------------------------
  void WS::add_exposure_window(const DateTime &window_begin,
                               const DateTime &window_end) {
    double duration = window_end - window_begin;

    // Define some constants.
    const double one_hour = DateTime::hours_to_days(1.0);
    const double one_week = 7.0;

    // If the window is longer than one week, then increment the
    // appropriate number of weeks.
    if (duration >= one_week) {
      double number_of_weeks = floor(duration / one_week);
      if (number_of_weeks >= 1) {
        exposure_ += number_of_weeks * one_hour;
        duration -= number_of_weeks * 7;
      }
    }

    // What remains is an interation over at most 168 time buckets.
    double time_to_next_hour = window_begin.time_to_next_hour();
    double dt = std::min<double>(duration, time_to_next_hour);
    DayNames day = window_begin.date().day_of_week();
    int hour = window_begin.hour();
    while (duration > 0) {
      exposure_(day, hour) += dt;
      duration -= dt;
      ++hour;
      if (hour == 24) {
        hour = 0;
        day = next(day);
      }
      dt = std::min<double>(one_hour, duration);
    }
  }
  //----------------------------------------------------------------------

  WS *WS::combine(const Ptr<WS> &s) { return combine(*s); }
  WS *WS::combine(const WS &s) {
    count_ += s.count_;
    exposure_ += s.exposure_;
    return this;
  }

  WS *WS::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  Vector WS::vectorize(bool) const {
    Vector ans(24 * 7 * 2);
    std::copy(count_.begin(), count_.end(), ans.begin());
    std::copy(exposure_.begin(), exposure_.end(), ans.begin() + 168);
    return ans;
  }

  Vector::const_iterator WS::unvectorize(Vector::const_iterator &v, bool) {
    count_.assign(v, v + 168);
    v += 168;
    exposure_.assign(v, v + 168);
    v += 168;
    return v;
  }

  Vector::const_iterator WS::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return this->unvectorize(it, minimal);
  }

  std::ostream &WS::print(std::ostream &out) const {
    out << "Counts (top) and exposure times:" << endl;
    out << setw(4) << " ";
    for (int d = 0; d < 7; ++d) {
      out << setw(10) << DayNames(d);
    }
    out << endl;

    for (int h = 0; h < 24; ++h) {
      out << setw(4) << std::left << h;
      for (int d = 0; d < 7; ++d) {
        out << setw(10) << count_(d, h);
      }
      out << endl;

      out << setw(4) << " ";
      for (int d = 0; d < 7; ++d) {
        out << setw(10) << exposure_(d, h);
      }
      out << endl;
    }
    return out;
  }

  Vector WS::daily_event_count() const { return count_ * one_24; }

  Vector WS::weekend_hourly_event_count() const {
    Vector ans(24, 0.0);
    ans += count_.row(Sat);
    ans += count_.row(Sun);
    return ans;
  }

  Vector WS::weekday_hourly_event_count() const {
    Vector ans(24, 0.0);
    for (int d = 0; d < 7; ++d) {
      if (d == Sat || d == Sun) continue;
      ans += count_.row(d);
    }
    return ans;
  }

  const Matrix &WS::count() const { return count_; }

  const Matrix &WS::exposure() const { return exposure_; }

  //======================================================================

  WP::WeeklyCyclePoissonProcess()
      : ParamPolicy(new UnivParams(1.0), new VectorParams(7, 1.0),
                    new VectorParams(24, 1.0), new VectorParams(24, 1.0)),
        DataPolicy(new WS) {}

  WP *WP::clone() const { return new WP(*this); }

  double WP::event_rate(const DateTime &t) const {
    DayNames day = t.date().day_of_week();
    int hour = t.hour();
    return event_rate(day, hour);
  }

  double WP::event_rate(const DayNames day, int hour) const {
    return average_daily_rate() * day_of_week_pattern()[day] *
           hourly_pattern(day)[hour];
  }

  double WP::expected_number_of_events(const DateTime &t0,
                                       const DateTime &t1) const {
    double duration = t1 - t0;
    int weeks = lround(floor(duration / 7));
    double lambda = average_daily_rate();
    double ans = 7 * weeks * lambda;
    duration -= 7 * weeks;

    const double one_hour = DateTime::hours_to_days(1.0);
    double time_to_next_hour = t0.time_to_next_hour();
    if (time_to_next_hour == 0) time_to_next_hour = one_hour;
    DayNames day = t0.date().day_of_week();
    int hour = t0.hour();
    double dt = std::min<double>(time_to_next_hour, duration);
    while (duration > 0) {
      ans += dt * event_rate(day, hour);
      duration -= dt;
      ++hour;
      if (hour == 24) {
        hour = 0;
        day = next(day);
      }
      dt = std::min<double>(one_hour, duration);
    }
    return ans;
  }

  double WP::loglike(const Vector &lam0_delta_weekday_weekend) const {
    const Matrix &exposure(suf()->exposure());
    const Matrix &count(suf()->count());

    double lambda0 = lam0_delta_weekday_weekend[0];
    Vector delta(7, 0.0);
    int pos = 1;
    VectorView(delta, 0, 6) =
        ConstVectorView(lam0_delta_weekday_weekend, pos, 6);
    delta[6] = 7.0 - sum(delta);
    pos += 6;

    Vector eta_weekday(24, 0.0);
    VectorView(eta_weekday, 0, 23) =
        ConstVectorView(lam0_delta_weekday_weekend, pos, 23);
    eta_weekday[23] = 24.0 - sum(eta_weekday);
    pos += 23;

    Vector eta_weekend(24, 0.0);
    VectorView(eta_weekend, 0, 23) =
        ConstVectorView(lam0_delta_weekday_weekend, pos, 23);
    eta_weekend[23] = 24.0 - sum(eta_weekend);

    double ans = 0;
    for (int d = 0; d < 7; ++d) {
      const Vector &eta((d == Sat || d == Sun) ? eta_weekend : eta_weekday);
      for (int h = 0; h < 24; ++h) {
        double lam = lambda0 * delta[d] * eta[h] * exposure(d, h);
        ans += dpois(count(d, h), lam, true);
      }
    }
    return ans;
  }

  const Vector &WP::hourly_pattern(int day) const {
    if (day == Sat || day == Sun) return weekend_hourly_pattern();
    return weekday_hourly_pattern();
  }

  namespace {
    // Copies all but the last element of 'from' to the sequence
    // pointed at by 'to'.  Returns the next available position in the
    // 'to' sequence.
    inline Vector::iterator copy_all_but_last_element(const Vector &from,
                                                      Vector::iterator to) {
      return std::copy(from.begin(), from.end() - 1, to);
    }
  }  // namespace

  Vector WP::concatenate_params(double lambda, const Vector &daily,
                                const Vector &weekday_hourly,
                                const Vector &weekend_hourly) {
    if (daily.size() != 7 || weekday_hourly.size() != 24 ||
        weekend_hourly.size() != 24) {
      report_error(
          "Wrong size inputs to WeeklyCyclePoissonProcess::"
          "concatenate_params()");
    }
    Vector ans(1 + 6 + 23 + 23);
    ans[0] = lambda;
    Vector::iterator it = copy_all_but_last_element(daily, ans.begin() + 1);
    it = copy_all_but_last_element(weekday_hourly, it);
    copy_all_but_last_element(weekend_hourly, it);
    return ans;
  }

  void WP::mle() {
    double old_loglike = loglike(
        concatenate_params(average_daily_rate(), day_of_week_pattern(),
                           weekday_hourly_pattern(), weekend_hourly_pattern()));
    double dloglike = 1.0;
    while (dloglike > 1e-5) {
      maximize_average_daily_rate();
      maximize_daily_pattern();
      maximize_hourly_pattern();
      double new_loglike = loglike(concatenate_params(
          average_daily_rate(), day_of_week_pattern(), weekday_hourly_pattern(),
          weekend_hourly_pattern()));
      dloglike = new_loglike - old_loglike;
      old_loglike = new_loglike;
    }
  }

  void WP::maximize_average_daily_rate() {
    const Matrix &count(suf()->count());
    const Matrix &exposure(suf()->exposure());
    double total_count = 0;
    double total_exposure = 0;
    const Vector &delta(day_of_week_pattern());
    for (int d = 0; d < 7; ++d) {
      const Vector &eta(hourly_pattern(d));
      for (int h = 0; h < 24; ++h) {
        total_count += count(d, h);
        total_exposure += delta[d] * eta[h] * exposure(d, h);
      }
    }
    set_average_daily_rate(total_count / total_exposure);
  }

  void WP::maximize_daily_pattern() {
    const Matrix &count(suf()->count());
    const Matrix &exposure(suf()->exposure());
    Vector delta(7);
    double lambda = average_daily_rate();
    for (int d = 0; d < 7; ++d) {
      const Vector &eta(hourly_pattern(d));
      double total_count = 0;
      double total_exposure = 0;
      for (int h = 0; h < 24; ++h) {
        total_count += count(d, h);
        total_exposure += exposure(d, h) * lambda * eta[h];
      }
      delta[d] = total_count / total_exposure;
    }
    set_day_of_week_pattern(delta);
    // TODO(stevescott):  check that this enforces sum(delta) == 7
  }

  void WP::maximize_hourly_pattern() {
    const Matrix &count(suf()->count());
    const Matrix &exposure(suf()->exposure());
    const Vector &delta(day_of_week_pattern());
    double lambda = average_daily_rate();
    Vector eta_weekend(24, 0.0);
    Vector eta_weekday(24, 0.0);
    for (int h = 0; h < 24; ++h) {
      double total_count_weekday = 0;
      double total_exposure_weekday = 0;
      double total_count_weekend = 0;
      double total_exposure_weekend = 0;
      double *total_count;
      double *total_exposure;
      for (int d = 0; d < 7; ++d) {
        if (d == Sat || d == Sun) {
          total_exposure = &total_exposure_weekend;
          total_count = &total_count_weekend;
        } else {
          total_exposure = &total_exposure_weekday;
          total_count = &total_count_weekday;
        }
        *total_count += count(d, h);
        *total_exposure += exposure(d, h) * lambda * delta[d];
      }
      eta_weekend[h] = total_count_weekend / total_exposure_weekend;
      eta_weekday[h] = total_count_weekday / total_exposure_weekday;
    }
    set_weekday_hourly_pattern(eta_weekday);
    set_weekend_hourly_pattern(eta_weekend);
  }

  double WP::average_daily_rate() const {
    return average_daily_event_rate_prm()->value();
  }
  void WP::set_average_daily_rate(double lam) {
    average_daily_event_rate_prm()->set(lam);
  }
  const Vector &WP::day_of_week_pattern() const {
    return day_of_week_cycle_prm()->value();
  }
  void WP::set_day_of_week_pattern(const Vector &delta) {
    day_of_week_cycle_prm()->set(delta);
  }
  const Vector &WP::weekend_hourly_pattern() const {
    return weekend_hour_of_day_cycle_prm()->value();
  }
  void WP::set_weekend_hourly_pattern(const Vector &eta) {
    weekend_hour_of_day_cycle_prm()->set(eta);
  }

  const Vector &WP::weekday_hourly_pattern() const {
    return weekday_hour_of_day_cycle_prm()->value();
  }
  void WP::set_weekday_hourly_pattern(const Vector &eta) {
    weekday_hour_of_day_cycle_prm()->set(eta);
  }

  // Simulate a WeeklyCyclePoissonProcess by thinning
  PointProcess WP::simulate(RNG &rng, const DateTime &t0, const DateTime &t1,
                            std::function<Data *()> mark_generator) const {
    PointProcess ans(t0, t1);
    double max_rate = 0;
    for (int d = 0; d < 7; ++d) {
      for (int h = 0; h < 24; ++h) {
        max_rate = std::max(max_rate, event_rate(DayNames(d), h));
      }
    }

    double duration = t1 - t0;
    int number_of_candidate_events = rpois_mt(rng, max_rate * duration);
    Vector times(number_of_candidate_events);
    for (int i = 0; i < number_of_candidate_events; ++i) {
      times[i] = runif_mt(rng, 0, duration);
    }
    times.sort();

    for (int i = 0; i < times.size(); ++i) {
      DateTime cand = t0 + times[i];
      double prob = event_rate(cand) / max_rate;
      if (runif_mt(rng, 0, 1) < prob) {
        Data *mark = mark_generator();
        if (mark) {
          ans.add_event(cand, Ptr<Data>(mark));
        } else {
          ans.add_event(cand);
        }
      }
    }
    return ans;
  }

  Ptr<UnivParams> WP::average_daily_event_rate_prm() { return prm1(); }
  const Ptr<UnivParams> WP::average_daily_event_rate_prm() const {
    return prm1();
  }
  Ptr<VectorParams> WP::day_of_week_cycle_prm() { return prm2(); }
  const Ptr<VectorParams> WP::day_of_week_cycle_prm() const { return prm2(); }
  Ptr<VectorParams> WP::weekday_hour_of_day_cycle_prm() { return prm3(); }
  const Ptr<VectorParams> WP::weekday_hour_of_day_cycle_prm() const {
    return prm3();
  }
  Ptr<VectorParams> WP::weekend_hour_of_day_cycle_prm() { return prm4(); }
  const Ptr<VectorParams> WP::weekend_hour_of_day_cycle_prm() const {
    return prm4();
  }

  void WP::add_data_raw(const PointProcess &data) { suf()->Update(data); }

  void WP::add_exposure_window(const DateTime &t0, const DateTime &t1) {
    suf()->add_exposure_window(t0, t1);
  }
  void WP::add_event(const DateTime &t) { suf()->add_event(t); }

}  // namespace BOOM
