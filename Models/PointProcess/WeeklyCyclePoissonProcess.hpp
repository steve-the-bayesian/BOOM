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
#ifndef BOOM_WEEKLY_CYCLE_POISSON_PROCESS_HPP_
#define BOOM_WEEKLY_CYCLE_POISSON_PROCESS_HPP_

#include <functional>
#include "Models/PointProcess/PointProcess.hpp"
#include "Models/PointProcess/PoissonProcess.hpp"
#include "Models/Policies/ParamPolicy_4.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Sufstat.hpp"

namespace BOOM {

  class WeeklyCyclePoissonSuf : public SufstatDetails<PointProcess> {
   public:
    WeeklyCyclePoissonSuf();
    WeeklyCyclePoissonSuf *clone() const override;
    void clear() override;

    void Update(const PointProcess &data) override;
    void add_exposure_window(const DateTime &t0, const DateTime &t1);
    void add_event(const DateTime &t);

    WeeklyCyclePoissonSuf *combine(const Ptr<WeeklyCyclePoissonSuf> &);
    WeeklyCyclePoissonSuf *combine(const WeeklyCyclePoissonSuf &);
    WeeklyCyclePoissonSuf *abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

    Vector daily_event_count() const;
    Vector weekday_hourly_event_count() const;
    Vector weekend_hourly_event_count() const;

    // Returns a matrix where the (day, hour) element gives the total
    // exposure time (measured in fractoins of a day) for that hour in
    // that day of the week.
    const Matrix &exposure() const;
    const Matrix &count() const;

   private:
    // Keeps track of the number of events that take place during each
    // hour of the week.  Indexed by (day, hour).
    Matrix count_;

    // Keeps track of the number of hours (including fractional hours)
    // exposed during each hour of the week.  Time in each cell is
    // measured in days (not hours).
    Matrix exposure_;

    static const Vector one_7;
    static const Vector one_24;
  };

  // A Poisson process containing a day of week and hour of day cycle.
  class WeeklyCyclePoissonProcess
      : public PoissonProcess,
        public ParamPolicy_4<UnivParams,     // Weekly rate
                             VectorParams,   // Daily factor
                             VectorParams,   // Weekday hourly factor
                             VectorParams>,  // Weekend hourly factor
        public SufstatDataPolicy<PointProcess, WeeklyCyclePoissonSuf>,
        public PriorPolicy,
        public LoglikeModel {
   public:
    WeeklyCyclePoissonProcess();
    WeeklyCyclePoissonProcess *clone() const override;

    // Concatenate a collection of 4 parameters into a single vector
    // that can be passed to loglike().
    // Args:
    //   lambda:  The average daily event rate (>0).
    //   daily:  A 7-vector of non-negative elements that sum to 7.
    //   weekday_hourly:  A 24-vector of non-negative elements that sums to 24.
    //   weekend_hourly:  A 24-vector of non-negative elements that sums to 24.
    //
    // Returns:
    //   A vector containing lambda, the first 6 elements of daily,
    //   the first 23 elements of weekday_hourly, and the first 23
    //   elements of weekend_hourly.
    static Vector concatenate_params(double lambda, const Vector &daily,
                                     const Vector &weekday_hourly,
                                     const Vector &weekend_hourly);
    double loglike(const Vector &lam0_delta_weekday_weekend) const override;
    void mle() override;

    double event_rate(const DateTime &t) const override;
    double event_rate(DayNames day, int hour) const;

    double expected_number_of_events(const DateTime &t0,
                                     const DateTime &t1) const override;
    double average_daily_rate() const;
    void set_average_daily_rate(double lambda);

    const Vector &day_of_week_pattern() const;  // sums to 7
    void set_day_of_week_pattern(const Vector &pattern);

    const Vector &weekday_hourly_pattern() const;  // sums to 24
    void set_weekday_hourly_pattern(const Vector &pattern);

    const Vector &weekend_hourly_pattern() const;  // sums to 24
    void set_weekend_hourly_pattern(const Vector &pattern);

    PointProcess simulate(RNG &rng, const DateTime &t0, const DateTime &t1,
                          std::function<Data *()> mark_generator =
                              NullDataGenerator()) const override;

    Ptr<UnivParams> average_daily_event_rate_prm();
    const Ptr<UnivParams> average_daily_event_rate_prm() const;
    Ptr<VectorParams> day_of_week_cycle_prm();
    const Ptr<VectorParams> day_of_week_cycle_prm() const;
    Ptr<VectorParams> weekday_hour_of_day_cycle_prm();
    const Ptr<VectorParams> weekday_hour_of_day_cycle_prm() const;
    Ptr<VectorParams> weekend_hour_of_day_cycle_prm();
    const Ptr<VectorParams> weekend_hour_of_day_cycle_prm() const;

    void add_data_raw(const PointProcess &);
    void add_exposure_window(const DateTime &t0, const DateTime &t1) override;
    void add_event(const DateTime &t) override;

   private:
    const Vector &hourly_pattern(int day) const;
    void maximize_average_daily_rate();
    void maximize_daily_pattern();
    void maximize_hourly_pattern();
  };

}  // namespace BOOM

#endif  // BOOM_WEEKLY_CYCLE_POISSON_PROCESS_HPP_
