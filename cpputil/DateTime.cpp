// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2008-2012 Steven L. Scott

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
#include "cpputil/DateTime.hpp"
#include <cassert>
#include <cmath>
#include <ctime>
#include <ostream>
#include <sstream>

namespace BOOM {

  const double DateTime::microseconds_in_day_(seconds_in_day_ * 1.0e+6);
  const double DateTime::milliseconds_in_day_(seconds_in_day_ * 1000.0);
  const uint DateTime::seconds_in_day_(86400);
  const uint DateTime::minutes_in_day_(1440);
  const uint DateTime::hours_in_day_(24);

  const double DateTime::time_scale_factor_[7] = {
      double(microseconds_in_day_),
      double(milliseconds_in_day_),
      double(seconds_in_day_),
      double(minutes_in_day_),
      double(hours_in_day_),
      1.0,     // days in a day
      1.0 / 7  // weeks in a day
  };

  DateTime::DateTime() : t_(0.0) {
    time_t now;
    time(&now);
    struct tm *timeinfo;
    timeinfo = localtime(&now);
    d_.set(*timeinfo);
    t_ = double(timeinfo->tm_hour) / hours_in_day_ +
         double(timeinfo->tm_min) / minutes_in_day_ +
         double(timeinfo->tm_sec) / seconds_in_day_;
  }

  DateTime::DateTime(const Date &d, double fraction_of_day)
      : d_(d), t_(fraction_of_day) {
    assert(t_ >= 0 && t_ < 1.0);
  }

  DateTime::DateTime(const Date &d, uint hour, uint min, double sec) : d_(d) {
    assert(hour < 24);
    assert(min < 60);
    assert(sec < 60.0000);
    t_ = hour / 24.0 + min / 24.0 / 60 + sec / 24 / 3600;
    assert(t_ >= 0 && t_ < 1.0);
  }

  DateTime::DateTime(double time_since_midnight_starting_jan_1_1970,
                     TimeScale scale) {
    double days =
        time_since_midnight_starting_jan_1_1970 / time_scale_factor_[scale];
    int integer_days = lround(floor(days));
    d_.set(integer_days);
    t_ = days - integer_days;
  }

  double DateTime::seconds_to_next_day() const {
    return seconds_in_day_ * (1 - t_);
  }
  double DateTime::seconds_into_day() const { return seconds_in_day_ * t_; }

  bool DateTime::operator<(const DateTime &rhs) const {
    if (d_ == rhs.d_) return t_ < rhs.t_;
    return d_ < rhs.d_;
  }

  bool DateTime::operator==(const DateTime &rhs) const {
    if (d_ != rhs.d_) return false;
    if (t_ < rhs.t_ || t_ > rhs.t_) return false;
    return true;
  }

  // Compute the remainder when x is divided by y.
  inline double rem(double x, double y) {
    double v = floor(x / y);
    return x - v * y;
  }

  DateTime &DateTime::operator+=(double days) {
    if (days < 0) return (*this) -= (-days);
    t_ += days;
    if (t_ >= 1) {
      double frac = rem(t_, 1.0);
      long ndays = lround(t_ - frac);
      d_ += ndays;
      t_ = frac;
    }
    return *this;
  }

  DateTime &DateTime::operator-=(double days) {
    if (days < 0) return (*this) += (-days);

    t_ -= days;
    if (t_ < 0) {
      double frac = rem(t_, 1.0);      // a negative number in (-1,0]
      long ndays = lround(floor(t_));  // a negative number <= t_
      d_ += ndays;
      t_ = 1 - frac;
    }
    return *this;
  }

  long DateTime::hour() const { return lround(floor(t_ * hours_in_day_)); }

  long DateTime::minute() const {
    double m = rem(t_ * minutes_in_day_, 60);
    assert(m >= 0);
    return lround(floor(m));
  }

  long DateTime::second() const {
    // t_ * seconds_in_day_ is the number of seconds you are into the
    // day.  If you divide by 60 then you will get the number of
    // minutes you are into the day, and the remainder will be the
    // number of seconds you are into the minute.
    double s = rem(t_ * seconds_in_day_, 60);
    assert(s >= 0);
    return lround(floor(s));
  }

  const Date &DateTime::date() const { return d_; }

  double DateTime::hours_left_in_day() const {
    return hours_in_day_ * (1 - t_);
  }

  double DateTime::minutes_left_in_hour() const {
    double current_hour = 24 * t_;
    double fraction_of_hour = current_hour - floor(current_hour);
    return fraction_of_hour > 0 ? 60 * (1 - fraction_of_hour) : 0;
  }

  double DateTime::seconds_left_in_minute() const {
    double current_minute = t_ * 24 * 60;
    double fraction_of_minute = current_minute - floor(current_minute);
    return fraction_of_minute > 0 ? 60 * (1 - fraction_of_minute) : 0;
  }

  double DateTime::seconds_left_in_hour() const {
    double current_hour = 24 * t_;
    double fraction_of_hour = current_hour - floor(current_hour);
    // fraction_of_hour is in [0, 1]
    const double seconds_in_an_hour = 3600;
    return fraction_of_hour > 0 ? seconds_in_an_hour * (1 - (fraction_of_hour))
                                : 0;
  }

  double DateTime::time_to_next_hour() const {
    double current_hour = 24 * t_;
    double fraction_of_hour = current_hour - floor(current_hour);
    return fraction_of_hour > 0 ? (1 - fraction_of_hour) / 24 : 1.0 / 24;
  }

  std::ostream &DateTime::print(std::ostream &out) const {
    double hr = hour();
    double min = minute();
    double sec = second();
    double frac = t_ - (hr / 24 + min / 24 / 60 + sec / 24 / 60 / 60);
    frac *= seconds_in_day_;
    sec += frac;
    // Writing to a string before writing to 'out' is a quick hack to
    // make sure we respect the setw() manipulator.
    std::ostringstream formatted_output;
    formatted_output << d_ << " " << hr << ":" << min << ":" << sec;
    out << formatted_output.str();
    return out;
  }

  std::ostream &operator<<(std::ostream &out, const DateTime &dt) {
    return dt.print(out);
  }

  double DateTime::operator-(const DateTime &rhs) const {
    int ndays = d_ - rhs.d_;
    double dt = t_ - rhs.t_;
    return dt + ndays;
  }

  double DateTime::weeks_to_days(double t) { return t * 7; }
  double DateTime::days_to_days(double t) { return t; }
  double DateTime::hours_to_days(double t) { return t / hours_in_day_; }
  double DateTime::minutes_to_days(double t) { return t / minutes_in_day_; }
  double DateTime::seconds_to_days(double t) { return t / seconds_in_day_; }
  double DateTime::milliseconds_to_days(double t) {
    return t / milliseconds_in_day_;
  }
  double DateTime::microseconds_to_days(double t) {
    return t / microseconds_in_day_;
  }

}  // namespace BOOM
