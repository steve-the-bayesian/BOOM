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

#ifndef BOOM_DATE_TIME_HPP
#define BOOM_DATE_TIME_HPP
#include <string>
#include "cpputil/Date.hpp"

namespace BOOM {

  // A DateTime is a point in continuous time.
  class DateTime {
   public:
    // Default constructor uses "now" in local time, with one second
    // resolution.
    DateTime();

    DateTime(const Date &, double fraction_of_day);
    DateTime(const Date &, uint hour, uint min, double sec);

    // Use this constructor when times are continuous real numbers
    // (e.g. Unix time_t).
    enum TimeScale {
      microsecond_scale,
      millisecond_scale,
      second_scale,
      minute_scale,
      hour_scale,
      day_scale,
      week_scale
    };
    explicit DateTime(double time_since_midnight_starting_jan_1_1970,
             TimeScale timescale = day_scale);

    bool operator<(const DateTime &rhs) const;
    bool operator==(const DateTime &rhs) const;
    
    // The remaining operations are in terms of < and ==.
    bool operator!=(const DateTime &rhs) const {
      return !(*this == rhs);
    }
    bool operator<=(const DateTime &rhs) const {
      return *this == rhs || *this < rhs;
    }
    bool operator>=(const DateTime &rhs) const {return !(*this < rhs);}
    bool operator>(const DateTime &rhs) const {return ! (*this <= rhs);}

    DateTime &operator+=(double days);
    DateTime &operator-=(double days);

    // Returns the (real) number of days between *this and rhs.
    double operator-(const DateTime &rhs) const;

    long hour() const;    // 0..23
    long minute() const;  // 0..59
    long second() const;  // 0..59

    const Date &date() const;

    // Compute the amount of time remaining before the next epoch.  In
    // each case, the final time period in the epoch is 0, the instant
    // of the start of the final time period is 1, etc.
    double hours_left_in_day() const;       // 0..24
    double minutes_left_in_hour() const;    // 0..60
    double seconds_left_in_minute() const;  // 0..60

    // The number of seconds into a day.
    double seconds_into_day() const;  // [0..86400)
    // The time to the next day is always strictly positive, so if the
    // current time is exactly on the start of a day then
    // seconds_to_next_day is 86400.
    double seconds_to_next_day() const;  // [86400..0)

    // Return the number of seconds (including fractional seconds)
    // remaining before the next hour.
    double seconds_left_in_hour() const;

    //  Time to next hour, as a fraction of a day.  Times are always
    //  strictly positive, so if the current time is exactly on an
    //  hour boundary then the time to the next hour will be one hour.
    double time_to_next_hour() const;

    // Time, as a fraction of a day, until the start of the next day.
    // Can return 0, but never 1.
    double fraction_of_day_remaining() const;

    std::ostream &print(std::ostream &) const;

    // Convert the given amount of the given time unit to days
    // (including fractions of a day).
    // Examples:
    // hours_to_days(1) = 1.0/24
    // weeks_to_days(2) = 14
    static double weeks_to_days(double duration);
    static double days_to_days(double duration);
    static double hours_to_days(double duration);
    static double minutes_to_days(double duration);
    static double seconds_to_days(double duration);
    static double milliseconds_to_days(double duration);
    static double microseconds_to_days(double duration);

   private:
    Date d_;
    double t_;  // fraction of day [0,1)
    static const double time_scale_factor_[7];
    static const uint seconds_in_day_;
    static const uint minutes_in_day_;
    static const uint hours_in_day_;
    static const double milliseconds_in_day_;
    static const double microseconds_in_day_;
  };

  std::ostream &operator<<(std::ostream &out, const DateTime &dt);

  inline DateTime operator+(const DateTime &time, double duration) {
    DateTime ans(time);
    ans += duration;
    return ans;
  }
  inline DateTime operator-(const DateTime &time, double duration) {
    return time + (-duration);
  }
  
}  // namespace BOOM

#endif  // BOOM_DATE_TIME_HPP
