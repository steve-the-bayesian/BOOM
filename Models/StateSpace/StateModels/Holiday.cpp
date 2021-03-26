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

#include "Models/StateSpace/StateModels/Holiday.hpp"
#include <algorithm>
#include <cassert>
#include "cpputil/report_error.hpp"

namespace BOOM {

  Date SingleDayHoliday::nearest(const Date &d) const {
    Date next_holiday(date_on_or_after(d));
    if (next_holiday == d) {
      return next_holiday;
    }
    Date previous_holiday(date_on_or_before(d));
    if ((d - previous_holiday) < (next_holiday - d)) {
      return previous_holiday;
    } else {
      return next_holiday;
    }
  }

  bool SingleDayHoliday::active(const Date &date) const {
    Date holiday_date(nearest(date));
    return (date <= holiday_date && date >= earliest_influence(holiday_date))
        || (date >= holiday_date && date <= latest_influence(holiday_date));
  }

  //======================================================================
  OrdinaryAnnualHoliday::OrdinaryAnnualHoliday(int days_before, int days_after)
      : days_before_(days_before), days_after_(days_after) {
    if (days_before < 0 || days_after < 0) {
      report_error("Influence window must have non-negative size.");
    }
  }

  Date OrdinaryAnnualHoliday::earliest_influence(
      const Date &date_in_window) const {
    return nearest(date_in_window) - days_before_;
  }

  Date OrdinaryAnnualHoliday::latest_influence(
      const Date &date_in_window) const {
    return nearest(date_in_window) + days_after_;
  }

  int OrdinaryAnnualHoliday::maximum_window_width() const {
    return 1 + days_before_ + days_after_;
  }

  Date OrdinaryAnnualHoliday::date(int year) const {
    std::map<Year, Date>::iterator it = date_lookup_table_.find(year);
    if (it != date_lookup_table_.end()) {
      return it->second;
    }
    Date ans = compute_date(year);
    date_lookup_table_[year] = ans;
    return ans;
  }

  Date OrdinaryAnnualHoliday::date_on_or_after(const Date &d) const {
    Date date_in_same_year(date(d.year()));
    if (date_in_same_year >= d) {
      return date_in_same_year;
    } else {
      return date(d.year() + 1);
    }
  }

  Date OrdinaryAnnualHoliday::date_on_or_before(const Date &d) const {
    Date date_in_same_year(date(d.year()));
    if (date_in_same_year <= d) {
      return date_in_same_year;
    } else {
      return date(d.year() - 1);
    }
  }

  //======================================================================
  FixedDateHoliday::FixedDateHoliday(int month, int day_of_month,
                                     int days_before, int days_after)
      : OrdinaryAnnualHoliday(days_before, days_after),
        month_name_(MonthNames(month)),
        day_of_month_(day_of_month) {}

  Date FixedDateHoliday::compute_date(int year) const {
    Date ans(month_name_, day_of_month_, year);
    return ans;
  }

  //======================================================================
  NthWeekdayInMonthHoliday::NthWeekdayInMonthHoliday(int which_week,
                                                     DayNames day,
                                                     MonthNames month,
                                                     int days_before,
                                                     int days_after)
      : OrdinaryAnnualHoliday(days_before, days_after),
        which_week_(which_week),
        day_name_(day),
        month_name_(month) {}

  Date NthWeekdayInMonthHoliday::compute_date(int year) const {
    return nth_weekday_in_month(which_week_, day_name_, month_name_, year);
  }

  //======================================================================
  LastWeekdayInMonthHoliday::LastWeekdayInMonthHoliday(DayNames day,
                                                       MonthNames month,
                                                       int days_before,
                                                       int days_after)
      : OrdinaryAnnualHoliday(days_before, days_after),
        day_name_(day),
        month_name_(month) {}

  Date LastWeekdayInMonthHoliday::compute_date(int year) const {
    return last_weekday_in_month(day_name_, month_name_, year);
  }
  //======================================================================
  FloatingHoliday::FloatingHoliday(int days_before, int days_after)
      : OrdinaryAnnualHoliday(days_before, days_after) {}

  //======================================================================
  DateRangeHoliday::DateRangeHoliday() : maximum_window_width_(-1) {}

  DateRangeHoliday::DateRangeHoliday(const std::vector<Date> &from,
                                     const std::vector<Date> &to)
      : maximum_window_width_(-1) {
    if (from.size() != to.size()) {
      report_error(
          "'from' and 'to' must contain the same number "
          "of elements.");
    }
    for (int i = 0; i < from.size(); ++i) {
      add_dates(from[i], to[i]);
    }
  }

  void DateRangeHoliday::add_dates(const Date &from, const Date &to) {
    if (to < from) {
      report_error("'from' must come before 'to'.");
    }
    if (!begin_.empty() && from <= begin_.back()) {
      report_error(
          "Dates must be added in sequential order.  "
          "Please sort by start date before calling add_dates.");
    }
    int width = to - from + 1;
    if (width > maximum_window_width_) {
      maximum_window_width_ = width;
    }
    begin_.push_back(from);
    end_.push_back(to);
  }

  bool DateRangeHoliday::active(const Date &arbitrary_date) const {
    const auto it =
        std::lower_bound(end_.cbegin(), end_.cend(), arbitrary_date);
    // lower_bound returns the first date greater than or equal to
    // arbitrary_date.
    if (it == end_.cend()) {
      // If no date was found then arbitrary_date is larger than all the dates
      // in the date range.
      return false;
    }
    if (arbitrary_date == *it) {
      // In this case arbitrary_date occurs on the last day of one of the
      // influence intervals.
      return true;
    } else {
      // Find the start of the interval corresponding to the endpoint referred
      // to by *it.  If the arbitrary_date >= this time point then it occurs
      // inside an interval covered by the holiday.  If not then it doesn't.
      int position = it - end_.cbegin();
      return arbitrary_date >= begin_[position];
    }
  }

  Date DateRangeHoliday::earliest_influence(const Date &date) const {
    // 'lower_bound' finds the first element >= date.  Use it to find the
    // endpoint of the interval containing 'date', then return the corresponding
    // start point for the interval.
    const auto it = std::lower_bound(end_.cbegin(), end_.cend(), date);
    if (it != end_.cend()) {
      int position = it - end_.cbegin();
      if (begin_[position] <= date) {
        return begin_[position];
      }
    }
    report_error("Holiday is not active on the given date.");
    return date;
  }

  Date DateRangeHoliday::latest_influence(const Date &date) const {
    const auto it = std::lower_bound(end_.cbegin(), end_.cend(), date);
    if (it != end_.cend()) {
      int position = it - end_.cbegin();
      if (date >= begin_[position]) {
        return *it;
      }
    }
    report_error("Holiday is not active on the given date.");
    return date;
  }

  //======================================================================
  SuperBowlSunday::SuperBowlSunday(int days_before, int days_after)
      : FloatingHoliday(days_before, days_after) {}

  // The super bowl is currently (2011) played on the first sunday in Feb.
  Date SuperBowlSunday::compute_date(int year) const {
    if (year == 2003)
      return Date(Jan, 26, 2003);
    else if (year == 1989)
      return Date(Jan, 22, 1989);
    else if (year == 1985)
      return Date(Jan, 20, 1985);
    else if (year == 1983)
      return Date(Jan, 30, 1983);
    else if (year == 1980)
      return Date(Jan, 20, 1980);
    else if (year == 1979)
      return Date(Jan, 21, 1979);
    else if (year == 1976)
      return Date(Jan, 18, 1976);
    else if (year == 1972)
      return Date(Jan, 16, 1972);
    else if (year == 1971)
      return Date(Jan, 17, 1971);
    if (year >= 2002) {
      // After 2002, the Super Bowl is played on the first Sunday in February.
      return nth_weekday_in_month(1, Sun, Feb, year);
    } else if (year >= 1986) {
      // Last Sun in Jan
      return last_weekday_in_month(Sun, Jan, year);
    } else if (year >= 1979) {
      // 4th Sun in Jan
      return nth_weekday_in_month(4, Sun, Jan, year);
    } else if (year >= 1967) {
      // 2nd Sunday, not counting new years
      Date jan1(Jan, 1, year);
      if (jan1.day_of_week() == Sun) ++jan1;
      return jan1 + (jan1.days_until(Sun) + 7);
    } else {
      report_error("No SuperBowl before 1967");
    }
    // should never get here
    return Date(Jan, 1, 1000);
  }
  //======================================================================

  USDaylightSavingsTimeBegins::USDaylightSavingsTimeBegins(int days_before,
                                                           int days_after)
      : FloatingHoliday(days_before, days_after) {}

  Date USDaylightSavingsTimeBegins::compute_date(int year) const {
    if (year < 1967) {
      report_error("Can't compute USDaylightSavingsTime before 1967.");
    }
    if (year > 2006) {
      // Second Sunday in March
      return nth_weekday_in_month(2, Sun, Mar, year);
    } else if (year >= 1987) {
      return nth_weekday_in_month(1, Sun, Apr, year);
    }
    return last_weekday_in_month(Sun, Apr, year);
  }
  //======================================================================
  USDaylightSavingsTimeEnds::USDaylightSavingsTimeEnds(int days_before,
                                                       int days_after)
      : FloatingHoliday(days_before, days_after) {}

  Date USDaylightSavingsTimeEnds::compute_date(int year) const {
    if (year < 1967) {
      report_error("Can't compute USDaylightSavingsTime before 1967.");
    }
    if (year > 2006) {
      return nth_weekday_in_month(1, Sun, Nov, year);
    }
    return last_weekday_in_month(Sun, Oct, year);
  }

  //======================================================================
  EasterSunday::EasterSunday(int days_before, int days_after)
      : FloatingHoliday(days_before, days_after) {}

  Date EasterSunday::compute_date(int year) const {
    // This code was copied off the internet from some student's
    // homework assignment.  It was able to reproduce Easter sunday
    // from 2004 to 2015.  It is claimed to work between 1900 and
    // 2600.  One can compare with
    // http://en.wikipedia.org/wiki/Computus#Algorithms or the
    // following section.
    //
    // Args:
    //   year: The four digit year for which Easter Sunday should be
    //     computed.
    if (year <= 1900 || year >= 2600) {
      report_error("Can only compute easter dates between 1900 and 2600.");
    }
    int a, b, c, d, e, day;
    a = year % 19;
    b = year % 4;
    c = year % 7;
    d = (19 * a + 24) % 30;
    e = (2 * b + 4 * c + 6 * d + 5) % 7;
    day = 22 + d + e;
    MonthNames month_name(Mar);
    if (day > 31) {
      month_name = Apr;
      day = d + e - 9;
      if (year == 1954 || year == 1981 || year == 2049 || year == 2076) {
        day = d + e - 16;
      }
    }
    Date ans(month_name, day, year);
    return ans;
  }

  //======================================================================
  MemorialDay::MemorialDay(int days_before, int days_after)
      : LastWeekdayInMonthHoliday(Mon, May, days_before, days_after) {}

  //======================================================================
  // Factory method to create a Holiday given a string containing
  // the holiday name.
  Holiday *CreateNamedHoliday(const std::string &holiday_name,
                              int days_before,
                              int days_after) {
    if (holiday_name == "NewYearsDay") {
      return new NewYearsDay(days_before, days_after);
    } else if (holiday_name == "MartinLutherKingDay") {
      return new MartinLutherKingDay(days_before, days_after);
    } else if (holiday_name == "SuperBowlSunday") {
      return new SuperBowlSunday(days_before, days_after);
    } else if (holiday_name == "PresidentsDay") {
      return new PresidentsDay(days_before, days_after);
    } else if (holiday_name == "ValentinesDay") {
      return new ValentinesDay(days_before, days_after);
    } else if (holiday_name == "SaintPatricksDay") {
      return new SaintPatricksDay(days_before, days_after);
    } else if (holiday_name == "USDaylightSavingsTimeBegins") {
      return new USDaylightSavingsTimeBegins(days_before, days_after);
    } else if (holiday_name == "USDaylightSavingsTimeEnds") {
      return new USDaylightSavingsTimeEnds(days_before, days_after);
    } else if (holiday_name == "EasterSunday") {
      return new EasterSunday(days_before, days_after);
    } else if (holiday_name == "USMothersDay") {
      return new USMothersDay(days_before, days_after);
    } else if (holiday_name == "IndependenceDay") {
      return new IndependenceDay(days_before, days_after);
    } else if (holiday_name == "LaborDay") {
      return new LaborDay(days_before, days_after);
    } else if (holiday_name == "ColumbusDay") {
      return new ColumbusDay(days_before, days_after);
    } else if (holiday_name == "Halloween") {
      return new Halloween(days_before, days_after);
    } else if (holiday_name == "Thanksgiving") {
      return new Thanksgiving(days_before, days_after);
    } else if (holiday_name == "MemorialDay") {
      return new MemorialDay(days_before, days_after);
    } else if (holiday_name == "VeteransDay") {
      return new VeteransDay(days_before, days_after);
    } else if (holiday_name == "Christmas") {
      return new Christmas(days_before, days_after);
    }
    ostringstream err;
    err << "Unknown holiday name passed to CreateHoliday:  " << holiday_name;
    report_error(err.str());
    return NULL;
  }

}  // namespace BOOM
