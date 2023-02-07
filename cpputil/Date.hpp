// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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

#ifndef BOOM_DATE_HPP
#define BOOM_DATE_HPP

#include <string>
#include <ctime>
#include "uint.hpp"
#include <ctime>

namespace BOOM {

  // Starting with Jan=1 helps keep things sane, but puts everything
  // off by 1 from the tm_mon field in struct tm from <ctime>.
  enum MonthNames {
    unknown_month = 0,
    Jan = 1,
    Feb,
    Mar,
    Apr,
    May,
    Jun,
    Jul,
    Aug,
    Sep,
    Oct,
    Nov,
    Dec
  };
  inline MonthNames next(MonthNames month) {
    return month < Dec ? MonthNames(month + 1) : Jan;
  }

  // Starting with Sun=0 matches struct tm::wday in <ctime>.
  enum DayNames { Sun = 0, Mon, Tue, Wed, Thu, Fri, Sat };
  inline DayNames next(DayNames day) {
    return day < Sat ? DayNames(day + 1) : Sun;
  }

  enum calendar_format { Full, full, Abbreviations, abbreviations, numeric };

  // Args:
  //   month_name: The following formats are acceptable
  //   * Full month name (either with initial caps or all lower case)
  //   * Three letter abbreviation (either initial caps or all lower case).
  //   * Month number (with January = 1, February = 2, etc).  A
  //     leading "0" for months prior to October is optional.
  // Returns:
  //   The enum corresponding to the month in the input string.
  MonthNames str2month(const std::string &month_name);

  // Args:
  //   day_name: Name of the day of the week.  The following formats
  //     are acceptable:
  //   * Full day name (initial caps or all lower case),
  //   * Three letter abbreviation (initial caps or all lower case).
  // Returns:
  //   The enum corresponding to the day of the week in the input string.
  DayNames str2day(const std::string &day_name);

  std::ostream &operator<<(std::ostream &, const DayNames &);

  class Date {
   public:
    enum print_order { mdy, dmy, ymd };
    enum date_format { slashes, dashes, script };

    Date();                                    // 'today'
    explicit Date(int days_after_jan_1_1970);  // Unix time, but in days
    Date(int m, int dd, int yyyy);       // January 3, 2007 is Date(1, 3, 2007)
    Date(MonthNames m, int dd, int yyyy);      // Date(Jan, 3, 2007)
    explicit Date(const std::string &mdy, char delim = '/');  // "Jan/3/2007"
    Date(const std::string &m, int d, int yyyy);  // Date("January", 3, 2007)
    Date(const Date &rhs);
    explicit Date(const struct tm &time_info);

    Date &operator=(const Date &rhs);
    Date &operator=(const struct tm &rhs);
    Date &set(const tm &rhs);
    Date &set(MonthNames month, int day, int four_digit_year);
    Date &set(int days_after_jan_1_1970);

    Date &operator++();       // next day
    Date operator++(int);     // next day (postfix)
    Date &operator--();       // previous day
    Date operator--(int);     // previous day (postfix);
    Date &operator+=(int n);  // add n days
    Date &operator-=(int n);  // subtract n days

    Date operator+(int n) const;
    Date operator-(int n) const;

    bool operator==(const Date &rhs) const;  // comparison operators
    bool operator!=(const Date &rhs) const;
    bool operator<(const Date &rhs) const;
    bool operator>(const Date &rhs) const;
    bool operator<=(const Date &rhs) const;
    bool operator>=(const Date &rhs) const;

    // Accessors
    MonthNames month() const;
    int day() const;
    int year() const;

    DayNames day_of_week() const;
    int days_until(DayNames day) const;
    int days_after(DayNames day) const;
    int days_left_in_month() const;  // jan 31 = 0, jan1=30
    int days_into_year() const;      // jan 1 is 1
    int days_left_in_year() const;   // dec 31 = 0; jan 1 = 364

    bool is_leap_year() const;
    long days_after_jan_1_1970() const;
    std::ostream &display(std::ostream &) const;
    std::ostream &display_month(std::ostream &) const;
    std::string str() const;

    time_t to_time_t() const;

    //---------------public static members below this line ------------------
    // These could also be free functions in an appropriate namespace.
    static void set_month_format(calendar_format f);
    static void set_day_format(calendar_format f);
    static void set_print_order(print_order d);
    static void set_date_format(date_format f);

    static bool is_leap_year(int yyyy) {
      // Divisible by 4, not if divisible by 100, but true if divisible
      // by 400.
      return (!(yyyy % 4)) && ((yyyy % 100) || (!(yyyy % 400)));
    }
    static int days_after_jan_1_1970(MonthNames month, int day, int year);
    static int days_before_jan_1_1970(MonthNames month, int day, int year);

    static int days_in_month(MonthNames month, bool leap_year = false) {
      if (month == Feb) return leap_year ? 29 : 28;
      return days_in_month_[month];
    }

    // The number of days (month,day) is into the year.  Unit based,
    // so Jan 1 is 1.
    static int days_into_year(MonthNames month, int day, bool leap) {
      int ans = leap ? days_before_month_in_leap_year_[month] + day
                     : days_before_month_[month] + day;
      return ans;
    }

    // The local time zone is the time zone of the computer on which
    // the program is run.  It is used only for the default
    // constructor.
    //
    // Args: minutes_after_gmt: The number of minutes difference
    //   between current time in the local time zone and current time
    //   gmt.
    static void set_local_time_zone(int minutes_after_gmt);

    // The number of minutes after gmt in the local time zone.  The
    // unit is minutes instead of hours because some time
    // zones are half an hour or 45 minutes off.
    static int local_time_zone();

    // Returns the number of full years since Jan 1, 1970 that have
    // been exhausted by the first argument.  The second argument
    // returns the number of days AFTER Jan 1 of the following year.
    static int years_after_jan_1_1970(int days_after_jan_1_1970,
                                      int *days_remaining);
    static int years_before_jan_1_1970(int days_before_jan_1_1970,
                                       int *days_remaining);

    // The number of leap years in [1970, year).
    static int number_of_leap_years_after_1970(int year,
                                               bool include_endpoint = false);
    // The number of leap years in (year, 1970].
    static int number_of_leap_years_before_1970(int year,
                                                bool include_endpoint = false);
    static void find_month_and_day(int days_into_year, bool leap,
                                   MonthNames *month, int *day);

   private:
    MonthNames month_;
    int day_;
    int year_;
    long days_after_origin_;  // Origin is Jan 1, 1970

    static calendar_format month_format;
    static calendar_format day_format;
    static date_format df;
    static print_order po;
    static const int seconds_in_a_day_;
    static const int seconds_in_an_hour_;

    // The number of minutes past GMT in the local time zone.  Minutes
    // are used instead of hours in order to handle time zones that
    // differ from GMT by an additional 30 or 45 minutes.
    static int local_time_zone_gmt_offset_minutes_;
    static const int days_in_month_[13];
    static const int days_before_month_[13];
    static const int days_before_month_in_leap_year_[13];

    Date &start_next_month();
    Date &end_prev_month();
    Date &set_before_1970(int days_before_jan_1_1970);

    void check(MonthNames month, int day, int four_digit_year) const;
    static int compute_local_time_zone();
  };
  //======================================================================
  int operator-(const Date &d1, const Date &d2);

  // To compute (e.g.) the third Monday in June, 1972:
  // nth_day_in_month(3, Mon, Jun, 1972);
  Date nth_weekday_in_month(int n, DayNames weekday, MonthNames month,
                            int year);

  // To compute (e.g.) the last Friday in Feb 2006:
  // last_weekday_in_month(Fri, Feb, 2006);
  Date last_weekday_in_month(DayNames weekday, MonthNames month, int year);

  std::ostream &operator<<(std::ostream &, const Date &d);
  std::ostream &display(std::ostream &, DayNames,
                        calendar_format = Abbreviations);
  Date guess_date_format(const std::string &s, char delim = '/');
}  // namespace BOOM
#endif  // BOOM_DATE_HPP
