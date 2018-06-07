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

#include "cpputil/Date.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>
#include <sstream>
#include "cpputil/report_error.hpp"
#include "cpputil/string_utils.hpp"

namespace BOOM {

  std::ostream &operator<<(std::ostream &out, const DayNames &d) {
    if (d == Sat)
      out << "Saturday";
    else if (d == Sun)
      out << "Sunday";
    else if (d == Mon)
      out << "Monday";
    else if (d == Tue)
      out << "Tuesday";
    else if (d == Wed)
      out << "Wednesday";
    else if (d == Thu)
      out << "Thursday";
    else if (d == Fri)
      out << "Friday";
    else {
      report_error("Unknown day name");
    }
    return out;
  }

  Date::Date() : month_(Jan), day_(1), year_(1970), days_after_origin_(0) {
    time_t time_value;
    time(&time_value);
    time_value += local_time_zone_gmt_offset_minutes_ * 60;
    long days = time_value / seconds_in_a_day_;
    set(days);
  }

  Date::Date(int m, int dd, int yyyy) { set(MonthNames(m), dd, yyyy); }

  Date::Date(MonthNames m, int dd, int yyyy) { set(m, dd, yyyy); }

  Date &Date::set(MonthNames m, int dd, int yyyy) {
    check(m, dd, yyyy);
    days_after_origin_ = days_after_jan_1_1970(m, dd, yyyy);
    month_ = m;
    day_ = dd;
    year_ = yyyy;
    return *this;
  }

  Date &Date::set(const tm &rhs) {
    return set(MonthNames(rhs.tm_mon + 1), rhs.tm_mday, rhs.tm_year + 1900);
  }

  Date &Date::set(int days_after_jan_1_1970) {
    if (days_after_jan_1_1970 == 0) return set(Jan, 1, 1970);
    days_after_origin_ = days_after_jan_1_1970;
    if (days_after_jan_1_1970 < 0) {
      return set_before_1970(-days_after_jan_1_1970);
    }
    int days_into_year;
    year_ =
        1970 + years_after_jan_1_1970(days_after_jan_1_1970, &days_into_year);
    bool leap = is_leap_year(year_);
    find_month_and_day(days_into_year, leap, &month_, &day_);
    check(month_, day_, year_);
    return *this;
  }

  Date &Date::set_before_1970(int days_before) {
    if (days_before < 0) return set(-days_before);
    days_after_origin_ = -days_before;
    // year_delta is the number of complete years in days_before days.
    int year_delta = years_before_jan_1_1970(days_before, &days_before);
    year_ = 1970 - year_delta;
    // now days_before is the number of days before the start of year_;
    if (days_before == 0) {
      month_ = Jan;
      day_ = 1;
      return *this;
    }
    --year_;  // Now year_ is the actual year of the Date object.
    bool leap = is_leap_year(year_);
    int days_into_year = 365 + leap - days_before;
    find_month_and_day(days_into_year, leap, &month_, &day_);
    check(month_, day_, year_);
    return *this;
  }

  void Date::find_month_and_day(int days_after_jan1, bool leap,
                                MonthNames *month, int *day) {
    const int *begin =
        leap ? days_before_month_in_leap_year_ : days_before_month_;
    ++begin;
    const int *end = begin + 12;

    const int *pos = std::upper_bound(begin, end, days_after_jan1) - 1;
    *month = MonthNames(pos - begin + 1);
    *day = days_after_jan1 - *pos + 1;
  }

  //----------------------------------------------------------------------
  int Date::years_before_jan_1_1970(int days_before, int *days_remaining) {
    assert(days_before >= 0);
    if (days_before == 0) {
      *days_remaining = 0;
      return 0;
    }
    // lower_bound is a lower bound on the number of full years prior
    // to 1970, the true answer might be even farther back.
    int lower_bound = days_before / 366;

    // 'year' is the year of the time pointer, which is one year
    // farther back than the number of years in lower_bound.
    int year = 1970 - lower_bound - 1;
    *days_remaining =
        days_before -
        365 * (lower_bound)-number_of_leap_years_before_1970(year);

    while (*days_remaining >= 365 + is_leap_year(year)) {
      --year;
      ++lower_bound;
      *days_remaining =
          days_before -
          365 * (lower_bound)-number_of_leap_years_before_1970(year);
    }
    return lower_bound;
  }
  //----------------------------------------------------------------------
  int Date::years_after_jan_1_1970(int days, int *days_remaining) {
    assert(days > 0);
    if (days <= 730) {
      *days_remaining = days % 365;
      return days / 365;
    }
    int lower_bound = days / 366;
    int year = 1970 + lower_bound;
    *days_remaining =
        days - (year - 1970) * 365 - number_of_leap_years_after_1970(year);
    while (*days_remaining >= 365 + is_leap_year(year)) {
      ++year;
      *days_remaining =
          days - (year - 1970) * 365 - number_of_leap_years_after_1970(year);
    }
    return year - 1970;
  }

  int Date::number_of_leap_years_before_1970(int year, bool include_endpoint) {
    assert(year <= 1970);
    if (year >= 1968) return 0;
    int ans = 1 + (1968 - year) / 4;
    if (!include_endpoint) ans -= is_leap_year(year);
    if (year <= 1900) {
      ans -= (2000 - year) / 100;
      ans += (2000 - year) / 400;
    }
    return ans;
  }

  int Date::number_of_leap_years_after_1970(int year, bool include_endpoint) {
    assert(year >= 1970);
    if (year <= 1972) return 0;
    int ans = 1 + (year - 1972) / 4;
    if (!include_endpoint) ans -= is_leap_year(year);
    if (year < 2100) return ans;

    // If year > 2100 then we need to subtract one leap year for every
    // 100 years after 2000, and add one back for every 400 years
    // after 2000.
    ans -= (year - 2000) / 100;
    ans += (year - 2000) / 400;
    return ans;
  }

  int Date::days_after_jan_1_1970(MonthNames month, int day, int year) {
    if (year < 1970) return -1 * days_before_jan_1_1970(month, day, year);
    int ans = 365 * (year - 1970) + number_of_leap_years_after_1970(year);
    ans += days_into_year(month, day, is_leap_year(year));
    // ans includes jan 1, 1970, so subtract it off before returning.
    return ans - 1;
  }

  // Compute the number of days that a particular date is before Jan
  // 1, 1970.
  int Date::days_before_jan_1_1970(MonthNames month, int day, int year) {
    if (year >= 1970) return -1 * days_after_jan_1_1970(month, day, year);

    // Compute the number of days needed to get to Jan 1 of next year
    bool leap = is_leap_year(year);

    // The number of days is the number of days until the end of the
    // year plus the number of days in the complete years between Jan
    // 1 next year and Jan 1 1970.
    int days_to_jan1 = 366 + leap - days_into_year(month, day, leap);
    ++year;
    return days_to_jan1 + 365 * (1970 - year) +
           number_of_leap_years_before_1970(year) + is_leap_year(year);
  }

  Date::Date(const std::string &m, int d, int yyyy) {
    MonthNames month_name = str2month(m);
    set(month_name, d, yyyy);
  }

  Date::Date(const std::string &mdy, char delim) {
    std::vector<std::string> tmp = split_delimited(mdy, delim);
    MonthNames m = str2month(tmp[0]);
    int d, y;
    std::istringstream(tmp[1]) >> d;
    std::istringstream(tmp[2]) >> y;
    set(m, d, y);
  }

  Date::Date(int n) { set(n); }

  Date::Date(const Date &rhs)
      : month_(rhs.month_),
        day_(rhs.day_),
        year_(rhs.year_),
        days_after_origin_(rhs.days_after_origin_) {}

  Date::Date(const struct tm &rhs) { set(rhs); }

  void Date::check(MonthNames month, int day, int year) const {
    if (month < 1 || month > 12) {
     std::ostringstream err;
      err << "Bad month name: " << month << endl;
      report_error(err.str());
    }

    if (day < 1 || day > days_in_month(month, is_leap_year(year))) {
     std::ostringstream err;
      err << "bad dateformat:  " << endl
          << "month = " << month << " day = " << day << " year = " << year;
      report_error(err.str());
    }
  }

  bool Date::is_leap_year() const { return is_leap_year(year()); }

  Date &Date::operator=(const Date &rhs) {
    if (&rhs == this) return *this;
    month_ = rhs.month_;
    day_ = rhs.day_;
    year_ = rhs.year_;
    days_after_origin_ = rhs.days_after_origin_;
    return *this;
  }

  Date &Date::operator=(const struct tm &rhs) { return set(rhs); }

  Date &Date::operator++() {
    ++days_after_origin_;
    ++day_;
    if (day_ > days_in_month(month_, is_leap_year())) {
      if (month_ == Dec) {
        month_ = Jan;
        day_ = 1;
        ++year_;
      } else {
        month_ = MonthNames(month_ + 1);
        day_ = 1;
      }
    }
    return *this;
  }

  Date &Date::operator--() {
    --days_after_origin_;
    --day_;
    if (day_ == 0) {
      if (month_ == Jan) {
        month_ = Dec;
        day_ = 31;
        --year_;
      } else {
        month_ = MonthNames(month_ - 1);
        day_ = days_in_month(month_, is_leap_year());
      }
    }
    return *this;
  }

  Date Date::operator++(int) {
    Date tmp(*this);
    operator++();
    return tmp;
  }

  Date Date::operator--(int) {
    Date tmp(*this);
    operator--();
    return tmp;
  }

  int Date::days_left_in_month() const {
    return days_in_month(month(), is_leap_year()) - day();
  }

  int Date::days_into_year() const {
    return days_into_year(month_, day_, is_leap_year());
  }

  int Date::days_left_in_year() const {
    bool leap = is_leap_year();
    return 365 + leap - days_into_year(month_, day_, leap);
  }

  MonthNames Date::month() const { return month_; }

  int Date::day() const { return day_; }

  int Date::year() const { return year_; }

  DayNames Date::day_of_week() const {
    // Jan 1 1970 was a Thursday.  So computing the day of week means taking the
    // julian date mod 7, then subtracting 4.
    int day = days_after_origin_ % 7;
    return DayNames((day + 4) % 7);
  }

  time_t Date::to_time_t() const {
    time_t ans = days_after_origin_ * seconds_in_a_day_ +
                 local_time_zone_gmt_offset_minutes_ * 60;
    return ans;
  }

  int Date::days_until(DayNames day) const {
    DayNames today = day_of_week();
    if (today <= day)
      return day - today;
    else
      return 7 - int(today - day);
  }

  int Date::days_after(DayNames day) const {
    DayNames today = day_of_week();
    if (today >= day)
      return today - day;
    else
      return 7 + today - day;
  }

  long Date::days_after_jan_1_1970() const { return days_after_origin_; }

  bool Date::operator==(const Date &rhs) const {
    return days_after_origin_ == rhs.days_after_origin_;
  }

  bool Date::operator<(const Date &rhs) const {
    return days_after_origin_ < rhs.days_after_origin_;
  }

  bool Date::operator<=(const Date &rhs) const {
    return days_after_origin_ <= rhs.days_after_origin_;
  }

  bool Date::operator!=(const Date &rhs) const {
    return days_after_origin_ != rhs.days_after_origin_;
  }

  bool Date::operator>(const Date &rhs) const {
    return days_after_origin_ > rhs.days_after_origin_;
  }

  bool Date::operator>=(const Date &rhs) const {
    return days_after_origin_ >= rhs.days_after_origin_;
  }

  Date &Date::start_next_month() {
    // TODO: test this to make sure days_after_origin_ is set correctly.
    days_after_origin_ += (1 + days_left_in_month());
    if (month_ == Dec) {
      ++year_;
      month_ = Jan;
    } else {
      month_ = MonthNames(month_ + 1);
    }
    day_ = 1;
    return *this;
  }

  Date &Date::end_prev_month() {
    days_after_origin_ -= day_;
    if (month_ == Jan) {
      month_ = Dec;
      day_ = 31;
      --year_;
    } else {
      month_ = MonthNames(month_ - 1);
      day_ = days_in_month(month_, is_leap_year());
    }
    return *this;
  }

  Date &Date::operator+=(int n) {
    if (n == 0) return *this;
    if (n < 0) return operator-=(-n);
    days_after_origin_ += n;
    if (n < days_left_in_month())
      day_ += n;
    else
      set(days_after_origin_);
    return *this;
  }

  Date &Date::operator-=(int n) {
    if (n == 0) return *this;
    if (n < 0) return operator+=(-n);
    days_after_origin_ -= n;
    if (n < day_)
      day_ -= n;
    else
      set(days_after_origin_);
    return *this;
  }

  Date Date::operator+(int more_days) const {
    Date ans(*this);
    ans += more_days;
    return ans;
  }

  Date Date::operator-(int days_prior) const {
    Date ans(*this);
    ans -= days_prior;
    return ans;
  }

  int Date::compute_local_time_zone() {
    //    std::cout << "computing local time zone!!!" << std::endl;
    time_t now;
    time(&now);
    struct tm local_time = *localtime(&now);
    int local_minutes_after_midnight =
        local_time.tm_hour * 60 + local_time.tm_min;

    struct tm standard_time = *gmtime(&now);
    int standard_minutes_after_midnight =
        standard_time.tm_hour * 60 + standard_time.tm_min;
    int minutes =
        local_minutes_after_midnight - standard_minutes_after_midnight;

    if (minutes < -12 * 60) {
      minutes += 24 * 60;
    } else if (minutes > 12 * 60) {
      minutes -= 24 * 60;
    }
    return minutes;
  }

  int Date::local_time_zone() { return local_time_zone_gmt_offset_minutes_; }
  void Date::set_local_time_zone(int time_zone_offset_in_minutes) {
    local_time_zone_gmt_offset_minutes_ = time_zone_offset_in_minutes;
  }

  int Date::local_time_zone_gmt_offset_minutes_(
      Date::compute_local_time_zone());
  Date::print_order Date::po(mdy);
  Date::date_format Date::df(slashes);
  calendar_format Date::month_format(Abbreviations);
  calendar_format Date::day_format(Abbreviations);
  const int Date::seconds_in_a_day_(60 * 60 * 24);
  const int Date::seconds_in_an_hour_(60 * 60);

  const int Date::days_in_month_[] = {0,  31, 28, 31, 30, 31, 30,
                                      31, 31, 30, 31, 30, 31};

  const int Date::days_before_month_[13] = {
      0,                                                      // NA
      0,                                                      // Jan
      31,                                                     // Feb
      31 + 28,                                                // Mar
      31 + 28 + 31,                                           // Apr
      31 + 28 + 31 + 30,                                      // May
      31 + 28 + 31 + 30 + 31,                                 // Jun
      31 + 28 + 31 + 30 + 31 + 30,                            // Jul
      31 + 28 + 31 + 30 + 31 + 30 + 31,                       // Aug
      31 + 28 + 31 + 30 + 31 + 30 + 31 + 31,                  // Sep
      31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30,             // Oct
      31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31,        // Nov
      31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30};  // Dec

  const int Date::days_before_month_in_leap_year_[13] = {
      0,                                                      // NA
      0,                                                      // Jan
      31,                                                     // Feb
      31 + 29,                                                // Mar
      31 + 29 + 31,                                           // Apr
      31 + 29 + 31 + 30,                                      // May
      31 + 29 + 31 + 30 + 31,                                 // Jun
      31 + 29 + 31 + 30 + 31 + 30,                            // Jul
      31 + 29 + 31 + 30 + 31 + 30 + 31,                       // Aug
      31 + 29 + 31 + 30 + 31 + 30 + 31 + 31,                  // Sep
      31 + 29 + 31 + 30 + 31 + 30 + 31 + 31 + 30,             // Oct
      31 + 29 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31,        // Nov
      31 + 29 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30};  // Dec

  void Date::set_month_format(calendar_format f) { Date::month_format = f; }
  void Date::set_day_format(calendar_format f) { Date::day_format = f; }
  void Date::set_date_format(Date::date_format f) { Date::df = f; }

  std::ostream &Date::display(std::ostream &out) const {
    if (df == script) {
      if (po == mdy) {
        display_month(out);
        out << " " << day() << "," << year();
      } else if (po == dmy) {
        out << day() << " ";
        display_month(out);
        out << ", " << year();
      } else if (po == ymd) {
        out << year() << ", ";
        display_month(out);
        out << day();
      }
      return out;
    }

    char delim(' ');
    if (df == slashes) delim = '/';
    if (df == dashes) delim = '-';

    if (po == mdy) {
      display_month(out);
      out << delim << day() << delim << year();
    } else if (po == dmy) {
      out << day() << delim;
      display_month(out);
      out << delim << year();
    } else if (po == ymd) {
      out << year() << delim;
      display_month(out);
      out << delim << day();
    }
    return out;
  }

  void Date::set_print_order(Date::print_order f) { po = f; }

  std::ostream &display(std::ostream &out, DayNames d, calendar_format f) {
    if (f == Full) {
      static const char *Days[] = {"Sunday",   "Monday", "Tuesday", "Wednesday",
                                   "Thursday", "Friday", "Saturday"};
      out << Days[d];
    } else if (f == full) {
      static const char *days[] = {"sunday",   "monday", "tuesday", "wednesday",
                                   "thursday", "friday", "saturday"};
      out << days[d];
    } else if (f == Abbreviations) {
      static const char *Ds[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
      out << Ds[d];
    } else if (f == abbreviations) {
      static const char *ds[] = {"sun", "mon", "tue", "wed", "thu", "fri", "sat"};
      out << ds[d];
    } else if (f == numeric) {
      uint tmp(d);
      out << tmp;
    } 
    return out;
  }

  std::ostream &Date::display_month(std::ostream &out) const {
    if (month_format == Full) {
      static const char *Month_names[] = {
        "",        "January",  "February", "March",  "April",
        "May",     "June",     "July",     "August", "September",
        "October", "November", "December"};
      out << Month_names[month()];
    } else if (month_format == full) {
      static const char *month_names[] = {
        "",        "january",  "february", "march",  "april",
        "may",     "june",     "july",     "august", "september",
        "october", "november", "december"};
      out << month_names[month()];
    } else if (month_format == Abbreviations) {
      static const char *Month_abbrevs[] = {
        "",    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
        "Oct", "Nov", "Dec"};
      out << Month_abbrevs[month()];
    } else if (month_format == abbreviations) {
      static const char *month_abbrevs[] = {
        "",    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
        "oct", "nov", "dec"};
      out << month_abbrevs[month()];
    } else {
      out << int(month());
    }
    return out;
  }

  Date guess_date_format(const std::string &s, char delim) {
    std::vector<std::string> fields = split_delimited(s, delim);
    int m, d, y;
    std::istringstream(fields[0]) >> m;
    std::istringstream(fields[1]) >> d;
    std::istringstream(fields[2]) >> y;

    if (y <= 31) {
      if (m > 12)
        std::swap(y, m);
      else if (d > 31)
        std::swap(y, d);
      else {
       std::ostringstream err;
        err << "Error in guess_date_format: " << endl
            << "called with argument: " << s << endl
            << "and delimiter = [" << delim << "]" << endl
            << "m = " << m << " d = " << d << " y = " << y;
        report_error(err.str());  // year <=31, but nothing to swap it with
      }
    }  // now year is okay;
    assert(y > 31);
    if (m > 12) std::swap(d, m);

    assert(m <= 12 && m >= 1 && d >= 1 &&
           d <= Date::days_in_month(MonthNames(m), Date::is_leap_year(y)));
    return Date(m, d, y);
  }

  std::string Date::str() const {
    std::ostringstream os;
    os << *this;
    return os.str();
  }

  std::ostream &operator<<(std::ostream &out, const Date &d) {
    d.display(out);
    return out;
  }

  MonthNames str2month(const std::string &m) {
    if (m == "January" || m == "january" || m == "Jan" || m == "jan" ||
        m == "01" || m == "1")
      return Jan;
    if (m == "February" || m == "february" || m == "Feb" || m == "feb" ||
        m == "02" || m == "2")
      return Feb;
    if (m == "March" || m == "march" || m == "Mar" || m == "mar" || m == "03" ||
        m == "3")
      return Mar;
    if (m == "April" || m == "april" || m == "Apr" || m == "apr" || m == "04" ||
        m == "4")
      return Apr;
    if (m == "May" || m == "may" || m == "05" || m == "5") return May;
    if (m == "June" || m == "june" || m == "Jun" || m == "jun" || m == "06" ||
        m == "6")
      return Jun;
    if (m == "July" || m == "july" || m == "Jul" || m == "jul" || m == "07" ||
        m == "7")
      return Jul;
    if (m == "August" || m == "august" || m == "Aug" || m == "aug" ||
        m == "08" || m == "8")
      return Aug;
    if (m == "September" || m == "september" || m == "Sep" || m == "sep" ||
        m == "09" || m == "9")
      return Sep;
    if (m == "October" || m == "october" || m == "Oct" || m == "oct" ||
        m == "10")
      return Oct;
    if (m == "November" || m == "november" || m == "Nov" || m == "nov" ||
        m == "11")
      return Nov;
    if (m == "December" || m == "december" || m == "Dec" || m == "dec" ||
        m == "12")
      return Dec;
    std::ostringstream err;
    err << "unkown month name: " << m;
    report_error(err.str());
    return unknown_month;
  }

  DayNames str2day(const std::string &s) {
    if (s.size() <= 4) {
      if (s == "Sun" || s == "sun") return Sun;
      if (s == "Mon" || s == "mon") return Mon;
      if (s == "Tue" || s == "tue") return Tue;
      if (s == "Wed" || s == "wed") return Wed;
      if (s == "Thu" || s == "thu") return Thu;
      if (s == "Fri" || s == "fri") return Fri;
      if (s == "Sat" || s == "sat") return Sat;
    } else {
      if (s == "Sunday" || s == "sunday") return Sun;
      if (s == "Monday" || s == "monday") return Mon;
      if (s == "Tuesday" || s == "tuesday") return Tue;
      if (s == "Wednesday" || s == "wednesday") return Wed;
      if (s == "Thursday" || s == "thursday") return Thu;
      if (s == "Friday" || s == "friday") return Fri;
      if (s == "Saturday" || s == "saturday") return Sat;
    }
    std::ostringstream err;
    err << "Unrecognized day name: " << s;
    report_error(err.str());
    return Sun;  // to keep the compiler quiet.
  }

  int operator-(const Date &d1, const Date &d2) {
    return d1.days_after_jan_1_1970() - d2.days_after_jan_1_1970();
  }

  //============================================================
  Date nth_weekday_in_month(int n, DayNames weekday, MonthNames month,
                            int year) {
    if (n < 1) report_error("n must be >= 1 in nth_weekday_in_month");
    Date ans(month, 1, year);
    int days_to = ans.days_until(weekday);
    ans += days_to + 7 * (n - 1);
    if (ans.month() != month) {
      std::ostringstream err;
      err << "n is too large in nth_weekday_in_month.  There are not " << n
          << " " << weekday << "s in " << month << " in " << year << ".";
      report_error(err.str());
    }
    return ans;
  }

  //============================================================
  Date last_weekday_in_month(DayNames day, MonthNames month, int year) {
    Date ans(month, Date::days_in_month(month, Date::is_leap_year(year)), year);
    return ans - ans.days_after(day);
  }
}  // namespace BOOM
