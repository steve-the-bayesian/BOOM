// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#ifndef BOOM_HOLIDAY_HPP_
#define BOOM_HOLIDAY_HPP_

#include <map>
#include <vector>
#include "cpputil/Date.hpp"
#include "cpputil/RefCounted.hpp"

namespace BOOM {
  //===========================================================================
  // A Holiday is a recurring Date.  It differs from a standard "season" in that
  // holidays can sometimes move, either because of complicated religious logic
  // (e.g. Easter), or because of calendar effects (e.g. when Independence Day
  // falls on a weekend people get the closest Monday or Friday off).
  //
  // A Holiday is defined in terms of a window, specified as the Date of the
  // holiday, as well as some number of days before or after.  This window might
  // be of different width each year, as holidays sometimes interact with
  // weekends and other holidays in strange ways.
  class Holiday : private RefCounted {
   public:

    virtual ~Holiday() {}

    // Returns the number of days that 'arbitrary_date' is into the holiday's
    // influence window.  If arbitrary_date is not in the influence window then
    // -1 is returned.
    int days_into_influence_window(const Date &arbitrary_date) const {
      if (active(arbitrary_date)) {
        return arbitrary_date - earliest_influence(arbitrary_date);
      } else {
        return -1;
      }
    }

    // Holidays can sometimes (or will usually) exert an influence before or
    // after the date of the actual holiday.  The number of days from the
    // earliest influenced day to the last influenced day (including the end
    // points) is the maximum_window_width.
    virtual int maximum_window_width() const = 0;

    // Indicates whether this holiday is active on the given date.
    virtual bool active(const Date &arbitrary_date) const = 0;

   protected:
    // The dates of earliest and latest influence for a holiday occurring on
    // 'holiday_date' meaning that holiday_date is a date in the influence
    // interval for the holiday.  If holiday_date is not in the influence
    // interval for a holiday then Jan 1, -1000000 is returned.  That's 1
    // million years BC for Raquel Welch fans!!

    virtual Date earliest_influence(const Date &holiday_date) const = 0;
    virtual Date latest_influence(const Date &holiday_date) const = 0;

   private:
    friend void intrusive_ptr_add_ref(Holiday *h) { h->up_count(); }
    friend void intrusive_ptr_release(Holiday *h) {
      h->down_count();
      if (h->ref_count() == 0) {
        delete h;
      }
    }
  };

  // A SingleDayHoliday is a holiday associated with a specific date.  Its
  // influence can extend beyond that date, but (e.g.) February 14 is
  // Valentine's day.  Most Holidays are SingleDayHolidays, some religious
  // holidays (e.g. Passover) and some sporting events (e.g. the Olympics or the
  // World Cup) are not.
  class SingleDayHoliday : public Holiday {
   public:
    // The first incidence of the holiday ON or AFTER the given date.  Returns a
    // Date object of 'arbitrary_date' or later.
    virtual Date date_on_or_after(const Date &arbitrary_date) const = 0;

    // The last incidence of the holiday ON or BEFORE the given date.  Returns a
    // Date object of 'arbitrary_date' or before.
    virtual Date date_on_or_before(const Date &arbitrary_date) const = 0;

    // The date of the closest holiday to 'arbitrary_date'.
    virtual Date nearest(const Date &arbitrary_date) const;

    bool active(const Date &d) const override;
  };

  // A factory function that will create a holiday based on its name.
  // Args:
  //   holiday_name: The name of the holiday.  It is an error to ask
  //     for an unrecognized holiday.
  //   days_before: The number of days before the date of the actual
  //     holiday that the holiday's influence can be felt.
  //   days_after: The number of days after the date of the actual
  //     holiday that the holiday's influence can be felt.
  // Returns:
  //   A heap-allocated pointer to the requested holiday.  The caller
  //   is responsible for deleting the returned object.
  Holiday *CreateNamedHoliday(const std::string &holiday_name,
                              int days_before,
                              int days_after);

  //==========================================================================
  // An OrdinaryAnnualHoliday is a Holiday that occurs once per year, with a
  // fixed-sized window of influence.  An OrdinaryAnnualHoliday keeps track of
  // two integers: days_before and days_after, that define its influence window.
  class OrdinaryAnnualHoliday : public SingleDayHoliday {
   public:
    OrdinaryAnnualHoliday(int days_before, int days_after);
    int maximum_window_width() const override;
    Date date_on_or_after(const Date &d) const override;
    Date date_on_or_before(const Date &d) const override;

    // The date the holiday occurs on a given year.  For floating holidays, the
    // date() function might be expensive to compute over and over again, so we
    // defer computation to a rarely called function compute_date(), and store
    // the results in a table.  This class implements the table logic, and
    // requires its children to implement compute_date().
    virtual Date date(int year) const;

    // Compute the date of this holiday in the given year.
    virtual Date compute_date(int year) const = 0;

   protected:
    Date earliest_influence(const Date &holiday_date) const override;
    Date latest_influence(const Date &holiday_date) const override;

   private:
    int days_before_;
    int days_after_;
    typedef int Year;
    mutable std::map<Year, Date> date_lookup_table_;
    mutable std::map<Year, Date> earliest_influence_by_year_;
    mutable std::map<Year, Date> latest_influence_by_year_;
  };

  //======================================================================
  // A FixedDateHoliday is a Holiday that occurs on the same date each
  // year.
  class FixedDateHoliday : public OrdinaryAnnualHoliday {
   public:
    // month is an integer between 1 and 12.
    FixedDateHoliday(int month, int day_of_month, int days_before = 1,
                     int days_after = 1);
    Date compute_date(int year) const override;

   private:
    // MonthNames is an enum in the range 1:12 defined in Date.hpp
    const MonthNames month_name_;
    const int day_of_month_;
  };

  //======================================================================
  // An NthWeekdayInMonthHoliday is an OrdinaryAnnualHoliday defined
  // as the n'th weekday in a month.  For example, Thanksgiving is the
  // 4th Thursday in November.
  class NthWeekdayInMonthHoliday : public OrdinaryAnnualHoliday {
   public:
    NthWeekdayInMonthHoliday(int which_week, DayNames day, MonthNames month,
                             int days_before, int days_after);
    Date compute_date(int year) const override;

   private:
    int which_week_;
    DayNames day_name_;
    MonthNames month_name_;
  };

  //======================================================================
  // An LastWeekdayInMonthHoliday is an OrdinaryAnnualHoliday defined
  // as the last weekday in a month.  For example, Memorial Day is the
  // last Monday in May.
  class LastWeekdayInMonthHoliday : public OrdinaryAnnualHoliday {
   public:
    LastWeekdayInMonthHoliday(DayNames day, MonthNames month, int days_before,
                              int days_after);
    Date compute_date(int year) const override;

   private:
    DayNames day_name_;
    MonthNames month_name_;
  };

  //======================================================================
  // A floating holiday is a holiday that does not occur on the same
  // date each year.  Children of this class must define their own
  // compute_date function.
  class FloatingHoliday : public OrdinaryAnnualHoliday {
   public:
    FloatingHoliday(int days_before, int days_after);
  };

  //===========================================================================
  // A holiday defined by arbitrary date ranges.
  class DateRangeHoliday : public Holiday {
   public:
    // Date ranges will need to be added using add_dates.
    DateRangeHoliday();

    // Args:
    //   begin: The start date of each holiday's influence period.  Elements
    //     must be in increasing order.
    //   end: The end date of each holiday's influence period.  Must have the
    //     same number of elements as begin, and end[i] >= begin[i].
    DateRangeHoliday(const std::vector<Date> &begin,
                     const std::vector<Date> &end);

    // Add a date range for specific incidences of the holiday.
    // Args:
    //   begin:  The first date of influence for this instance of the holiday.
    //   end;  The final date of influence for this instance of the holiday.
    //
    // Example:
    //   In 2016 the super bowl was played on Sunday, Feb 7.  If we model the
    //   super bowl influence as starting on Friday and ending on Monday, then
    //   add_dates(Date(Feb, 5, 2016), Date(Feb, 8, 2016)) would add the 2016
    //   super bowl.  Repeat for other years in the data set.  Add years in
    //   order.
    void add_dates(const Date &begin, const Date &end);

    int maximum_window_width() const override { return maximum_window_width_; }
    bool active(const Date &arbitrary_date) const override;

   protected:
    Date earliest_influence(const Date &holiday_date) const override;
    Date latest_influence(const Date &holiday_date) const override;

   private:
    std::vector<Date> begin_;
    std::vector<Date> end_;
    int maximum_window_width_;
  };

  //----------------------------------------------------------------------
  // Specific holidays observed in the US
  class NewYearsDay : public FixedDateHoliday {
   public:
    NewYearsDay(int days_before, int days_after)
        : FixedDateHoliday(Jan, 1, days_before, days_after) {}
  };

  class MartinLutherKingDay : public NthWeekdayInMonthHoliday {
   public:
    MartinLutherKingDay(int days_before, int days_after)
        : NthWeekdayInMonthHoliday(3, Mon, Jan, days_before, days_after) {}
  };

  class SuperBowlSunday : public FloatingHoliday {
   public:
    SuperBowlSunday(int days_before, int days_after);
    Date compute_date(int year) const override;
  };

  class PresidentsDay : public NthWeekdayInMonthHoliday {
   public:
    PresidentsDay(int days_before, int days_after)
        : NthWeekdayInMonthHoliday(3, Mon, Feb, days_before, days_after) {}
  };

  class ValentinesDay : public FixedDateHoliday {
   public:
    ValentinesDay(int days_before, int days_after)
        : FixedDateHoliday(Feb, 14, days_before, days_after) {}
  };

  class SaintPatricksDay : public FixedDateHoliday {
   public:
    SaintPatricksDay(int days_before, int days_after)
        : FixedDateHoliday(Mar, 17, days_before, days_after) {}
  };

  class USDaylightSavingsTimeBegins : public FloatingHoliday {
   public:
    USDaylightSavingsTimeBegins(int days_before, int days_after);
    Date compute_date(int year) const override;
  };

  class USDaylightSavingsTimeEnds : public FloatingHoliday {
   public:
    USDaylightSavingsTimeEnds(int days_before, int days_after);
    Date compute_date(int year) const override;
  };

  class EasterSunday : public FloatingHoliday {
   public:
    EasterSunday(int days_before, int days_after);
    Date compute_date(int year) const override;
  };

  // The US definition of Mother's day: second Sunday in May.
  class USMothersDay : public NthWeekdayInMonthHoliday {
   public:
    USMothersDay(int days_before, int days_after)
        : NthWeekdayInMonthHoliday(2, Sun, May, days_before, days_after) {}
  };

  class MemorialDay : public LastWeekdayInMonthHoliday {
   public:
    MemorialDay(int days_before, int days_after);
  };

  class IndependenceDay : public FixedDateHoliday {
   public:
    IndependenceDay(int days_before, int days_after)
        : FixedDateHoliday(Jul, 4, days_before, days_after) {}
  };

  class LaborDay : public NthWeekdayInMonthHoliday {
   public:
    LaborDay(int days_before, int days_after)
        : NthWeekdayInMonthHoliday(1, Mon, Sep, days_before, days_after) {}
  };

  class ColumbusDay : public NthWeekdayInMonthHoliday {
   public:
    ColumbusDay(int days_before, int days_after)
        : NthWeekdayInMonthHoliday(2, Mon, Oct, days_before, days_after) {}
  };

  class Halloween : public FixedDateHoliday {
   public:
    Halloween(int days_before, int days_after)
        : FixedDateHoliday(Oct, 31, days_before, days_after) {}
  };

  class VeteransDay : public FixedDateHoliday {
   public:
    VeteransDay(int days_before, int days_after)
        : FixedDateHoliday(Nov, 11, days_before, days_after) {}
  };

  class Thanksgiving : public NthWeekdayInMonthHoliday {
   public:
    Thanksgiving(int days_before, int days_after)
        : NthWeekdayInMonthHoliday(4, Thu, Nov, days_before, days_after) {}
  };

  // Note that there can be very different numbers of shopping days
  // between Thanksgiving and Christmas in different years.
  class Christmas : public FixedDateHoliday {
   public:
    Christmas(int days_before, int days_after)
        : FixedDateHoliday(Dec, 25, days_before, days_after) {}
  };

}  // namespace BOOM

#endif  // BOOM_HOLIDAY_HPP_
