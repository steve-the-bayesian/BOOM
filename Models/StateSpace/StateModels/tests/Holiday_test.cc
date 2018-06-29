#include "gtest/gtest.h"
#include "Models/StateSpace/StateModels/Holiday.hpp"
#include "cpputil/Date.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  class HolidayTest : public ::testing::Test {
   protected:
    HolidayTest() {
    }
  };

  TEST_F(HolidayTest, NewYears) {
    NewYearsDay nyd(2, 1);

    Date before_nyd(Dec, 31, 2014);
    EXPECT_EQ(nyd.nearest(before_nyd),
              Date(Jan, 1, 2015));
    EXPECT_EQ(nyd.date_on_or_after(before_nyd),
              Date(Jan, 1, 2015));
    EXPECT_EQ(nyd.date_on_or_before(before_nyd),
              Date(Jan, 1, 2014));

    EXPECT_EQ(4, nyd.maximum_window_width());

    EXPECT_FALSE(nyd.active(Date(Dec, 29, 2011)));
    EXPECT_TRUE(nyd.active(Date(Dec, 30, 2011)));
    EXPECT_TRUE(nyd.active(Date(Dec, 31, 2011)));
    EXPECT_TRUE(nyd.active(Date(Jan, 1, 2012)));
    EXPECT_TRUE(nyd.active(Date(Jan, 2, 2012)));
    EXPECT_FALSE(nyd.active(Date(Jan, 3, 2012)));

    EXPECT_EQ(nyd.date_on_or_after(Date(Sep, 3, 2004)),
              Date(Jan, 1, 2005));
    EXPECT_EQ(nyd.date_on_or_after(Date(Jan, 1, 2005)),
              Date(Jan, 1, 2005));
    EXPECT_EQ(nyd.date_on_or_after(Date(Jan, 2, 2005)),
              Date(Jan, 1, 2006));

    EXPECT_EQ(-1,
              nyd.days_into_influence_window(Date(Dec, 29, 2011)));
    EXPECT_EQ(0,
              nyd.days_into_influence_window(Date(Dec, 30, 2011)));
    EXPECT_EQ(1,
              nyd.days_into_influence_window(Date(Dec, 31, 2011)));
    
    EXPECT_EQ(nyd.date(2004), Date(Jan, 1, 2004));
    EXPECT_EQ(nyd.date(2007), Date(Jan, 1, 2007));
  }

  TEST_F(HolidayTest, Easter) {
    EasterSunday easter(3, 1);

    // In 2014 easter was April 20.
    Date easter_2014(Apr, 20, 2014);
    Date before_easter = easter_2014 - 1;
    EXPECT_EQ(easter.nearest(before_easter),
              easter_2014);
    EXPECT_EQ(easter.nearest(easter_2014 + 1),
              easter_2014);
    EXPECT_EQ(easter.date_on_or_after(easter_2014),
              easter_2014);
    EXPECT_EQ(easter.date_on_or_after(easter_2014 - 1),
              easter_2014);
    EXPECT_EQ(easter.date_on_or_after(easter_2014 - 100),
              easter_2014);

    Date easter_2015(Apr, 5, 2015);
    EXPECT_EQ(easter.date_on_or_after(easter_2014 + 1),
              easter_2015);
    EXPECT_EQ(easter.date_on_or_before(easter_2015 - 1),
              easter_2014);

    EXPECT_EQ(5, easter.maximum_window_width());

    EXPECT_FALSE(easter.active(easter_2014 - 4));
    EXPECT_TRUE(easter.active(easter_2014 - 3));
    EXPECT_TRUE(easter.active(easter_2014 - 2));
    EXPECT_TRUE(easter.active(easter_2014 - 1));
    EXPECT_TRUE(easter.active(easter_2014));
    EXPECT_TRUE(easter.active(easter_2014 + 1));
    EXPECT_FALSE(easter.active(easter_2014 + 2));
    EXPECT_FALSE(easter.active(easter_2014 + 230));

    EXPECT_EQ(-1,
              easter.days_into_influence_window(easter_2014 - 4));
    EXPECT_EQ(0,
              easter.days_into_influence_window(easter_2014 - 3));
    EXPECT_EQ(1,
              easter.days_into_influence_window(easter_2014 - 2));
    EXPECT_EQ(2,
              easter.days_into_influence_window(easter_2014 - 1));
    EXPECT_EQ(3,
              easter.days_into_influence_window(easter_2014));
    EXPECT_EQ(4,
              easter.days_into_influence_window(easter_2014 + 1));
    EXPECT_EQ(-1,
              easter.days_into_influence_window(easter_2014 + 2));
    
    EXPECT_EQ(easter.date(2014), easter_2014);
    EXPECT_EQ(easter.date(2015), easter_2015);
  }

  // MartinLutherKingDay is an NthWeekdayInMonth holiday.
  TEST_F(HolidayTest, MlkTest) {
    MartinLutherKingDay mlk(1, 1);
    Date mlk_2004(Jan, 19, 2004);
    Date mlk_2005(Jan, 17, 2005);
    EXPECT_EQ(mlk_2004.day_of_week(), Mon);
    EXPECT_EQ(mlk_2005.day_of_week(), Mon);
    EXPECT_EQ(mlk.nearest(Date(Mar, 10, 2004)), mlk_2004);
    EXPECT_EQ(mlk.nearest(mlk_2004), mlk_2004);
    EXPECT_EQ(mlk.nearest(Date(Sep, 10, 2004)), mlk_2005);
    EXPECT_EQ(mlk.nearest(mlk_2005), mlk_2005);

    EXPECT_FALSE(mlk.active(mlk_2004 - 2));
    EXPECT_TRUE(mlk.active(mlk_2004 - 1));
    EXPECT_TRUE(mlk.active(mlk_2004));
    EXPECT_TRUE(mlk.active(mlk_2004 + 1));
    EXPECT_FALSE(mlk.active(mlk_2004 + 2));

    EXPECT_EQ(mlk.date_on_or_after(mlk_2004 - 9), mlk_2004);
    EXPECT_EQ(mlk.date_on_or_after(mlk_2004 - 1), mlk_2004);
    EXPECT_EQ(mlk.date_on_or_after(mlk_2004), mlk_2004);
    EXPECT_EQ(mlk.date_on_or_after(mlk_2004 + 1), mlk_2005);

    EXPECT_EQ(mlk.date_on_or_before(mlk_2004), mlk_2004);
    EXPECT_EQ(mlk.date_on_or_before(mlk_2004 + 1), mlk_2004);
    EXPECT_EQ(mlk.date_on_or_before(mlk_2004 + 300), mlk_2004);
    EXPECT_EQ(mlk.date_on_or_before(mlk_2005 - 1), mlk_2004);

    EXPECT_EQ(mlk.days_into_influence_window(mlk_2004 - 2), -1);
    EXPECT_EQ(mlk.days_into_influence_window(mlk_2004 - 1), 0);
    EXPECT_EQ(mlk.days_into_influence_window(mlk_2004), 1);
    EXPECT_EQ(mlk.days_into_influence_window(mlk_2004 + 1), 2);
    EXPECT_EQ(mlk.days_into_influence_window(mlk_2004 + 2), -1);
    EXPECT_EQ(mlk.days_into_influence_window(mlk_2004 + 200), -1);

    EXPECT_EQ(mlk.date(2004), mlk_2004);
    EXPECT_EQ(mlk.date(2005), mlk_2005);
  }

  // MemorialDay is a LastWeekdayInMonthHoliday
  TEST_F(HolidayTest, MemorialDayTest) {
    MemorialDay md(1, 2);
    Date md_2010(May, 31, 2010);
    Date md_2011(May, 30, 2011);
    EXPECT_EQ(md_2010.day_of_week(), Mon);
    EXPECT_EQ(md_2011.day_of_week(), Mon);
    EXPECT_EQ(md.nearest(md_2010 - 1), md_2010);
    EXPECT_EQ(md.nearest(md_2010), md_2010);
    EXPECT_EQ(md.nearest(md_2010 + 1), md_2010);
    EXPECT_EQ(md.nearest(md_2010 + 120), md_2010);
    EXPECT_EQ(md.nearest(md_2010 + 300), md_2011);

    EXPECT_FALSE(md.active(md_2010 - 2));
    EXPECT_TRUE(md.active(md_2010 - 1));
    EXPECT_TRUE(md.active(md_2010));
    EXPECT_TRUE(md.active(md_2010 + 1));
    EXPECT_TRUE(md.active(md_2010 + 2));
    EXPECT_FALSE(md.active(md_2010 + 3));
    EXPECT_FALSE(md.active(md_2010 + 37));
  }
  
  TEST_F(HolidayTest, DateRangeHoliday) {
    DateRangeHoliday holiday;
    EXPECT_EQ(-1, holiday.maximum_window_width());
    EXPECT_FALSE(holiday.active(Date(Jan, 1, 2012)));
    EXPECT_FALSE(holiday.active(Date(Jan, 3, 2012)));

    holiday.add_dates(Date(Dec, 30, 2010), Date(Jan, 2, 2011));
    holiday.add_dates(Date(Dec, 30, 2011), Date(Jan, 2, 2012));
    holiday.add_dates(Date(Dec, 30, 2012), Date(Jan, 2, 2013));
    holiday.add_dates(Date(Dec, 30, 2013), Date(Jan, 2, 2014));
    holiday.add_dates(Date(Dec, 30, 2014), Date(Jan, 2, 2015));
    holiday.add_dates(Date(Dec, 30, 2015), Date(Jan, 2, 2016));

    EXPECT_TRUE(holiday.active(Date(Dec, 30, 2012)));
    EXPECT_TRUE(holiday.active(Date(Dec, 31, 2012)));
    EXPECT_TRUE(holiday.active(Date(Jan, 1, 2013)));
    EXPECT_TRUE(holiday.active(Date(Jan, 2, 2013)));
    EXPECT_FALSE(holiday.active(Date(Jan, 3, 2012)));
    EXPECT_FALSE(holiday.active(Date(Jan, 1, 2020)));
    EXPECT_EQ(4, holiday.maximum_window_width());

    EXPECT_EQ(-1, holiday.days_into_influence_window(
        Date(Dec, 29, 2010)));
    EXPECT_EQ(0, holiday.days_into_influence_window(
        Date(Dec, 30, 2010)));
    EXPECT_EQ(1, holiday.days_into_influence_window(
        Date(Dec, 31, 2010)));
    EXPECT_EQ(2, holiday.days_into_influence_window(
        Date(Jan, 1, 2011)));
    EXPECT_EQ(3, holiday.days_into_influence_window(
        Date(Jan, 2, 2011)));
    EXPECT_EQ(-1, holiday.days_into_influence_window(
        Date(Jan, 3, 2011)));

    NewYearsDay new_years_day(2, 1);
    EXPECT_EQ(-1, new_years_day.days_into_influence_window(
        Date(Dec, 29, 2010)));
    EXPECT_EQ(0, new_years_day.days_into_influence_window(
        Date(Dec, 30, 2010)));
    EXPECT_EQ(1, new_years_day.days_into_influence_window(
        Date(Dec, 31, 2010)));
    EXPECT_EQ(2, new_years_day.days_into_influence_window(
        Date(Jan, 1, 2011)));
    EXPECT_EQ(3, new_years_day.days_into_influence_window(
        Date(Jan, 2, 2011)));
    EXPECT_EQ(-1, new_years_day.days_into_influence_window(
        Date(Jan, 3, 2011)));

    
    DateRangeHoliday single_day_holiday;
    single_day_holiday.add_dates(Date(Jan, 14, 1971), Date(Jan, 14, 1971));
    EXPECT_TRUE(single_day_holiday.active(Date(Jan, 14, 1971)));
    EXPECT_EQ(1, single_day_holiday.maximum_window_width());

    std::vector<Date> start_dates;
    start_dates.push_back(Date(Dec, 30, 2010));
    start_dates.push_back(Date(Dec, 30, 2011));
    start_dates.push_back(Date(Dec, 30, 2012));
    start_dates.push_back(Date(Dec, 30, 2013));
    start_dates.push_back(Date(Dec, 30, 2014));
    start_dates.push_back(Date(Dec, 30, 2015));


    std::vector<Date> end_dates;
    end_dates.push_back(Date(Jan, 2, 2011));
    end_dates.push_back(Date(Jan, 2, 2012));
    end_dates.push_back(Date(Jan, 2, 2013));
    end_dates.push_back(Date(Jan, 2, 2014));
    end_dates.push_back(Date(Jan, 2, 2015));
    end_dates.push_back(Date(Jan, 2, 2016));

    DateRangeHoliday second_holiday(start_dates, end_dates);
    EXPECT_TRUE(second_holiday.active(Date(Dec, 30, 2012)));
    EXPECT_TRUE(second_holiday.active(Date(Dec, 31, 2012)));
    EXPECT_TRUE(second_holiday.active(Date(Jan, 1, 2013)));
    EXPECT_TRUE(second_holiday.active(Date(Jan, 2, 2013)));
    EXPECT_FALSE(second_holiday.active(Date(Jan, 3, 2012)));
    EXPECT_FALSE(second_holiday.active(Date(Jan, 1, 2020)));
    EXPECT_EQ(4, second_holiday.maximum_window_width());
  }
  
}  // namespace
