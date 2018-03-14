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
