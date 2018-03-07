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

    DateRangeHoliday single_day_holiday;
    single_day_holiday.add_dates(Date(Jan, 14, 1971), Date(Jan, 14, 1971));
    EXPECT_TRUE(single_day_holiday.active(Date(Jan, 14, 1971)));
    EXPECT_EQ(1, single_day_holiday.maximum_window_width());
  }
  
}  // namespace
