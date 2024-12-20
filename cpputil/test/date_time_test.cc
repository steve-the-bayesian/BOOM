#include "gtest/gtest.h"
#include "cpputil/DateTime.hpp"

namespace {
  using namespace BOOM;
  using std::string;
  using std::endl;

  TEST(DateTimeTest, NanosecondConversion) {
    Date date(MonthNames::May, 15, 2004);
    DateTime dt(date, 0.0);

    int64_t expected_ns = 1084579200000000000;
    EXPECT_EQ(dt.nanoseconds_since_epoch(),
              expected_ns);

    DateTime dt1(date, 0, 0, 1.1);
    EXPECT_EQ(dt1.nanoseconds_after_second(),
              1e+8);

    EXPECT_EQ(dt1.nanoseconds_into_day(),
              1.1e+9);

    Date epoch_date(MonthNames::Jan, 1, 1970);
    DateTime dt_epoch(epoch_date, 0, 0, 0.0);
    EXPECT_EQ(dt_epoch.nanoseconds_since_epoch(), 0);

    DateTime dt_one_second(epoch_date, 0, 0, 1.0);
    EXPECT_EQ(dt_one_second.nanoseconds_since_epoch(),
              1e+9);

    Date May_15_2024(MonthNames::May, 15, 2024);
    DateTime May_15_2024_midnight(May_15_2024, 0, 0, 0);
    int64_t May_15_2024_ns = May_15_2024_midnight.nanoseconds_since_epoch();
    EXPECT_EQ(May_15_2024_ns,
              1715731200000000000);

  }

}  // namespace
