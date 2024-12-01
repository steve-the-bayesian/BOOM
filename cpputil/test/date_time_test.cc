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
  }

}  // namespace
