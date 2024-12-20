#include "gtest/gtest.h"
#include "cpputil/Date.hpp"

namespace {
  using namespace BOOM;
  using std::string;
  using std::endl;

  TEST(DateTest, TestMonthNameFromInteger) {
    int year = 2020;
    int month = 4;
    int day = 12;
    Date d(MonthNames(month), day, year);
  }

  TEST(DateTest, NextDay) {
    int year = 2020;
    int month = 4;
    int day = 12;
    Date d(MonthNames(month), day, year);

    Date d1 = d;
    ++d1;
    EXPECT_EQ(d1.month(), MonthNames::Apr);
    EXPECT_EQ(d1.day(), 12 + 1);
    EXPECT_EQ(d1.year(), 2020);

    Date d2 = d + 19;
    EXPECT_EQ(d2.month(), MonthNames::May);
    EXPECT_EQ(d2.day(), 1);
    EXPECT_EQ(d2.year(), 2020);

    EXPECT_LT(d, d2);
    EXPECT_GT(d2, d);
  }



}  // namespace
