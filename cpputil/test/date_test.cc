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



}  // namespace
