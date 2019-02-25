#include "gtest/gtest.h"
#include "cpputil/Split.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::string;
  using std::endl;

  TEST(StringSplitTest, AllWhiteSpace) {
    StringSplitter split;
    string all_spaces = "                    ";
    std::vector<string> result = split(all_spaces);
    EXPECT_TRUE(result.empty());

    string extra_space = "1 2 3  4 5";
    EXPECT_EQ(std::vector<string>({"1", "2", "3", "4", "5"}),
              split(extra_space));
  }

  TEST(StringSplitTest, Comma) {
    StringSplitter split(",");
    string test = "8, 6, 7, 5,,3,   0, 9";
    EXPECT_EQ(std::vector<string>({"8", " 6", " 7", " 5", "", "3",
              "   0", " 9"}),
              split(test));
  }
  
}  // namespace
