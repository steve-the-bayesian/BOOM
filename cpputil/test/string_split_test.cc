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

    std::string test = " 1 \"2  \"   ";
    std::vector<std::string> expected = {"1", "2  "};
    EXPECT_EQ(expected, split(test));

    string extra_space = "1 2 3  4 5";
    expected = {"1", "2", "3", "4", "5"};
    EXPECT_EQ(expected, split(extra_space));

    test = "1";
    expected = {"1"};
    EXPECT_EQ(expected, split(test));

    test = "1 ";
    expected = {"1"};
    EXPECT_EQ(expected, split(test));

    test = " 1";
    expected = {"1"};
    EXPECT_EQ(expected, split(test));

    test = "   1   ";
    expected = {"1"};
    EXPECT_EQ(expected, split(test));

    test = "";
    expected = {};
    EXPECT_EQ(expected, split(test));

    test = "1 2";
    expected = {"1", "2"};
    EXPECT_EQ(expected, split(test));

    test = " 1 2";
    expected = {"1", "2"};
    EXPECT_EQ(expected, split(test));

    test = "1 2 ";
    expected = {"1", "2"};
    EXPECT_EQ(expected, split(test));

    test = " 1 2 ";
    expected = {"1", "2"};
    EXPECT_EQ(expected, split(test));
  }

  TEST(StringSplitTest, Comma) {
    StringSplitter split(",");
    string test = "8, 6, 7, 5,,3,   0, 9";
    std::vector<string> expected{
      "8", "6", "7", "5", "", "3", "0", "9"};
    EXPECT_EQ(expected, split(test));

    test = ",3";
    expected = {"", "3"};
    EXPECT_EQ(expected, split(test));

    test = "3,";
    expected = {"3", ""};
    EXPECT_EQ(expected, split(test));

    string all_commas = ",";
    expected = std::vector<std::string>(2, "");
    EXPECT_EQ(expected, split(all_commas));

    test = "3";
    expected = {"3"};
    EXPECT_EQ(expected, split(test));

    test = "8, 6, 7,\"53   09\"";
    expected = {"8", "6", "7", "53   09"};
    EXPECT_EQ(expected, split(test));
  }

}  // namespace
