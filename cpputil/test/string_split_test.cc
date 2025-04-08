#include "gtest/gtest.h"
#include "cpputil/StringSplitter.hpp"
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

  TEST(StringSplitTest, PopBack) {
    std::string test = "foo/bar/baz";
    StringSplitter split("/");
    std::pair<std::string, std::string> result = split.pop_back(test);
    EXPECT_EQ(result.first, "foo/bar");
    EXPECT_EQ(result.second, "baz");

    test = "";
    result = split.pop_back(test);
    EXPECT_EQ(result.first, "");
    EXPECT_EQ(result.second, "");

    test = "foo";
    result = split.pop_back(test);
    EXPECT_EQ(result.first, "");
    EXPECT_EQ(result.second, "foo");
    
    test = "foo/";
    result = split.pop_back(test);
    EXPECT_EQ(result.first, "foo");
    EXPECT_EQ(result.second, "");
  }

  TEST(StringSplitTest, SkipEmpty) {
    StringSplitter split("/");
    std::string test = "/foo/bar/baz/";
    std::vector<std::string> elements = split(test);
    EXPECT_EQ(5, elements.size());
    EXPECT_EQ("", elements[0]);
    EXPECT_EQ("foo", elements[1]);
    EXPECT_EQ("bar", elements[2]);
    EXPECT_EQ("baz", elements[3]);
    EXPECT_EQ("", elements[4]);

    split.omit_empty();
    elements = split(test);
    EXPECT_EQ(3, elements.size());
    EXPECT_EQ("foo", elements[0]);
    EXPECT_EQ("bar", elements[1]);
    EXPECT_EQ("baz", elements[2]);

    test = "////foo///bar//baz////";
    elements.clear();
    elements = split(test);
    EXPECT_EQ(3, elements.size());
    EXPECT_EQ("foo", elements[0]);
    EXPECT_EQ("bar", elements[1]);
    EXPECT_EQ("baz", elements[2]);
  }

}  // namespace
