#include "gtest/gtest.h"
#include "cpputil/string_utils.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  std::vector<std::string> strings = {"foo", "bar", "  baz"};
  
  TEST(ConcatenateStringsTest, NormalCase) {
    std::string cat = concatenate(strings);
    EXPECT_EQ(13, cat.size());
    EXPECT_EQ("foo bar   baz", cat);
  }

  TEST(ConcatenateEmpty, EmptyTest) {
    std::vector<std::string> empty;
    std::string cat = concatenate(empty);
    EXPECT_TRUE(cat.empty());
  }

  TEST(ConcatenateNewline, NewlineTest) {
    std::string cat = concatenate(strings, "\n");
    EXPECT_EQ(cat, "foo\nbar\n  baz");
  }
  
}  // namespace
