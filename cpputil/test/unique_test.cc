#include "gtest/gtest.h"
#include "cpputil/make_unique.hpp"
#include "cpputil/make_unique_preserve_order.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::string;
  using std::endl;

  TEST(MakeUniqueTest, Works) {
    std::vector<int> some_data = {1, 2, 1, 1, 1, 3};
    std::vector<int> unique_data = make_unique(some_data);
    EXPECT_EQ(unique_data.size(), 3);
    EXPECT_EQ(unique_data[0], 1);
    EXPECT_EQ(unique_data[1], 2);
    EXPECT_EQ(unique_data[2], 3);
  }

  TEST(MakeUniqueTest, Empty) {
    std::vector<int> some_data = {};
    std::vector<int> unique_data = make_unique(some_data);
    EXPECT_TRUE(unique_data.empty());
  }

  TEST(MakeUniqueTest, PreserveOrder) {
    std::vector<int> some_data = {3, 1, 2, 3, 3, 1, 2, 1, 1, 2};
    std::vector<int> unique_data = make_unique(some_data);

    // The output of make_unique has elements in ascending order.
    EXPECT_EQ(unique_data.size(), 3);
    EXPECT_EQ(unique_data[0], 1);
    EXPECT_EQ(unique_data[1], 2);
    EXPECT_EQ(unique_data[2], 3);

    // The output of make_unique_preserve_order has elements in their original
    // order.
    std::vector<int> ordered = make_unique_preserve_order(some_data);
    EXPECT_EQ(ordered.size(), 3);
    EXPECT_EQ(ordered[0], 3);
    EXPECT_EQ(ordered[1], 1);
    EXPECT_EQ(ordered[2], 2);
  }

}  // namespace
