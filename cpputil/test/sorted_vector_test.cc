#include "gtest/gtest.h"
#include "test_utils/test_utils.hpp"
#include "cpputil/SortedVector.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  // Shufle the same vector many times, and check that the values in each vector
  // position are uniformly distributed.
  TEST(SortedVectorTest, UnionIntersection) {
    GlobalRng::rng.seed(8675309);

    SortedVector<int> values{1, 2, 1, 1, 3};
    EXPECT_EQ(values.size(), 3);
    EXPECT_EQ(values[0], 1);
    EXPECT_EQ(values[1], 2);
    EXPECT_EQ(values[2], 3);

    SortedVector<int> more_values{5, 1, 1, 4, 3};
    EXPECT_EQ(more_values.size(), 4);

    SortedVector<int> ans = values.set_union(more_values);
    EXPECT_EQ(ans.size(), 5);
    EXPECT_EQ(ans[0], 1);
    EXPECT_EQ(ans[1], 2 );
    EXPECT_EQ(ans[2], 3);
    EXPECT_EQ(ans[3], 4);
    EXPECT_EQ(ans[4], 5);

    SortedVector<int> intersection = values.intersection(more_values);
    EXPECT_EQ(intersection.size(), 2);
    EXPECT_EQ(intersection[0], 1);
    EXPECT_EQ(intersection[1], 3);
  }

  TEST(SortedVectorTest, InsertAndErase) {
    SortedVector<int> values;

    values.insert(5);
    values.insert(3);
    values.insert(7);
    values.insert(1);
    values.insert(1);

    EXPECT_EQ(values.size(), 4);
    EXPECT_EQ(values[0], 1);
    EXPECT_EQ(values[1], 3);
    EXPECT_EQ(values[2], 5);
    EXPECT_EQ(values[3], 7);

    values.remove(3);
    EXPECT_EQ(values.size(), 3);
    auto it = values.remove(5);
    EXPECT_EQ(*it, 7);
    EXPECT_EQ(values[0], 1);
    EXPECT_EQ(values[1], 7);
  }

  TEST(SortedVectorTest, TestIsSubset) {
    SortedVector<int> values;
    values.insert(2);
    values.insert(3);
    values.insert(4);

    SortedVector<int> superset;
    superset.insert(1);
    superset.insert(2);
    superset.insert(3);
    superset.insert(4);
    superset.insert(5);

    EXPECT_TRUE(values.is_subset(superset));

    values.insert(7);
    EXPECT_FALSE(values.is_subset(superset));
  }

  TEST(SortedVectorTest, TestAbsorb) {
    SortedVector<int> values;
    values.insert(2);
    values.insert(3);
    values.insert(4);

    SortedVector<int> more_values;
    more_values.insert(1);
    more_values.insert(3);
    more_values.insert(5);
    more_values.insert(7);

    values.absorb(more_values);
    EXPECT_EQ(6, values.size());
    EXPECT_EQ(1, values[0]);
    EXPECT_EQ(2, values[1]);
    EXPECT_EQ(3, values[2]);
    EXPECT_EQ(4, values[3]);
    EXPECT_EQ(5, values[4]);
    EXPECT_EQ(7, values[5]);
  }

  TEST(SortedVectorTest, TestPosition) {
    SortedVector<int> values;
    values.insert(2);
    values.insert(3);
    values.insert(4);
    values.insert(2);
    values.insert(2);
    values.insert(2);
    values.insert(2);

    EXPECT_EQ(values.position(2), 0);
    EXPECT_EQ(values.position(3), 1);
    EXPECT_EQ(values.position(4), 2);
    EXPECT_EQ(values.position(17), -1);
    EXPECT_EQ(values.position(-8), -1);
  }

  // Test that it is safe to call 'remove' on an element that is not part of the
  // SortedVector.
  TEST(SortedVectorTest, TestRemove) {
    SortedVector<int> values;
    values.insert(1);
    auto it = values.remove(3);
    EXPECT_EQ(it, values.end());
  }

}  // namespace
