#include "gtest/gtest.h"
#include "math/Permutation.hpp"
#include "test_utils/test_utils.hpp"
#include <algorithm>
#include <random>
#include "distributions.hpp"
#include "cpputil/seq.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  template <class T>
  std::string print_vector(const std::vector<T> &x) {
    std::ostringstream out;
    for (const auto &el : x) {
      out << el << ' ';
    }
    out << "\n";
    return out.str();
  }

  TEST(PermutationTest, ApplyToInts) {
    Permutation<int> perm({2, 0, 1, 3});
    Vector x = {1, 2, 3, 4};
    Vector y = perm * x;
    EXPECT_DOUBLE_EQ(y[0], 3.0);
    EXPECT_DOUBLE_EQ(y[1], 1.0);
    EXPECT_DOUBLE_EQ(y[2], 2.0);
    EXPECT_DOUBLE_EQ(y[3], 4.0);

    Permutation<int> inv = perm.inverse();
    Vector z = inv * y;
    EXPECT_TRUE(VectorEquals(z, x));
  }

  TEST(PermutationTest, TestRandomPermutation) {
    Permutation<int> perm = random_permutation<int>(10);
    // Check that all the numbers are present, and that they're not in order.

    EXPECT_EQ(perm.size(), 10);
    std::vector<int> integers = seq<int>(0, 9);
    EXPECT_NE(integers, perm.elements());
  }

  TEST(PermutationTest, Composition) {
    std::vector<Int> digits;
    for (int i = 0; i < 10; ++i) {
      digits.push_back(i);
    }

    std::shuffle(digits.begin(), digits.end(), std::default_random_engine(3));
    Permutation<Int> p1(digits);

    std::shuffle(digits.begin(), digits.end(), std::default_random_engine(3));
    Permutation<Int> p2(digits);

    Vector z = rnorm_vector(10, 0, 1);
    Vector z1 = (p1 * p2) * z;
    Vector z2 = p1 * (p2 * z);
    EXPECT_TRUE(VectorEquals(z1, z2));

    Vector z3 = p2 * p1 * z;
    Vector z4 = p2 * (p1 * z);
    Vector z5 = (p2 * p1) * z;
    EXPECT_TRUE(VectorEquals(z3, z4));
    EXPECT_TRUE(VectorEquals(z3, z5));
  }


}  // namespace
