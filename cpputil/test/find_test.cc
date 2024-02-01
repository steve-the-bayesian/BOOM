#include "gtest/gtest.h"
#include "cpputil/find.hpp"
#include "cpputil/string_utils.hpp"
#include "test_utils/test_utils.hpp"
#include "distributions.hpp"
#include <fstream>

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

  TEST(FindTest, Jenny) {
    std::vector<int> sorted_digits = {0, 3, 5, 6, 7, 8, 9};
    std::vector<Int> indices = find_sorted<int, int>({0,3,8}, sorted_digits);
    EXPECT_EQ(3, indices.size());
    EXPECT_EQ(0, indices[0]);
    EXPECT_EQ(1, indices[1]);
    EXPECT_EQ(5, indices[2]);
  }

  // If an element is not in the target set, it should get assigned -1.
  TEST(FindTest, ElementNotPresent) {
    std::vector<int> sorted_digits = {0, 3, 5, 6, 7, 8, 9};
    std::vector<Int> indices = find_sorted<int, int>({0, 1, 3, 8},
                                                     sorted_digits);
    EXPECT_EQ(4, indices.size());
    EXPECT_EQ(0, indices[0]);
    EXPECT_EQ(-1, indices[1]);
    EXPECT_EQ(1, indices[2]);
    EXPECT_EQ(5, indices[3]);
  }

  // If an element appears multiple times we should get the smallest possible
  // indices, but numbers should not repeat.
  TEST(FindTest, MultipleElements) {
    std::vector<int> sorted_digits = {0, 3, 5, 5, 5, 8, 9};
    std::vector<Int> indices = find_sorted<int, int>({0, 5, 5, 8},
                                                     sorted_digits);
    EXPECT_EQ(4, indices.size()) << print_vector(indices);
    EXPECT_EQ(0, indices[0]) << print_vector(indices);
    EXPECT_EQ(2, indices[1]) << print_vector(indices);
    EXPECT_EQ(3, indices[2]) << print_vector(indices);
    EXPECT_EQ(5, indices[3]) << print_vector(indices);
  }

  // Given a set of unsorted inputs and outputs, return the positions of the
  // inputs in the target set.
  TEST(FindTest, UnorderedTest) {
    std::vector<Int> digits = {8, 6, 7, 5, 3, 0, 9};
    std::vector<Int> inputs = {3, 5, 8};

    std::vector<Int> positions = find(inputs, digits);
    EXPECT_EQ(positions.size(), 3);
    EXPECT_EQ(positions[0], 4) << print_vector(positions);
    EXPECT_EQ(positions[1], 3) << print_vector(positions);
    EXPECT_EQ(positions[2], 0) << print_vector(positions);
  }

  // Now try it with a bunch of random numbers.
  TEST(FindTest, UnorderedRandomNumbersTest) {
    Vector z = rnorm_vector(20, 0.0, 1.0);
    Vector inputs = {z[3], z[1], z[8]};

    std::vector<Int> positions = find(inputs, z);
    EXPECT_EQ(positions.size(), 3);
    EXPECT_EQ(positions[0], 3) << print_vector(positions);
    EXPECT_EQ(positions[1], 1) << print_vector(positions);
    EXPECT_EQ(positions[2], 8) << print_vector(positions);
  }

  // Now try it with a bunch of random numbers.
  TEST(FindTest, UnsortedWithRepeats) {
    std::vector<int> targets = {8, 6, 7, 5, 3, 0, 9, 5, 5, 5};
    std::vector<int> inputs = {6, 0, 5, 5, 5};

    std::vector<Int> positions = find(inputs, targets);
    EXPECT_EQ(positions.size(), 5);
    EXPECT_EQ(positions[0], 1) << print_vector(positions);
    EXPECT_EQ(positions[1], 5) << print_vector(positions);
    EXPECT_EQ(positions[2], 3) << print_vector(positions);
    EXPECT_EQ(positions[3], 7) << print_vector(positions);
    EXPECT_EQ(positions[4], 8) << print_vector(positions);
  }

  // Unsorted inputs with repeats and missing values.
  TEST(FindTest, UnsortedWithRepeatsAndMissings) {
    std::vector<int> targets = {8, 6, 7, 5, 3, 0, 9, 5, 5, 5};
    std::vector<int> inputs = {6, -3, 0, 5, 5, 5};

    std::vector<Int> positions = find(inputs, targets);
    EXPECT_EQ(positions.size(), 6);
    EXPECT_EQ(positions[0], 1) << print_vector(positions);
    EXPECT_EQ(positions[1], -1) << print_vector(positions);
    EXPECT_EQ(positions[2], 5) << print_vector(positions);
    EXPECT_EQ(positions[3], 3) << print_vector(positions);
    EXPECT_EQ(positions[4], 7) << print_vector(positions);
    EXPECT_EQ(positions[5], 8) << print_vector(positions);
  }

  // If searching for more instances of an input than exist in the target list,
  // the "extra" instances are marked as missing.
  TEST(FindTest, FindMultiples) {
    std::vector<std::string> inputs = {
      "Fred", "Barney", "Wilma", "Fred", "Fred"
    };
    std::vector<std::string> targets = {
      "Barney", "Fred", "Betty", "Wilma", "BamBam", "Pebbles", "Dino"
    };
    std::vector<Int> positions = find(inputs, targets);
    EXPECT_EQ(positions.size(), inputs.size());
    EXPECT_EQ(positions[0], 1);
    EXPECT_EQ(positions[1], 0);
    EXPECT_EQ(positions[2], 3);
    EXPECT_EQ(positions[3], -1);
    EXPECT_EQ(positions[4], -1);
  }

}  // namespace
