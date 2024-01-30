#include "gtest/gtest.h"
#include "cpputil/index_table.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  template <class T>
  void print_vector(const std::vector<T> &x) {
    for (const auto &el : x) {
      std::cout << el << ' ';
    }
    std::cout << "\n";
  }

  TEST(IndexTableTest, JennyIndex) {
    std::vector<int> digits = {8, 6, 7, 5, 3, 0, 9};
    // when 'digits' is sorted it becomes
    // {0, 3, 5, 6, 7, 8, 9}, so the index table is
    // {5, 4, 3, 1, 2, 0, 6}
    // Each of these numbers is the position of the sorted array element in the
    // original array.
    std::vector<int> indices = index_table<int, int>(digits);
    EXPECT_EQ(indices[0], 5);
    EXPECT_EQ(indices[1], 4);
    EXPECT_EQ(indices[2], 3);
    EXPECT_EQ(indices[3], 1);
    EXPECT_EQ(indices[4], 2);
    EXPECT_EQ(indices[5], 0);
    EXPECT_EQ(indices[6], 6);
  }

  TEST(IndexTableTest, SelfInversion) {
    std::vector<int> digits = {8, 6, 7, 5, 3, 0, 9};

    std::vector<int> order = index_table<int, int>(digits);
    std::vector<int> inverse_order = index_table<int, int>(order);

    for (int i = 0; i < digits.size(); ++i ){
      EXPECT_EQ(digits[order[inverse_order[i]]], digits[i]);
      EXPECT_EQ(digits[inverse_order[order[i]]], digits[i]);
    }
  }

  TEST(IndexTableTest, MoreInts) {
    std::vector<int> inputs =  {6, -3, 0, 5, 5, 5};
    std::vector<int> positions = index_table<int, int>(inputs);

    EXPECT_EQ(positions.size(), inputs.size());
    EXPECT_EQ(positions[0], 1);
    EXPECT_EQ(positions[1], 2);
    EXPECT_EQ(positions[2], 3);
    EXPECT_EQ(positions[3], 4);
    EXPECT_EQ(positions[4], 5);
    EXPECT_EQ(positions[5], 0);
  }

}  // namespace
