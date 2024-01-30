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
    std::vector<int> indices = index_table(digits);
    EXPECT_EQ(indices[0], 5);
    EXPECT_EQ(indices[1], 4);
    EXPECT_EQ(indices[2], 3);
    EXPECT_EQ(indices[3], 1);
    EXPECT_EQ(indices[4], 2);
    EXPECT_EQ(indices[5], 0);
    EXPECT_EQ(indices[6], 6);
  }
  
  TEST(IndexTableTest, JennyRank) {
    std::vector<int> digits = {8, 6, 7, 5, 3, 0, 9};
    // when 'digits' is sorted it becomes
    // {0, 3, 5, 6, 7, 8, 9}, so the index table is
    // {5, 3, 4, 2, 1, 0, 6}
    // Each of these numbers is the position of the sorted array element in the
    // original array.
    std::vector<int> ranks = rank_table(digits);
    EXPECT_EQ(ranks[0], 5);
    EXPECT_EQ(ranks[1], 3);
    EXPECT_EQ(ranks[2], 4);
    EXPECT_EQ(ranks[3], 2);
    EXPECT_EQ(ranks[4], 1);
    EXPECT_EQ(ranks[5], 0);
    EXPECT_EQ(ranks[6], 6);
  }

  // Verifies that the rank_table and the inverse_table are inverses of one
  // another.
  TEST(IndexTableTest, JennyInverse) {
    std::vector<int> digits = {8, 6, 7, 5, 3, 0, 9};
    
    std::vector<int> ranks = rank_table(digits);
    std::vector<int> order = index_table(digits);

    for (int i = 0; i < digits.size(); ++i ){
      EXPECT_EQ(digits[order[ranks[i]]], digits[i]);
      EXPECT_EQ(digits[ranks[order[i]]], digits[i]);
    }
  }

  TEST(IndexTableTest, SelfInversion) {
    std::vector<int> digits = {8, 6, 7, 5, 3, 0, 9};

    std::vector<int> order = index_table(digits);
    std::vector<int> inverse_order = index_table(order);

    for (int i = 0; i < digits.size(); ++i ){
      EXPECT_EQ(digits[order[inverse_order[i]]], digits[i]);
      EXPECT_EQ(digits[inverse_order[order[i]]], digits[i]);
    }
    
  }
  
}  // namespace
