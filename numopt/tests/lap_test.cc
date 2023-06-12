#include "gtest/gtest.h"
#include "numopt/LinearAssignment.hpp"
#include "LinAlg/Matrix.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  template <class T>
  std::string print_vector(const std::vector<T> &stuff) {
    std::ostringstream out;
    for (size_t i = 0; i < stuff.size(); ++i) {
      out << stuff[i];
      if (i + 1  < stuff.size()) {
        out << ", ";
      }
    }
    return out.str();
  }


  class LinearAssignmentTest : public ::testing::Test {
   protected:
    LinearAssignmentTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(LinearAssignmentTest, SmallExample) {
    Matrix cost(std::string("1.4775657 2.8113523397 0.2206586 |"
                            "0.2069574 4.6304862991 0.2063028 |"
                            "1.2440799 0.0005641581 0.4644157"));

    std::cout << "cost = \n" << cost;
    LinearAssignment lap(cost);
    double min_cost = lap.solve();
    std::vector<long> ans = lap.row_solution();
    EXPECT_EQ(ans[0], 2);
    EXPECT_EQ(ans[1], 0);
    EXPECT_EQ(ans[2], 1);

    Vector solution_cost(6);
    std::vector<int> assignment = {0, 1, 2}; solution_cost[0] = lap.cost(assignment);
    assignment = {0, 2, 1}; solution_cost[1] = lap.cost(assignment);
    assignment = {1, 2, 0}; solution_cost[2] = lap.cost(assignment);
    assignment = {1, 0, 2}; solution_cost[3] = lap.cost(assignment);
    assignment = {2, 0, 1}; solution_cost[4] = lap.cost(assignment);
    assignment = {2, 1, 0}; solution_cost[5] = lap.cost(assignment);
    EXPECT_DOUBLE_EQ(min_cost, solution_cost[4]);
  }

}  // namespace
