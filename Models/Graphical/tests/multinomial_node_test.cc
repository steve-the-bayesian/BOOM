#include "gtest/gtest.h"

#include "Models/Graphical/Node.hpp"
#include "Models/Graphical/MultinomialNode.hpp"
#include "distributions.hpp"
#include "distributions/rng.hpp"
#include "stats/fake_data_table.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::Graphical;

  class MultinomialNodeTest : public ::testing::Test {
   protected:
    MultinomialNodeTest() {
      GlobalRng::rng.seed(8675309);
    }

    DataTable data_;
  };

  TEST_F(MultinomialNodeTest, test_stuff) {
    // Create a fake data table with 3 columns V1, V2, V3, filled with
    // categorical variables with 3, 2, and 4 levels.
    data_ = fake_data_table(100, 0, {3, 2, 4});
    for (const auto &el : data_.vnames()) {
      std::cout << el << "\n";
    }

    NEW(MultinomialNode, v1)(data_, "V1");
    EXPECT_EQ(v1->node_type(), NodeType::CATEGORICAL);
  }
}
