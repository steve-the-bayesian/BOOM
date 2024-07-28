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
    data_ = fake_data_table(100, 0, {3, 2, 4});
  }
}
