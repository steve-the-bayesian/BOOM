#include "gtest/gtest.h"

#include "stats/DataTable.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>


namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class MultinomialLogitTest : public ::testing::Test {
   protected:
    MultinomialLogitTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(MultinomialLogitTest, ReadData) {
    std::string raw_path = "Models/Glm/tests/autopref.txt";
    DataTable data(raw_path, false, "\t");
  }

}  // namespace
