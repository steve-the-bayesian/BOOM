#include "gtest/gtest.h"
#include "Models/MarkovModel.hpp"
#include "distributions.hpp"
#include "numopt/NumericalDerivatives.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using BOOM::uint;
  using std::endl;
  using std::cout;

  class MarkovTest : public ::testing::Test {
   protected:
    MarkovTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(MarkovTest, Data) {
    // Check that links are set and destroyed properly.
    std::vector<std::string> raw_values = {"a", "b", "a", "a", "c"};
    for (int i = 0; i < 100; ++i) {
      Ptr<TimeSeries<MarkovData>> out = make_markov_data(raw_values);
    }

    std::vector<uint> numeric_values = {1, 2, 1, 1, 2, 3, 1, 2};
    for (int i = 0; i < 100; ++i) {
      Ptr<TimeSeries<MarkovData>> out = make_markov_data(numeric_values);
    }

  }


}  // namespace
