#include "gtest/gtest.h"

#include "cpputil/Ptr.hpp"
#include "distributions.hpp"
#include "Models/Glm/Glm.hpp"
#include "stats/FreqDist.hpp"
#include "stats/ChiSquareTest.hpp"

#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  TEST(Ptr, works_as_intended) {
    GlobalRng::rng.seed(8675309);

    int sample_size = 1000000;
    std::vector<Ptr<RegressionData>> data_vector;
    
    for (int i = 0; i < sample_size; ++i) {
      Vector x(4);
      x.randomize();
      NEW(RegressionData, data_point)(rnorm(), x);
      data_vector.push_back(data_point);
    }

    // Try to trigger ASAN.
    std::vector<Ptr<RegressionData>> other_data_vector(data_vector);

    // Try to trigger ASAN.
    std::vector<Ptr<RegressionData>> moved_data_vector(
        std::move(data_vector));
  }
  
}  // namespace

