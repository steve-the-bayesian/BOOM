#include "gtest/gtest.h"
#include "Models/StateSpace/AggregatedStateSpaceRegression.hpp"
#include "distributions.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class FineNowcastingDataTest : public ::testing::Test {
   protected:
    FineNowcastingDataTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  void allocate_and_deallocate(int sample_size, int xdim) {
    std::vector<Ptr<FineNowcastingData>> data;
    for (int i = 0; i < sample_size; ++i) {
      Vector x(xdim);
      x.randomize();
      double y = rnorm();
      bool observed = runif() < .5;
      bool contains_end = runif() < .5;
      double fraction = runif();
      NEW(FineNowcastingData, dp)(x, y, observed, contains_end, fraction);
      data.push_back(dp);
    }
  }

  TEST_F(FineNowcastingDataTest, TestFunctionality) {
    for (int n = 0; n < 100; ++n) {
      for (int p = 0; p < 5; ++p) {
        allocate_and_deallocate(n, p);
      }
    }

  }

  // TEST_F(FineNowcastingDataTest, TestAsanFailure) {
  //   std::vector<int> x(3);
  //   x[500] = 4;
  // }

}  // namespace
