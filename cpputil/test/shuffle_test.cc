#include "gtest/gtest.h"
#include "cpputil/shuffle.hpp"
#include "test_utils/test_utils.hpp"
#include "stats/FreqDist.hpp"
#include "stats/ChiSquareTest.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  // Shufle the same vector many times, and check that the values in each vector
  // position are uniformly distributed.
  TEST(shuffle, works_as_intended) {
    GlobalRng::rng.seed(8675309);

    std::vector<int> values = {8, 6, 7, 5, 3, 0, 9};
    std::vector<std::vector<int>> positions(7);

    for (int i = 0; i < 500000; ++i) {
      shuffle(values, GlobalRng::rng);
      for (int i = 0; i < 7; ++i) {
        positions[i].push_back(values[i]);
      }
    }

    for (int i = 0; i < 7; ++i) {
      FrequencyDistribution dist(positions[i], false);
      OneWayChiSquareTest test(dist, Vector(7, 1.0 / 7));
      EXPECT_GT(test.p_value() , .05)
          << std::endl
          << "position " << i << dist;
    }
  }
  
}  // namespace

