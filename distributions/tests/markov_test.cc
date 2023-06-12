#include "gtest/gtest.h"
#include "distributions.hpp"
#include "distributions/Markov.hpp"
#include "test_utils/test_utils.hpp"
#include <functional>

namespace {

  using namespace BOOM;

  // Verify that the vector of stationary probabilities found by get_stat_dist
  // is actually the invariant distribution.
  TEST(GetStatDistTest, FindsInvariantDistribution) {
    int S = 4;
    Matrix P(S, S);
    Vector ones(S, 1.0);
    for (int iteration = 0; iteration < 100; ++iteration) {
      for (int i = 0; i < S; ++i) {
        P.row(i) = rdirichlet(ones);
      }

      Vector probs = get_stat_dist(P);
      Vector new_probs = probs * P;

      EXPECT_TRUE(VectorEquals(probs, new_probs));
      EXPECT_NEAR(probs.sum(), 1.0, 1e-8);
      EXPECT_GE(probs.min(), 0.0);
      EXPECT_LE(probs.max(), 1.0);
    }
  }


}  // namespace
