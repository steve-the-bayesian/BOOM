#include "gtest/gtest.h"
#include "distributions.hpp"
#include "Models/Mixtures/identify_permutation.hpp"
#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;

  class IdentifyPermutationTest : public ::testing::Test {
   protected:
    IdentifyPermutationTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  bool isin(int value, std::vector<int> &values) {
    for (size_t i = 0; i < values.size(); ++i) {
      if (values[i] == value) {
        return true;
      }
    }
    return false;
  }

  TEST_F(IdentifyPermutationTest, IdentifyProbs) {
    Matrix draws(
        "1.349097e-01 2.828971e-03 8.622613e-01 |"
        "3.555298e-08 9.999983e-01 1.633781e-06 |"
        "5.454057e-01 4.537361e-01 8.581825e-04 |"
        "1.765472e-03 7.259953e-12 9.982345e-01 |"
        "9.990390e-01 3.023542e-05 9.307388e-04 |"
        "9.999994e-01 6.350167e-07 5.614189e-11 |"
        "2.055999e-03 2.163755e-01 7.815685e-01 |"
        "8.598675e-01 9.276903e-07 1.401316e-01 |"
        "7.603148e-05 1.662744e-06 9.999223e-01 |"
        "5.656471e-03 9.943435e-01 6.248708e-09 ");

    Matrix more_draws(
        "1.263361e-01 0.873663923 1.005958e-18 |"
        "8.600590e-01 0.139941000 1.660316e-08 |"
        "6.692977e-02 0.933068662 1.564199e-06 |"
        "9.949724e-01 0.001021329 4.006221e-03 |"
        "2.102381e-03 0.413140858 5.847568e-01 |"
        "9.894634e-01 0.010366576 1.700637e-04 |"
        "2.092655e-20 0.999999997 2.774452e-09 |"
        "1.522979e-01 0.845446078 2.256025e-03 |"
        "2.543521e-01 0.741540422 4.107509e-03 |"
        "4.494382e-06 0.201376480 7.986190e-01 ");

    Matrix even_more_draws(
        "9.375064e-01 4.099853e-03 5.839378e-02 |"
        "5.550616e-01 7.203924e-07 4.449377e-01 |"
        "1.945662e-02 9.802557e-01 2.876400e-04 |"
        "3.299871e-01 6.630538e-01 6.959044e-03 |"
        "9.039558e-13 9.999927e-01 7.271936e-06 |"
        "9.985907e-01 1.409269e-03 1.701939e-14 |"
        "7.891489e-03 2.763948e-03 9.893446e-01 |"
        "9.567300e-01 1.098952e-05 4.325906e-02 |"
        "9.890624e-01 5.658377e-20 1.093756e-02 |"
        "1.055451e-04 1.264707e-05 9.998818e-01 ");

    std::vector<Matrix> cluster_probs = {
      draws, more_draws, even_more_draws};

    std::vector<std::vector<int>> permutation =
        identify_permutation_from_probs(cluster_probs);

    EXPECT_EQ(3, permutation.size());
    EXPECT_EQ(3, permutation[0].size());
    for (int i = 0; i < 3; ++i) {
      EXPECT_TRUE(isin(0, permutation[i]));
      EXPECT_TRUE(isin(1, permutation[i]));
      EXPECT_TRUE(isin(2, permutation[i]));
    }
  }

  TEST_F(IdentifyPermutationTest, identify_labels) {
    std::vector<std::vector<int>> draws;
    draws.push_back(std::vector<int>{0, 0, 0, 0, 1, 1, 1, 1});
    draws.push_back(std::vector<int>{0, 0, 0, 0, 1, 1, 1, 1});
    draws.push_back(std::vector<int>{0, 0, 0, 0, 1, 1, 1, 1});
    draws.push_back(std::vector<int>{0, 0, 0, 0, 1, 1, 1, 1});
    draws.push_back(std::vector<int>{0, 0, 0, 0, 1, 1, 1, 1});
    draws.push_back(std::vector<int>{0, 0, 0, 1, 1, 1, 1, 1});
    draws.push_back(std::vector<int>{0, 0, 0, 0, 0, 1, 1, 1});
    draws.push_back(std::vector<int>{1, 1, 1, 1, 0, 0, 0, 0});
    draws.push_back(std::vector<int>{1, 1, 1, 0, 0, 0, 0, 0});
    draws.push_back(std::vector<int>{1, 1, 1, 1, 1, 0, 0, 0});

    std::vector<std::vector<int>> permutation =
        identify_permutation_from_labels(draws);
    EXPECT_EQ(draws.size(), permutation.size());
    EXPECT_EQ(2, permutation[0].size());
    for (int i = 0; i < permutation.size(); ++i) {
      EXPECT_EQ(permutation[i][0], 1 - permutation[i][1]);
    }

    int perm1 = permutation[0][0];
    for (int i = 0; i < 7; ++i) {
      EXPECT_EQ(permutation[i][0], perm1);
    }
    for (int i = 7; i < permutation.size(); ++i) {
      EXPECT_EQ(permutation[i][0], 1 - perm1);
    }
  }

}  // namespace
