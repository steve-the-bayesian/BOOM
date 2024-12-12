#include "gtest/gtest.h"

#include "stats/optimal_arm_probabilities.hpp"
#include "stats/Encoders.hpp"

#include "distributions.hpp"
#include "LinAlg/Vector.hpp"



#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  class OptimalArmProbabilityTest : public ::testing::Test {
   protected:
    OptimalArmProbabilityTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(OptimalArmProbabilityTest, ArgMaxRandomTieTest) {
    Vector v1 = {1.0, 1.0, .99};
    FrequencyDistribution counts(3);

    int sample_size = 100000;
    for (int i = 0; i < sample_size; ++i) {
      size_t winner = argmax_random_ties(v1);
      counts.add_count(winner);
    }

    Vector oap = counts.relative_frequencies();
    EXPECT_EQ(oap.size(), v1.size());
    EXPECT_DOUBLE_EQ(oap[2], 0.0);
    double se = sqrt(.5 * .5 / sample_size);

    EXPECT_NEAR(oap[0], .5, 4 * se);
    EXPECT_NEAR(oap[1], .5, 4 * se);
  }

  TEST_F(OptimalArmProbabilityTest, LinearBanditTest) {

    NEW(CatKey, stooge_key)({"Larry", "Moe", "Curly", "Shemp"});
    NEW(EffectsEncoder, stooge_encoder)("Stooge", stooge_key);

    NEW(CatKey, color_key)({"Red", "Blue", "Green"});
    NEW(EffectsEncoder, color_encoder)("Color", color_key);

    NEW(IdentityEncoder, context_encoder)("Context");

    // The model is Y ~ intercept + Larry + Moe + Curly + context, so the
    // predictor dimension is 5

    DatasetEncoder encoder;
    encoder.add_encoder(stooge_encoder);
    encoder.add_encoder(context_encoder);

    int sample_size = 100;
    int niter = 10000;
    Vector context = rnorm_vector(sample_size, 0, 1);

    Vector true_coefficients = {1.8, .5, 1.0, -.5, 5.3};
    Matrix draws(niter, true_coefficients.size());
    for (int i = 0; i < true_coefficients.size(); ++i) {
      draws.col(i) = rnorm_vector(niter, true_coefficients[i], .1);
    }

    DataTable arm_definitions_table;
    DataTable context_table;

    Matrix oap = compute_user_specific_optimal_arm_probabilities_linear_bandit(
        draws,
        arm_definitions_table,
        context_table,
        encoder);
  }

}  // namespace
