#include "gtest/gtest.h"

#include "Models/Glm/VariableSelectionPrior.hpp"
#include "LinAlg/Selector.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class VariableSelectionPriorTest : public ::testing::Test {
   protected:
    VariableSelectionPriorTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(VariableSelectionPriorTest, Prior) {
    VariableSelectionPrior prior;
    EXPECT_EQ(0, prior.potential_nvars());
    Selector empty;
    EXPECT_EQ(0.0, prior.logp(empty));
    EXPECT_EQ(0, prior.potential_nvars());

    Vector prior_inclusion_probs = {1.0, 0.0, .25};
    prior.set_prior_inclusion_probabilities(prior_inclusion_probs);
    EXPECT_EQ(3, prior.potential_nvars());
    
    Selector constrained("101");
    EXPECT_TRUE(std::isfinite(prior.logp(constrained)));
    constrained.flip(0);
    EXPECT_FALSE(std::isfinite(prior.logp(constrained)))
        << "001 omits a variable with prior probability 1.";
    constrained.flip(0);
    EXPECT_TRUE(std::isfinite(prior.logp(constrained)));

    constrained.flip(1);
    EXPECT_FALSE(std::isfinite(prior.logp(constrained)))
        << "111 includes a variable with prior probability 0.";
    constrained.flip(1);
    EXPECT_TRUE(std::isfinite(prior.logp(constrained)));
    constrained.flip(2);
    EXPECT_TRUE(std::isfinite(prior.logp(constrained)));

    // Check that make_valid works as expected.
    EXPECT_TRUE(VectorEquals(prior_inclusion_probs,
                             prior.prior_inclusion_probabilities()));
    constrained = Selector("111");
    prior.make_valid(constrained);
    EXPECT_TRUE(constrained[0]);
    EXPECT_FALSE(constrained[1]);
    EXPECT_TRUE(constrained[2]);

    constrained = Selector("000");
    prior.make_valid(constrained);
    EXPECT_TRUE(constrained[0]);
    EXPECT_FALSE(constrained[1]);
    EXPECT_FALSE(constrained[2]);

    constrained = Selector("011");
    prior.make_valid(constrained);
    EXPECT_TRUE(constrained[0]);
    EXPECT_FALSE(constrained[1]);
    EXPECT_TRUE(constrained[2]);

    // Check that log probabilities get updated when prior probabilities get
    // set.
    prior.set_prior_inclusion_probabilities(Vector{.25, .5, .3});
    EXPECT_DOUBLE_EQ(prior.logp(Selector("111")),
                     log(.25) + log(.5) + log(.3));

    prior.set_prior_inclusion_probabilities(Vector{.25, .5, .3});
    EXPECT_DOUBLE_EQ(prior.logp(Selector("110")),
                     log(.25) + log(.5) + log(.7));

  }

  TEST_F(VariableSelectionPriorTest, MatrixPrior) {
    Matrix prior_probs(2, 3);
    prior_probs.randomize();

    MatrixVariableSelectionPrior prior(prior_probs);
    EXPECT_EQ(2, prior_probs.nrow());
    EXPECT_EQ(3, prior_probs.ncol());

    SelectorMatrix inc(2, 3, true);
    inc.flip(1, 1);
    inc.flip(1, 2);

    EXPECT_DOUBLE_EQ(
        prior.logp(inc),
        log(prior_probs(0, 0)) + log(prior_probs(0, 1)) + log(prior_probs(0, 2))
        + log(prior_probs(1, 0)) + log(1 - prior_probs(1, 1))
        + log(1 - prior_probs(1, 2)));
    
  }
  
  
}  // namespace
