#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "stats/logit.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  
  TEST(multinomial_logit, works_as_intended) {
    Vector probs = {.2, .3, .5};
    Vector logits = multinomial_logit(probs);
    EXPECT_TRUE(VectorEquals(
        probs,
        multinomial_logit_inverse(logits)))
        << endl
        << "probs = " << probs << endl
        << "logits = " << logits << endl
        << "recovered_probs = " << multinomial_logit_inverse(logits);

    // Check that arbitrary values can be converted into probabilities.
    logits = Vector({12.3, -17, 2.1, 0, 92});
    probs = multinomial_logit_inverse(logits);
    EXPECT_EQ(probs.size(), logits.size() + 1);
    EXPECT_NEAR(probs.sum(), 1.0, 1e-8);
  }
  
}  // namespace
