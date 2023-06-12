#include "gtest/gtest.h"
#include "TargetFun/MultinomialLogitTransform.hpp"
#include "TargetFun/JacobianChecker.hpp"
#include "LinAlg/Matrix.hpp"
#include "test_utils/test_utils.hpp"

#include "numopt/NumericalDerivatives.hpp"

namespace {
  using namespace BOOM;

  using Mapping = std::function<Vector(const Vector &)>;

  class MultinomialLogitTransformTest : public ::testing::Test {
   protected:
    MultinomialLogitTransformTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(MultinomialLogitTransformTest, BasicTransform) {
    Vector probs = {.2, .3, .5};

    MultinomialLogitTransform tf;
    Vector logits = tf.to_logits(probs);

    EXPECT_EQ(2, logits.size());
    Vector restored_probs = tf.to_probs(logits);
    EXPECT_TRUE(VectorEquals(probs, restored_probs));

    EXPECT_NEAR(logits[0], log(1.5), 1e-6);
    EXPECT_NEAR(logits[1], log(2.5), 1e-6);
  }

  TEST_F(MultinomialLogitTransformTest, JacobianTest) {

    Vector probs = {.2, .3, .5};
    MultinomialLogitTransform tf;
    Vector truncated_probs = ConstVectorView(probs, 1);
    EXPECT_TRUE(VectorEquals(truncated_probs, Vector{.3, .5}));

    Mapping inverse_transformation(
        [&tf](const Vector &logits) {
          Vector probs = tf.to_probs(logits);
          return Vector(ConstVectorView(probs, 1));
        });
    Mapping transformation(
        [&tf](const Vector &truncated_probs) {
          return tf.to_logits(truncated_probs, true);});

    std::shared_ptr<MultinomialLogitJacobian> jake(new MultinomialLogitJacobian);

    JacobianChecker checker(transformation, inverse_transformation, jake, 1e-8);
    Vector logits = tf.to_logits(probs);
    Matrix expected = diag(truncated_probs) - outer(truncated_probs);
    EXPECT_TRUE(MatrixEquals(expected, jake->matrix(truncated_probs)));

    NumericJacobian numeric_jacobian_computer(inverse_transformation);
    Matrix numeric_jacobian = numeric_jacobian_computer.matrix(logits);

    EXPECT_TRUE(checker.check_matrix(logits))
        << "Jacobian matrix = \n" << jake->matrix(truncated_probs)
        << "Expected matrix = \n" << expected
        << "Inverse of expected matrix =  \n" << expected.inv()
        << "Numeric Jacobian: \n" << numeric_jacobian;

  }
}  // namespace
