#include "gtest/gtest.h"
#include "TargetFun/LogitTransform.hpp"
#include "TargetFun/JacobianChecker.hpp"
#include "LinAlg/Matrix.hpp"
#include "test_utils/test_utils.hpp"

#include "numopt/NumericalDerivatives.hpp"
#include "stats/logit.hpp"

namespace {
  using namespace BOOM;

  using Mapping = std::function<Vector(const Vector &)>;

  class LogitTransformTest : public ::testing::Test {
   protected:
    LogitTransformTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(LogitTransformTest, BasicTransform) {
  }

  TEST_F(LogitTransformTest, Derivatives) {
    LogitTransformJacobian jake;
    ScalarNumericalDerivatives nd([](double x) {return BOOM::logit_inv(x);});

    double prob = .6;
    double logit_value = logit(prob);
    double d1 = prob * (1 - prob);
    EXPECT_NEAR(d1, nd.first_derivative(logit_value), 1e-5);

    double d2 = d1 * (1 - 2 * prob);
    EXPECT_NEAR(d2, nd.second_derivative(logit_value), 1e-5);
  }

  // Debugging tools.  Left here in case errors surface again.
  //
  // std::string PrintErrorMessages(const std::vector<std::string> &msg) {
  //   std::ostringstream err;
  //   for (const auto &el : msg) {
  //     err << el << "\n";
  //   }
  //   return err.str();
  // }

  TEST_F(LogitTransformTest, JacobianTest) {
    Vector probs = {.3, .8};
    Vector logits = logit(probs);

    Mapping transform = [](const Vector &probs) {return logit(probs);};
    Mapping inverse_transform =
        [](const Vector &logits) { return logit_inv(logits); };
    std::shared_ptr<LogitTransformJacobian> jake(new LogitTransformJacobian);

    NumericJacobian numeric_jacobian_computer(inverse_transform);
    Matrix expected = jake->matrix(probs);
    Matrix numeric_jacobian = numeric_jacobian_computer.matrix(logits);

    JacobianChecker checker(transform, inverse_transform, jake, 1e-8);

    EXPECT_TRUE(checker.check_matrix(logits))
        << "Jacobian matrix = \n" << jake->matrix(probs)
        << "Expected matrix = \n" << expected
        << "Inverse of expected matrix =  \n" << expected.inv()
        << "Numeric Jacobian: \n" << numeric_jacobian;

    EXPECT_TRUE(checker.check_logdet(logits));

  }
}  // namespace
