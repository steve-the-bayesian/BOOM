#include "gtest/gtest.h"
#include "TargetFun/SumMultinomialLogitTransform.hpp"
#include "TargetFun/JacobianChecker.hpp"
#include "LinAlg/Matrix.hpp"
#include "test_utils/test_utils.hpp"

#include "numopt/NumericalDerivatives.hpp"

namespace {
  using namespace BOOM;

  using Mapping = std::function<Vector(const Vector &)>;

  class NumericalLogdet {
   public:

    double operator()(const Vector &sum_logits) const {
      SumMultinomialLogitTransform transform;
      SumMultinomialLogitJacobian jake;
      Vector positive_numbers = transform.from_sum_logits(sum_logits);
      return jake.logdet(positive_numbers);
    }

   private:

  };

  class SumMultinomialLogitTransformTest : public ::testing::Test {
   protected:
    SumMultinomialLogitTransformTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(SumMultinomialLogitTransformTest, BasicTransform) {
    Vector positive_numbers = {2, 3, 5};

    SumMultinomialLogitTransform tf;
    Vector sum_logits = tf.to_sum_logits(positive_numbers);

    EXPECT_EQ(3, sum_logits.size());
    Vector restored = tf.from_sum_logits(sum_logits);
    EXPECT_TRUE(VectorEquals(positive_numbers, restored));

    EXPECT_NEAR(sum_logits[0], 10.0, 1e-6);
    EXPECT_NEAR(sum_logits[1], log(1.5), 1e-6);
    EXPECT_NEAR(sum_logits[2], log(2.5), 1e-6);
  }

  TEST_F(SumMultinomialLogitTransformTest, JacobianTest) {
    Vector positive_numbers = {2, 3, 5, 4};
    SumMultinomialLogitTransform tf;

    Mapping inverse_transformation(
        [&tf](const Vector &sum_logits) {
          return tf.from_sum_logits(sum_logits);
        });
    Mapping transformation(
        [&tf](const Vector &positive_numbers) {
          return tf.to_sum_logits(positive_numbers);});

    std::shared_ptr<SumMultinomialLogitJacobian> jake(
        new SumMultinomialLogitJacobian);

    JacobianChecker checker(transformation, inverse_transformation, jake, 1e-8);
    Vector sum_logits = tf.to_sum_logits(positive_numbers);

    double total = positive_numbers.sum();
    Vector probs = positive_numbers / total;
    double p1 = positive_numbers[1] / total;
    double p2 = positive_numbers[2] / total;

    Matrix expected = diag(probs) - outer(probs);
    expected *= total;
    expected.row(0) = probs;
    EXPECT_TRUE(MatrixEquals(expected, jake->matrix(positive_numbers)));

    // Check that 'element' returns the same values found in the matrix.
    for (int i = 0; i < expected.nrow(); ++i) {
      for (int j = 0; j < expected.ncol(); ++j) {
        EXPECT_NEAR(expected(i, j),
                    jake->element(i, j, positive_numbers),
                    1e-6);
      }
    }

    NumericJacobian numeric_jacobian_computer(inverse_transformation);
    Matrix numeric_jacobian = numeric_jacobian_computer.matrix(sum_logits);

    EXPECT_TRUE(checker.check_matrix(sum_logits))
        << "Jacobian matrix = \n" << jake->matrix(positive_numbers)
        << "Expected matrix = \n" << expected
        << "Inverse of expected matrix =  \n" << expected.inv()
        << "Numeric Jacobian: \n" << numeric_jacobian;

    // Check that the logdet() method really does return the log determinant of
    // the Jacobian matrix.
    EXPECT_NEAR(log(fabs(jake->matrix(positive_numbers).det())),
                jake->logdet(positive_numbers),
                1e-6);

    // Check that the inverse matrix really is the inverse of the matrix.
    EXPECT_TRUE(MatrixEquals(
        jake->matrix(positive_numbers) * jake->inverse_matrix(positive_numbers),
        SpdMatrix(positive_numbers.size(), 1.0)))
        << "Jacobian matrix is not the numeric inverse of the "
        << "'inverse Jacobian matrix.'";

    double H_1_2_2 = jake->second_order_element(1, 2, 2, positive_numbers);
    EXPECT_NEAR(H_1_2_2,
                -total * p1 * p2 * (1 - 2 * p2),
                1e-5);

    EXPECT_NEAR(jake->second_order_element(1, 2, 1, positive_numbers),
                -total * (-p1 * p1 * p2 + p2 * (p1 - p1 * p1)),
                1e-5);
    EXPECT_NEAR(jake->second_order_element(2, 0, 1, positive_numbers),
                -p2 * p1,
                1e-5);

    checker.set_epsilon(1e-3);
    EXPECT_EQ("", checker.check_second_order_elements(sum_logits));
  }

  TEST_F(SumMultinomialLogitTransformTest, CheckLogdetDerivs) {
    Vector positive_numbers = {2, 3, 5};
    SumMultinomialLogitTransform transform;
    SumMultinomialLogitJacobian jake;
    Vector sum_logits = transform.to_sum_logits(positive_numbers);

    NumericalLogdet numerical_logdet;
    NumericalDerivatives logdet_derivs(numerical_logdet);
    Vector analytic_gradient(positive_numbers.size(), 0.0);
    jake.add_logdet_gradient(analytic_gradient, positive_numbers);
    Vector numeric_gradient = logdet_derivs.gradient(sum_logits);

    EXPECT_TRUE(VectorEquals(numeric_gradient, analytic_gradient))
        << "\nnumeric_gradient: " << numeric_gradient << "\n"
        << "analytic_gradient: " << analytic_gradient<< "\n";

    Matrix analytic_hessian(3, 3, 0.0);
    jake.add_logdet_Hessian(analytic_hessian, positive_numbers);
    Matrix numeric_hessian = logdet_derivs.Hessian(sum_logits);
    EXPECT_TRUE(MatrixEquals(analytic_hessian, numeric_hessian, 1e-3))
        << "\nAnalytic Hessian:\n"
        << analytic_hessian << "\n"
        << "Numeric Hessian:\n"
        << numeric_hessian;
  }

}  // namespace
