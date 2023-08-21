#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
#include "LinAlg/Eigen.hpp"
#include "distributions.hpp"
#include "cpputil/math_utils.hpp"
#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class EigenTest : public ::testing::Test {
   protected:
    EigenTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(EigenTest, ValuesAndVectors) {
    int dim = 10;
    Matrix A(dim, dim);
    A.randomize();
    cout << "Starting decomposition..." << endl;
    EigenDecomposition eigen_a(A, true);

    cout << "Accessing eigenvalues." << endl;
    std::vector<std::complex<double>> values = eigen_a.eigenvalues();
    EXPECT_EQ(values.size(), dim);
    EXPECT_EQ(dim, eigen_a.real_eigenvalues().size());
    EXPECT_EQ(dim, eigen_a.imaginary_eigenvalues().size());

    for (int i = 0; i < dim; ++i) {
      std::complex<double> lambda = values[i];
      Vector re = eigen_a.real_eigenvector(i);
      Vector im = eigen_a.imaginary_eigenvector(i);
      EXPECT_TRUE(VectorEquals(lambda.real() * re - lambda.imag() * im,
                               A * re));
      EXPECT_TRUE(VectorEquals(lambda.imag() * re + lambda.real() * im,
                               A * im));
    }

    EigenDecomposition values_only(A, false);
    EXPECT_EQ(dim, values_only.eigenvalues().size());
    EXPECT_EQ(dim, values_only.real_eigenvalues().size());
    EXPECT_EQ(dim, values_only.imaginary_eigenvalues().size());
  }

  TEST_F(EigenTest, SpdValuesOnly) {

    SpdMatrix V0(2);
    V0(0, 0) = 2;
    V0(1, 1) = 2;
    V0(0, 1) = 1;
    V0(1, 0) = 1;
    SymmetricEigen eigen0(V0, true);
    EXPECT_EQ(2, eigen0.eigenvalues().size());
    EXPECT_DOUBLE_EQ(1.0, eigen0.eigenvalues().min());
    EXPECT_DOUBLE_EQ(3.0, eigen0.eigenvalues().max());

    int dim = 4;
    SpdMatrix V(dim);
    V.randomize();

    SymmetricEigen values_only(V, false);
    EXPECT_EQ(0, values_only.eigenvectors().nrow());
    EXPECT_EQ(0, values_only.eigenvectors().ncol());

    SymmetricEigen both(V, true);
    EXPECT_TRUE(VectorEquals(values_only.eigenvalues(), both.eigenvalues()));

    DiagonalMatrix D(both.eigenvalues());

    EXPECT_TRUE(VectorEquals(V * both.eigenvectors().col(3),
                             both.eigenvalues()[3] *
                             both.eigenvectors().col(3)))
        << "V * eigenvectors().col(3) = " << endl
        << V * both.eigenvectors().col(3) << endl
        << " lambda(3) * eigenvectors.col(3) = " << endl
        << both.eigenvalues()[3] * both.eigenvectors().col(3);

    EXPECT_TRUE(MatrixEquals(V * both.eigenvectors(),
                             both.eigenvectors() * D))
        << "V * eigenvectors: " << endl
        << V * both.eigenvectors() << endl
        << "eigenvectors * D: " << endl
        << both.eigenvectors() * D;
  }

  TEST_F(EigenTest, SymmetricEigenTest) {
    SpdMatrix blah(3);
    blah.randomize();

    SymmetricEigen eigen(blah, true);
    EXPECT_TRUE(MatrixEquals(blah, eigen.original_matrix()));

    blah(2, 2) = -1.0;
    SymmetricEigen eigen2(blah, true);
    SpdMatrix fixed = eigen2.closest_positive_definite();
    EXPECT_TRUE(fixed.is_pos_def())
        << "matrix should be positive definite:\n"
        << fixed;

    // Verify that eigenvalues are always given in increasing order.
    for (int i = 0; i < 100; ++i) {
      int dim = rpois(10);
      SpdMatrix S(dim);
      S.randomize();

      SymmetricEigen dcmp(S);
      const Vector &values(dcmp.eigenvalues());
      for (int j = 1; j < dim; ++j) {
        EXPECT_LE(values[j-1], values[j]);
      }
    }
  }

  TEST_F(EigenTest, GenInverseTest) {
    SpdMatrix S(5);
    S.randomize();

    SymmetricEigen eigen(S);
    SpdMatrix Sinv = eigen.generalized_inverse(1e-8);
    EXPECT_TRUE(MatrixEquals(S * Sinv, SpdMatrix(5, 1.0)))
        << "Original Matrix: \n"
        << S << "\n"
        << "Inverse: \n"
        << Sinv << "\n"
        << "product: \n"
        << S * Sinv;

    Matrix A(8, 5);
    A.randomize();

    SpdMatrix V = A * S * A.transpose();
    SymmetricEigen dcmp(V);
    SpdMatrix Vinv = dcmp.generalized_inverse();

    EXPECT_TRUE(MatrixEquals(V, V * Vinv * V))
        << "Original Matrix: \n"
        << V << "\n"
        << "Generalized Inverse: \n"
        << Vinv << "\n"
        << "product V * Vinv * V: \n"
        << V * Vinv * V;

    EXPECT_TRUE(MatrixEquals(Vinv, Vinv * V * Vinv))
        << "Generalized Inverse: \n"
        << Vinv << "\n"
        << "Original Matrix: \n"
        << V << "\n"
        << "product Vinv * V * Vinv: \n"
        << Vinv * V * Vinv;
  }

  TEST_F(EigenTest, GeneralizedDeterminantTest) {
    SpdMatrix S(3);
    S.randomize();

    SymmetricEigen dcmp(S);
    EXPECT_NEAR(dcmp.generalized_inverse_logdet(), -S.logdet(), 1e-5);

    Matrix A(8, 3);
    A.randomize();
    SpdMatrix V = A * S * A.transpose();

    SymmetricEigen eigen(V);
    EXPECT_TRUE(std::isfinite(eigen.generalized_inverse_logdet()));
    EXPECT_FALSE(std::isfinite(V.logdet()));
  }

}  // namespace
