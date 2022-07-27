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
  }

}  // namespace
