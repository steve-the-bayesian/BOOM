#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Eigen.hpp"
#include "distributions.hpp"
#include "cpputil/math_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  template <class V1, class V2>
  bool VectorEquals(const V1 &v1, const V2 &v2) {
    Vector v = v1 - v2;
    return v.max_abs() < 1e-8;
  }

  template <class M1, class M2>
  bool MatrixEquals(const M1 &m1, const M2 &m2) {
    Matrix m = m1 - m2;
    return m.max_abs() < 1e-8;
  }
  
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
  
}  // namespace
