#include "gtest/gtest.h"
#include "distributions.hpp"
#include "LinAlg/Cholesky.hpp"
#include "LinAlg/SpdMatrix.hpp"
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
  
  class CholeskyTest : public ::testing::Test {
   protected:
    CholeskyTest()
        : spd_(4)
    {
      GlobalRng::rng.seed(8675309);
      spd_.randomize();
    }
    SpdMatrix spd_;
  };

  TEST_F(CholeskyTest, WholeClass) {
    SpdMatrix spd(4);
    spd.randomize();

    Chol cholesky(spd);
    EXPECT_EQ(4, cholesky.nrow());
    EXPECT_EQ(4, cholesky.ncol());
    EXPECT_EQ(4, cholesky.dim());

    Matrix L = cholesky.getL();
    Matrix LT = cholesky.getLT();

    EXPECT_TRUE(MatrixEquals(L.t(), LT));
    EXPECT_TRUE(MatrixEquals(spd, L * LT))
        << "original matrix = " << endl << spd
        << "recovered matrix = " << endl << L * LT << endl;

    SpdMatrix spd_inverse = cholesky.inv();
    EXPECT_TRUE(MatrixEquals(spd * spd_inverse, SpdMatrix(4, 1.0)));

    Vector v(4);
    v.randomize();

    Vector x = cholesky.solve(v);
    EXPECT_TRUE(VectorEquals(spd * x, v));

    EXPECT_TRUE(MatrixEquals(spd, cholesky.original_matrix()))
        << "Original matrix = " << endl << spd << endl
        << "Recovered matrix = " << endl << cholesky.original_matrix() << endl;
    
    EXPECT_NEAR(cholesky.logdet(),
                log(fabs(spd.Matrix::det())),
                1e-8);

    EXPECT_NEAR(cholesky.det(),
                fabs(spd.Matrix::det()),
                1e-8);
  }
  
}  // namespace
