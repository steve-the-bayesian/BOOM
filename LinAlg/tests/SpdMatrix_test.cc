#include "gtest/gtest.h"
#include "distributions.hpp"
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
  
  class SpdMatrixTest : public ::testing::Test {
   protected:
    SpdMatrixTest() {
      GlobalRng::rng.seed(8675309);
    }
  };


  TEST_F(SpdMatrixTest, Inv) {
    SpdMatrix Sigma(4);
    Sigma.randomize();

    SpdMatrix siginv = Sigma.inv();

    SpdMatrix I(4);
    I.diag() = 1.0;

    EXPECT_TRUE(MatrixEquals(Sigma * siginv, I))
        << "Sigma = " << endl << Sigma << endl
        << "siginv = " << endl << siginv << endl
        << "Sigma * siginv = " << endl
        << Sigma * siginv << endl;

    SpdMatrix Sigma_copy(Sigma);
    EXPECT_TRUE(MatrixEquals(Sigma, Sigma_copy))
        << "Sigma = " << endl << Sigma << endl
        << "Sigma_copy = " << endl << Sigma_copy;
  }
  
}  // namespace
