#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/Matrix.hpp"
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
  
  class VectorTest : public ::testing::Test {
   protected:
    VectorTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(VectorTest, DotProduct) {
    Vector v(4);
    v.randomize();

    Vector w(4);
    w.randomize();
    EXPECT_DOUBLE_EQ(v.dot(w),
                     v[0] * w[0] + v[1] * w[1] + v[2] * w[2] + v[3] * w[3]);

    EXPECT_DOUBLE_EQ(v.dot(w), v.dot(ConstVectorView(w)));
    VectorView w_view(w);
    EXPECT_DOUBLE_EQ(v.dot(w_view), v.dot(w));
  }

  TEST_F(VectorTest, SpdMult) {
    Vector v(4);
    v.randomize();

    SpdMatrix Sigma(4);
    Sigma.randomize();

    Vector ans(4);
    EXPECT_TRUE(VectorEquals(v.mult(Sigma, ans), Sigma * v));
  }

  TEST_F(VectorTest, Outer) {
    Vector v(3);
    Vector w(2);
    v.randomize();
    w.randomize();
    Matrix vw = v.outer(w, 2.7);
    EXPECT_DOUBLE_EQ(vw(0, 0), v[0] * w[0] * 2.7);
    EXPECT_DOUBLE_EQ(vw(1, 0), v[1] * w[0] * 2.7);
    EXPECT_DOUBLE_EQ(vw(2, 0), v[2] * w[0] * 2.7);
    EXPECT_DOUBLE_EQ(vw(0, 1), v[0] * w[1] * 2.7);
    EXPECT_DOUBLE_EQ(vw(1, 1), v[1] * w[1] * 2.7);
    EXPECT_DOUBLE_EQ(vw(2, 1), v[2] * w[1] * 2.7);
  }

  TEST_F(VectorTest, Norm) {
    Vector v(4);
    v.randomize();

    EXPECT_DOUBLE_EQ(v.normsq(),
                     v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
    EXPECT_DOUBLE_EQ(v.normalize_L2().normsq(), 1.0);
    EXPECT_DOUBLE_EQ(v.abs_norm(),
                     fabs(v[0]) + fabs(v[1]) + fabs(v[2]) + fabs(v[3]));
  }

  TEST_F(VectorTest, Operators) {
    Vector v(3);
    Vector w(3);
    v.randomize();
    w.randomize();

    Vector x = v + w;
    EXPECT_DOUBLE_EQ(x[0], v[0] + w[0]);
    EXPECT_DOUBLE_EQ(x[1], v[1] + w[1]);
    EXPECT_DOUBLE_EQ(x[2], v[2] + w[2]);

    x = v + ConstVectorView(w);
    EXPECT_TRUE(VectorEquals(x, v + w));
    VectorView w_view(w);
    EXPECT_TRUE(VectorEquals(v + w, v + w_view));

    x = v - w;
    EXPECT_DOUBLE_EQ(x[0], v[0] - w[0]);
    EXPECT_DOUBLE_EQ(x[1], v[1] - w[1]);
    EXPECT_DOUBLE_EQ(x[2], v[2] - w[2]);
    EXPECT_TRUE(VectorEquals(v - w, v - ConstVectorView(w)));
    EXPECT_TRUE(VectorEquals(v - w, v - w_view));

    x = v * w;
    EXPECT_DOUBLE_EQ(x[0], v[0] * w[0]);
    EXPECT_DOUBLE_EQ(x[1], v[1] * w[1]);
    EXPECT_DOUBLE_EQ(x[2], v[2] * w[2]);
    EXPECT_TRUE(VectorEquals(v * w, v * ConstVectorView(w)));
    EXPECT_TRUE(VectorEquals(v * w, v * w_view));

    x = v / w;
    EXPECT_DOUBLE_EQ(x[0], v[0] / w[0]);
    EXPECT_DOUBLE_EQ(x[1], v[1] / w[1]);
    EXPECT_DOUBLE_EQ(x[2], v[2] / w[2]);
    EXPECT_TRUE(VectorEquals(v / w, v / ConstVectorView(w)));
    EXPECT_TRUE(VectorEquals(v / w, v / w_view));
  }
  
}  // namespace
