#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/Matrix.hpp"
#include "distributions.hpp"
#include "cpputil/math_utils.hpp"
#include "test_utils/test_utils.hpp"
#include <fstream>
#include <limits>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  class VectorTest : public ::testing::Test {
   protected:
    VectorTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(VectorTest, Constructors) {
    Vector x1;
    EXPECT_EQ(x1.size(), 0);
    EXPECT_TRUE(x1.empty());

    Vector x2(3, 2.7);
    EXPECT_EQ(3, x2.size());
    EXPECT_DOUBLE_EQ(2.7, x2[0]);
    EXPECT_DOUBLE_EQ(2.7, x2[1]);
    EXPECT_DOUBLE_EQ(2.7, x2[2]);

    Vector x3("8 6 7 5 3 0 9");
    EXPECT_EQ(7, x3.size());
    EXPECT_DOUBLE_EQ(x3[3], 5.0);

    Vector x4("8, 6, 7, 5, 3, 0, 9");
    EXPECT_TRUE(VectorEquals(x3, x4));

    Vector x5 = {8, 6, 7, 5, 3, 0, 9};
    EXPECT_TRUE(VectorEquals(x5, x3));

    std::vector<double> vector_data = {8, 6, 7, 5, 3, 0, 9};
    Vector x6(vector_data);
    EXPECT_TRUE(VectorEquals(x6, x5));

    Vector x7(vector_data.begin(), vector_data.end());
    EXPECT_TRUE(VectorEquals(x7, x6));

    VectorView x3_view(x3);
    x3[0] = 7;
    Vector x8(x3_view);
    EXPECT_TRUE(VectorEquals(x8, x3));

    ConstVectorView x3_const_view(x3);
    Vector x9(x3_const_view);
    EXPECT_TRUE(VectorEquals(x9, x3));
    
  }
  

  TEST_F(VectorTest, ZeroAndOne) {
    Vector x(3);
    x.randomize();
    x.set_to_zero();
    EXPECT_EQ(x.size(), 3);
    EXPECT_DOUBLE_EQ(x[0], 0.0);
    EXPECT_DOUBLE_EQ(x[1], 0.0);
    EXPECT_DOUBLE_EQ(x[2], 0.0);

    Vector y;
    y.set_to_zero();
    EXPECT_EQ(0, y.size());

    Vector one = x.one();
    EXPECT_EQ(one.size(), x.size());
    EXPECT_DOUBLE_EQ(one[0], 1.0);
    EXPECT_DOUBLE_EQ(one[1], 1.0);
    EXPECT_DOUBLE_EQ(one[2], 1.0);

    Vector zero = one.zero();
    EXPECT_EQ(zero.size(), one.size());
    EXPECT_DOUBLE_EQ(0.0, zero[0]);
    EXPECT_DOUBLE_EQ(0.0, zero[1]);
    EXPECT_DOUBLE_EQ(0.0, zero[2]);
  }

  TEST_F(VectorTest, Concat) {
    Vector x = {1, 2, 3};
    Vector y = {4, 5};

    x.concat(y);
    EXPECT_EQ(x.size(), 5);
    EXPECT_DOUBLE_EQ(x[0], 1);
    EXPECT_DOUBLE_EQ(x[1], 2);
    EXPECT_DOUBLE_EQ(x[2], 3);
    EXPECT_DOUBLE_EQ(x[3], 4);
    EXPECT_DOUBLE_EQ(x[4], 5);

    x.push_back(6);
    EXPECT_EQ(x.size(), 6);
    EXPECT_EQ(x.back(), 6);
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

  TEST_F(VectorTest, ScalarOperators) {
    Vector x(3);
    x.randomize();
    double y = 2.6;

    Vector original_x(x);
    x += y;
    EXPECT_DOUBLE_EQ(x[0], original_x[0] + y);
    EXPECT_DOUBLE_EQ(x[1], original_x[1] + y);
    EXPECT_DOUBLE_EQ(x[2], original_x[2] + y);

    x = original_x;
    x -= y;
    EXPECT_DOUBLE_EQ(x[0], original_x[0] - y);
    EXPECT_DOUBLE_EQ(x[1], original_x[1] - y);
    EXPECT_DOUBLE_EQ(x[2], original_x[2] - y);

    x = original_x;
    x *= y;
    EXPECT_DOUBLE_EQ(x[0], original_x[0] * y);
    EXPECT_DOUBLE_EQ(x[1], original_x[1] * y);
    EXPECT_DOUBLE_EQ(x[2], original_x[2] * y);

    x = original_x;
    x /= y;
    EXPECT_DOUBLE_EQ(x[0], original_x[0] / y);
    EXPECT_DOUBLE_EQ(x[1], original_x[1] / y);
    EXPECT_DOUBLE_EQ(x[2], original_x[2] / y);
  }

  TEST_F(VectorTest, AllFinite) {
    Vector x(3);
    x.randomize();
    EXPECT_TRUE(x.all_finite());

    x[0] = infinity();
    EXPECT_FALSE(x.all_finite());

    x[0] = 2.7;
    x[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(x.all_finite());
  }

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

  TEST_F(VectorTest, Axpy) {
    Vector x(3);
    x.randomize();
    Vector y(3);
    y.randomize();

    Vector original_x(x);
    x.axpy(y, 2.7);
    EXPECT_DOUBLE_EQ(x[0], original_x[0] + 2.7 * y[0]);
    EXPECT_DOUBLE_EQ(x[1], original_x[1] + 2.7 * y[1]);
    EXPECT_DOUBLE_EQ(x[2], original_x[2] + 2.7 * y[2]);

    VectorView y_view(y);
    x = original_x;
    x.axpy(y_view, 3.1);
    EXPECT_DOUBLE_EQ(x[0], original_x[0] + 3.1 * y[0]);
    EXPECT_DOUBLE_EQ(x[1], original_x[1] + 3.1 * y[1]);
    EXPECT_DOUBLE_EQ(x[2], original_x[2] + 3.1 * y[2]);

    ConstVectorView y_const_view(y);
    x = original_x;
    x.axpy(y_const_view, 5.2);
    EXPECT_DOUBLE_EQ(x[0], original_x[0] + 5.2 * y[0]);
    EXPECT_DOUBLE_EQ(x[1], original_x[1] + 5.2 * y[1]);
    EXPECT_DOUBLE_EQ(x[2], original_x[2] + 5.2 * y[2]);
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

    SpdMatrix V = v.outer();
    EXPECT_EQ(V.nrow(), v.size());
    EXPECT_DOUBLE_EQ(V(0, 0), v[0] * v[0]);
    EXPECT_DOUBLE_EQ(V(0, 1), v[0] * v[1]);
    EXPECT_DOUBLE_EQ(V(0, 2), v[0] * v[2]);
    EXPECT_DOUBLE_EQ(V(1, 0), v[1] * v[0]);
    EXPECT_DOUBLE_EQ(V(1, 1), v[1] * v[1]);
    EXPECT_DOUBLE_EQ(V(1, 2), v[1] * v[2]);
    EXPECT_DOUBLE_EQ(V(2, 0), v[2] * v[0]);
    EXPECT_DOUBLE_EQ(V(2, 1), v[2] * v[1]);
    EXPECT_DOUBLE_EQ(V(2, 2), v[2] * v[2]);
  }

  TEST_F(VectorTest, Normalize) {
    Vector x(3);
    x.randomize();
    Vector original_x(x);
    
    x.normalize_prob();
    EXPECT_DOUBLE_EQ(x.sum(), 1.0);

    EXPECT_DOUBLE_EQ(x[0] / x[1],
                     original_x[0] / original_x[1]);
    
    x[1] = -2;
    EXPECT_THROW(x.normalize_prob(), std::runtime_error);

    x.randomize();
    original_x = x;
    x.normalize_logprob();
    double total = exp(original_x[0]) + exp(original_x[1]) + exp(original_x[2]);
    EXPECT_DOUBLE_EQ(x[0], exp(original_x[0]) / total);
    EXPECT_DOUBLE_EQ(x[1], exp(original_x[1]) / total);
    EXPECT_DOUBLE_EQ(x[2], exp(original_x[2]) / total);

    x.randomize();
    original_x = x;
    x.normalize_L2();
    EXPECT_DOUBLE_EQ(x.dot(x), 1.0);
    EXPECT_DOUBLE_EQ(x[0] / x[1],
                     original_x[0] / original_x[1]);
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

  // Given two vector-like objects x and y, check the field operations between
  // them and between the first and a scalar.
  // Args:
  //   x, y: The vectors to compare.
  //   msg: A message to print in the event of an error, identifying the case
  //     being compared.  E.g. "vector-vector"
  template <class V1, class V2>
  void CheckFieldOperators(const V1 &x, const V2 &y, const std::string &msg) {
    ASSERT_GE(x.size(), 3) << "Need to pass a vector of size >= 3.";

    Vector z = x + y;
    double a = 1.2;
    EXPECT_DOUBLE_EQ(z[0], x[0] + y[0] ) << msg;
    EXPECT_DOUBLE_EQ(z[1], x[1] + y[1] ) << msg;
    EXPECT_DOUBLE_EQ(z[2], x[2] + y[2] ) << msg;

    z = x - y;
    EXPECT_DOUBLE_EQ(z[0], x[0] - y[0] ) << msg;
    EXPECT_DOUBLE_EQ(z[1], x[1] - y[1] ) << msg;
    EXPECT_DOUBLE_EQ(z[2], x[2] - y[2] ) << msg;

    z = x * y;
    EXPECT_DOUBLE_EQ(z[0], x[0] * y[0] ) << msg;
    EXPECT_DOUBLE_EQ(z[1], x[1] * y[1] ) << msg;
    EXPECT_DOUBLE_EQ(z[2], x[2] * y[2] ) << msg;

    z = x / y;
    EXPECT_DOUBLE_EQ(z[0], x[0] / y[0] ) << msg;
    EXPECT_DOUBLE_EQ(z[1], x[1] / y[1] ) << msg;
    EXPECT_DOUBLE_EQ(z[2], x[2] / y[2] ) << msg;

    z = a + x;
    EXPECT_DOUBLE_EQ(z[0], x[0] + a ) << msg;
    EXPECT_DOUBLE_EQ(z[1], x[1] + a ) << msg;
    EXPECT_DOUBLE_EQ(z[2], x[2] + a ) << msg;

    z = a - x;
    EXPECT_DOUBLE_EQ(z[0], a - x[0] ) << msg;
    EXPECT_DOUBLE_EQ(z[1], a - x[1] ) << msg;
    EXPECT_DOUBLE_EQ(z[2], a - x[2] ) << msg;

    z = x + a;
    EXPECT_DOUBLE_EQ(z[0], x[0] + a ) << msg;
    EXPECT_DOUBLE_EQ(z[1], x[1] + a ) << msg;
    EXPECT_DOUBLE_EQ(z[2], x[2] + a ) << msg;

    z = x - a;
    EXPECT_DOUBLE_EQ(z[0], x[0] - a ) << msg;
    EXPECT_DOUBLE_EQ(z[1], x[1] - a ) << msg;
    EXPECT_DOUBLE_EQ(z[2], x[2] - a ) << msg;

    z = a * x;
    EXPECT_DOUBLE_EQ(z[0], x[0] * a ) << msg;
    EXPECT_DOUBLE_EQ(z[1], x[1] * a ) << msg;
    EXPECT_DOUBLE_EQ(z[2], x[2] * a ) << msg;

    z = a / x;
    EXPECT_DOUBLE_EQ(z[0], a / x[0] ) << msg;
    EXPECT_DOUBLE_EQ(z[1], a / x[1] ) << msg;
    EXPECT_DOUBLE_EQ(z[2], a / x[2] ) << msg;

    z = x * a;
    EXPECT_DOUBLE_EQ(z[0], x[0] * a ) << msg;
    EXPECT_DOUBLE_EQ(z[1], x[1] * a ) << msg;
    EXPECT_DOUBLE_EQ(z[2], x[2] * a ) << msg;

    z = x / a;
    EXPECT_DOUBLE_EQ(z[0], x[0] / a ) << msg;
    EXPECT_DOUBLE_EQ(z[1], x[1] / a ) << msg;
    EXPECT_DOUBLE_EQ(z[2], x[2] / a ) << msg;
  }

  TEST_F(VectorTest, FieldOperators) {
    Vector x(3);
    Vector y(3);
    x.randomize();
    y.randomize();
    VectorView xview(x);
    VectorView yview(y);
    ConstVectorView cxview(x);
    ConstVectorView cyview(y);

    CheckFieldOperators(x, y, "vector, vector");
    CheckFieldOperators(xview, y, "view, vector");
    CheckFieldOperators(cxview, y, "const view, vector");
    CheckFieldOperators(x, yview, "vector, view");
    CheckFieldOperators(xview, yview, "view, view");
    CheckFieldOperators(cxview, yview, "const view, view");
    CheckFieldOperators(x, cyview, "vector, const view");
    CheckFieldOperators(xview, cyview, "view, const view");
    CheckFieldOperators(cxview, cyview, "const view, const view");

  }
  
}  // namespace
