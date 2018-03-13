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
  
  class VectorViewTest : public ::testing::Test {
   protected:
    VectorViewTest()
        : v_(3),
          v_view_(v_),
          cv_view_(v_),
          w_(3),
          w_view_(w_),
          cw_view_(w_)
    {
      GlobalRng::rng.seed(8675309);
      v_.randomize();
      w_.randomize();
    }
    Vector v_;
    VectorView v_view_;
    ConstVectorView cv_view_;

    Vector w_;
    VectorView w_view_;
    ConstVectorView cw_view_;
  };

  TEST_F(VectorViewTest, DotProduct) {
    EXPECT_DOUBLE_EQ(v_view_.dot(w_), v_.dot(w_));
    EXPECT_DOUBLE_EQ(v_view_.dot(w_view_), v_.dot(w_));
    EXPECT_DOUBLE_EQ(v_view_.dot(cw_view_), v_.dot(w_));

    EXPECT_DOUBLE_EQ(cv_view_.dot(w_), v_.dot(w_));
    EXPECT_DOUBLE_EQ(cv_view_.dot(w_view_), v_.dot(w_));
    EXPECT_DOUBLE_EQ(cv_view_.dot(cw_view_), v_.dot(w_));
  }

  TEST_F(VectorViewTest, Norm) {
    EXPECT_DOUBLE_EQ(v_.normsq(), v_view_.normsq());
    EXPECT_DOUBLE_EQ(v_.normsq(), cv_view_.normsq());
    
    EXPECT_DOUBLE_EQ(v_.abs_norm(), v_view_.abs_norm());
    EXPECT_DOUBLE_EQ(v_.abs_norm(), cv_view_.abs_norm());
  }

  TEST_F(VectorViewTest, Operators) {
    EXPECT_TRUE(VectorEquals(v_view_ + w_, v_ + w_));
    EXPECT_TRUE(VectorEquals(cv_view_ + w_, v_ + w_));
    EXPECT_TRUE(VectorEquals(v_view_ + w_view_, v_ + w_));
    EXPECT_TRUE(VectorEquals(cv_view_ + w_view_, v_ + w_));
    EXPECT_TRUE(VectorEquals(v_view_ + cw_view_, v_ + w_));
    EXPECT_TRUE(VectorEquals(cv_view_ + cw_view_, v_ + w_));

    EXPECT_TRUE(VectorEquals(v_view_ - w_, v_ - w_));
    EXPECT_TRUE(VectorEquals(cv_view_ - w_, v_ - w_));
    EXPECT_TRUE(VectorEquals(v_view_ - w_view_, v_ - w_));
    EXPECT_TRUE(VectorEquals(cv_view_ - w_view_, v_ - w_));
    EXPECT_TRUE(VectorEquals(v_view_ - cw_view_, v_ - w_));
    EXPECT_TRUE(VectorEquals(cv_view_ - cw_view_, v_ - w_));

    EXPECT_TRUE(VectorEquals(v_view_ * w_, v_ * w_));
    EXPECT_TRUE(VectorEquals(cv_view_ * w_, v_ * w_));
    EXPECT_TRUE(VectorEquals(v_view_ * w_view_, v_ * w_));
    EXPECT_TRUE(VectorEquals(cv_view_ * w_view_, v_ * w_));
    EXPECT_TRUE(VectorEquals(v_view_ * cw_view_, v_ * w_));
    EXPECT_TRUE(VectorEquals(cv_view_ * cw_view_, v_ * w_));

    EXPECT_TRUE(VectorEquals(v_view_ / w_, v_ / w_));
    EXPECT_TRUE(VectorEquals(cv_view_ / w_, v_ / w_));
    EXPECT_TRUE(VectorEquals(v_view_ / w_view_, v_ / w_));
    EXPECT_TRUE(VectorEquals(cv_view_ / w_view_, v_ / w_));
    EXPECT_TRUE(VectorEquals(v_view_ / cw_view_, v_ / w_));
    EXPECT_TRUE(VectorEquals(cv_view_ / cw_view_, v_ / w_));
  }
  
}  // namespace
