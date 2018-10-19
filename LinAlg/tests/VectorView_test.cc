#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/Matrix.hpp"
#include "distributions.hpp"
#include "cpputil/math_utils.hpp"
#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
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

  TEST_F(VectorViewTest, SumLog) {
    double x1 = sumlog(v_);
    double x2 = sum(log(v_));
    EXPECT_DOUBLE_EQ(x1, x2);
  }
  
}  // namespace
