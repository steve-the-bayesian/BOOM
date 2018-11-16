#include "gtest/gtest.h"
#include "Models/SpdData.hpp"
#include "distributions.hpp"

#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Cholesky.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class SpdDataTest : public ::testing::Test {
   protected:
    SpdDataTest()
        : dim_(3),
          variance_(dim_),
          precision_(dim_)
    {
      GlobalRng::rng.seed(8675309);
      variance_.randomize();
      precision_ = variance_.inv();
      variance_cholesky_.decompose(variance_);
      precision_cholesky_.decompose(precision_);
    }

    int dim_;
    SpdMatrix variance_;
    SpdMatrix precision_;
    Cholesky variance_cholesky_;
    Cholesky precision_cholesky_;
  };

  TEST_F(SpdDataTest, Construction) {
    SpdData d1(variance_, false);
    EXPECT_TRUE(MatrixEquals(d1.var(), variance_));
    EXPECT_TRUE(MatrixEquals(d1.ivar(), precision_));
    EXPECT_TRUE(MatrixEquals(d1.var_chol(), variance_cholesky_.getL()));
    EXPECT_TRUE(MatrixEquals(d1.ivar_chol(), precision_cholesky_.getL()));
    EXPECT_NEAR(d1.ldsi(), precision_.logdet(), 1e-8);
    
    SpdData d2(precision_, true);
    EXPECT_TRUE(MatrixEquals(d2.var(), variance_));
    EXPECT_TRUE(MatrixEquals(d2.ivar(), precision_));
    EXPECT_TRUE(MatrixEquals(d2.var_chol(), variance_cholesky_.getL()));
    EXPECT_TRUE(MatrixEquals(d2.ivar_chol(), precision_cholesky_.getL()));
    EXPECT_NEAR(d2.ldsi(), precision_.logdet(), 1e-8);
    
    SpdMatrix default_variance(4, 3.0);
    SpdData d3(4, 3.0, false);
    SpdMatrix default_precision(4, 1.0 / 3.0);
    Cholesky default_variance_cholesky(default_variance);
    Cholesky default_precision_cholesky(default_precision);
    EXPECT_TRUE(MatrixEquals(d3.var(), default_variance));
    EXPECT_TRUE(MatrixEquals(d3.ivar(), default_precision));
    EXPECT_TRUE(MatrixEquals(d3.var_chol(), default_variance_cholesky.getL()));
    EXPECT_TRUE(MatrixEquals(d3.ivar_chol(), default_precision_cholesky.getL()));
    EXPECT_NEAR(d3.ldsi(), default_precision.logdet(), 1e-8);
    
    SpdData d4(4, 1.0 / 3.0, true);
    EXPECT_TRUE(MatrixEquals(d4.var(), default_variance));
    EXPECT_TRUE(MatrixEquals(d4.ivar(), default_precision));
    EXPECT_TRUE(MatrixEquals(d4.var_chol(), default_variance_cholesky.getL()));
    EXPECT_TRUE(MatrixEquals(d4.ivar_chol(), default_precision_cholesky.getL()));
    EXPECT_NEAR(d4.ldsi(), default_precision.logdet(), 1e-8);    
  }

  TEST_F(SpdDataTest, Setters) {
    SpdData d1(dim_, 1.0);
    d1.set_var(variance_);
    EXPECT_TRUE(MatrixEquals(d1.var(), variance_));
    EXPECT_TRUE(MatrixEquals(d1.ivar(), precision_));
    EXPECT_TRUE(MatrixEquals(d1.var_chol(), variance_cholesky_.getL()));
    EXPECT_TRUE(MatrixEquals(d1.ivar_chol(), precision_cholesky_.getL()));
    EXPECT_NEAR(d1.ldsi(), precision_.logdet(), 1e-8);

    SpdData d2(dim_, 1.0);
    d2.set_ivar(precision_);
    EXPECT_TRUE(MatrixEquals(d2.var(), variance_));
    EXPECT_TRUE(MatrixEquals(d2.ivar(), precision_));
    EXPECT_TRUE(MatrixEquals(d2.var_chol(), variance_cholesky_.getL()));
    EXPECT_TRUE(MatrixEquals(d2.ivar_chol(), precision_cholesky_.getL()));
    EXPECT_NEAR(d2.ldsi(), precision_.logdet(), 1e-8);

    SpdData d3(dim_, 1.0);
    d3.set_var_chol(variance_cholesky_.getL());
    EXPECT_TRUE(MatrixEquals(d3.var(), variance_));
    EXPECT_TRUE(MatrixEquals(d3.ivar(), precision_));
    EXPECT_TRUE(MatrixEquals(d3.var_chol(), variance_cholesky_.getL()));
    EXPECT_TRUE(MatrixEquals(d3.ivar_chol(), precision_cholesky_.getL()));
    EXPECT_EQ(d3.ivar_chol().nrow(), dim_);
    EXPECT_EQ(d3.ivar_chol().ncol(), dim_);
    EXPECT_NEAR(d3.ldsi(), precision_.logdet(), 1e-8);

    SpdData d4(dim_, 1.0);
    d4.set_ivar_chol(precision_cholesky_.getL());
    EXPECT_TRUE(MatrixEquals(d4.var_chol(), variance_cholesky_.getL()));
    EXPECT_TRUE(MatrixEquals(d4.ivar_chol(), precision_cholesky_.getL()));
    EXPECT_TRUE(MatrixEquals(d4.var(), variance_));
    EXPECT_TRUE(MatrixEquals(d4.ivar(), precision_));
    EXPECT_NEAR(d4.ldsi(), precision_.logdet(), 1e-8);
  }

  TEST_F(SpdDataTest, MyBug) {
    SpdData d(4, 1.0);
    EXPECT_EQ(d.ivar_chol().nrow(), 4);
    EXPECT_EQ(d.ivar_chol().ncol(), 4);
  }
  
}  // namespace
