#include "gtest/gtest.h"
#include "Models/SpdParams.hpp"
#include "distributions.hpp"

#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Cholesky.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class SpdParamsTest : public ::testing::Test {
   protected:
    SpdParamsTest()
        : dim_(2),
          variance_(dim_)
    {
      GlobalRng::rng.seed(8675309);
      variance_.randomize();
    }

    int dim_;
    SpdMatrix variance_;
  };

  TEST_F(SpdParamsTest, Vectorize) {
    SpdParams d1(variance_);
    EXPECT_EQ(dim_, d1.value().nrow());
    EXPECT_EQ(dim_, d1.value().ncol());
    Vector vec = d1.vectorize(false);
    EXPECT_EQ(vec.size(), dim_ * dim_);

    vec = d1.vectorize(true);
    EXPECT_EQ(vec.size(), dim_ * (dim_ + 1) / 2);
    EXPECT_DOUBLE_EQ(vec(0), variance_(0, 0));
    EXPECT_DOUBLE_EQ(vec(1), variance_(0, 1));
    EXPECT_DOUBLE_EQ(vec(2), variance_(1, 1));
  }

}  // namespace
