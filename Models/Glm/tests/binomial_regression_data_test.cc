#include "gtest/gtest.h"
#include "Models/Glm/BinomialRegressionData.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class BinomialRegressionDataTest : public ::testing::Test {
   protected:
   BinomialRegressionDataTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(BinomialRegressionDataTest, DataTest) {
    Vector x = {2, -3, 17};
    BinomialRegressionData d1(1, 2, x);
    EXPECT_TRUE(VectorEquals(x, d1.x()));
    EXPECT_DOUBLE_EQ(1.0, d1.y());
    EXPECT_DOUBLE_EQ(2.0, d1.n());

    NEW(VectorData, px)(x);
    BinomialRegressionData d2(3, 3, px);
    EXPECT_TRUE(VectorEquals(x, d2.x()));
    x[0] = 1.2;
    EXPECT_TRUE(VectorEquals(d1.x(), Vector{2, -3, 17}));
    EXPECT_TRUE(VectorEquals(d2.x(), Vector{2, -3, 17}));
    px->set_element(-.75, 0);
    EXPECT_TRUE(VectorEquals(d2.x(), Vector{-.75, -3, 17}));

    Ptr<BinomialRegressionData> d3 = d1.clone();
    EXPECT_TRUE(VectorEquals(d3->x(), Vector{2, -3, 17}));
    EXPECT_DOUBLE_EQ(1.0, d3->y());
    EXPECT_DOUBLE_EQ(2.0, d3->n());

    d1.set_n(10);
    EXPECT_DOUBLE_EQ(10, d1.n());
    d1.set_y(8);
    EXPECT_DOUBLE_EQ(8, d1.y());
    d1.increment(3, 5);
    EXPECT_DOUBLE_EQ(15, d1.n());
    EXPECT_DOUBLE_EQ(11, d1.y());
  }
  
}  // namespace
