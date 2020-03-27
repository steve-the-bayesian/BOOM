#include "gtest/gtest.h"

#include "distributions.hpp"
#include "LinAlg/Vector.hpp"
#include "stats/Bspline.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  class BsplineTest : public ::testing::Test {
   protected:
    BsplineTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(BsplineTest, BasicOperation) {
    Vector knots = {1, 2, 3};
    Bspline spline(knots);

    EXPECT_EQ(spline.basis_dimension(), 5);
    EXPECT_EQ(spline.order(), 4);

    Vector basis_2_1 = {0, 0.18224999999999994, 0.48599999999999999,
                        0.3307500000000001, 0.0010000000000000026};

    EXPECT_TRUE(VectorEquals(spline.basis(2.1),
                             basis_2_1))
        << "basis = " << spline.basis(2.1);

  }

}  // namespace
