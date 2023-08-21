#include "gtest/gtest.h"
#include "Models/PositiveSemidefiniteData.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class PositiveSemidefiniteDataTest : public ::testing::Test {
   protected:
    PositiveSemidefiniteDataTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(PositiveSemidefiniteDataTest, RootTest) {
    SpdMatrix S(3);
    S.randomize();

    PositiveSemidefiniteData data(S);
    Matrix R = data.root();
    Matrix S2 = R * R.transpose();
    EXPECT_TRUE(MatrixEquals(S, S2))
        << "S = \n" << S
        << "S2 = \n" << S2;


    Matrix A(5, 3);
    SpdMatrix V = A * S * A.transpose();

    data.set(V);
    SpdMatrix V2 = data.root() * data.root().transpose();
    EXPECT_TRUE(MatrixEquals(V, V2));
  }

  TEST_F(PositiveSemidefiniteDataTest, GenInverseTest) {
    SpdMatrix S(3);
    S.randomize();

    PositiveSemidefiniteData data(S);
    Matrix R = data.root();
    Matrix S2 = R * R.transpose();
    EXPECT_TRUE(MatrixEquals(S, S2))
        << "S = \n" << S
        << "S2 = \n" << S2;
  }

}  // namespace
