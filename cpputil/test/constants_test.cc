#include "gtest/gtest.h"
#include "cpputil/Constants.hpp"
#include "cpputil/math_utils.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  double epsilon = 1e-10;
  
  TEST(Constants, correct_values) {
    EXPECT_NEAR(Constants::pi_squared,
                square(Constants::pi),
                epsilon);

    EXPECT_NEAR(Constants::half_pi_squared,
                .5 * Constants::pi_squared,
                epsilon);

    EXPECT_NEAR(3.0 * Constants::pi_squared_over_3,
                Constants::pi_squared,
                epsilon);

    EXPECT_NEAR(6.0 * Constants::pi_squared_over_6,
                Constants::pi_squared,
                epsilon);
    
    EXPECT_NEAR(square(Constants::root_2pi),
                2.0 * Constants::pi,
                epsilon);

    EXPECT_NEAR(exp(Constants::log_root_2pi),
                Constants::root_2pi,
                epsilon);

    EXPECT_NEAR(exp(Constants::log_pi),
                Constants::pi,
                epsilon);

    EXPECT_NEAR(square(Constants::root2),
                2.0,
                epsilon);
  }
  
}  // namespace
