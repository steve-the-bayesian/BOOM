#include "gtest/gtest.h"

#include "Bandits/LinearBanditEncoder.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class ArmMapTest : public ::testing::Test {
   protected:
    ArmMapTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(ArmMapTest, CheckFactorsTest) {
    ExperimentStructure xp;
    xp.add_factor("Color", {"Red", "Blue", "Green"});
    xp.add_factor("Direction", {"Left", "Right"});

    ArmMap arm_map(xp);
    EXPECT_EQ(arm_map.number_of_arms(), 6);
    EXPECT_EQ(arm_map.number_of_factors(), 2);

    EXPECT_EQ(arm_map.factor_names()[0], "Color");
    EXPECT_EQ(arm_map.factor_names()[1], "Direction");
    
    EXPECT_EQ(0, arm_map.integer_factor_levels(0)[0]);
    EXPECT_EQ(0, arm_map.integer_factor_levels(0)[1]);

    // We iterate through the factors by moving the right-most element the fastest.
    EXPECT_EQ(0, arm_map.integer_factor_levels(1)[0]);
    EXPECT_EQ(1, arm_map.integer_factor_levels(1)[1]);

    EXPECT_EQ(1, arm_map.integer_factor_levels(2)[0]);
    EXPECT_EQ(0, arm_map.integer_factor_levels(2)[1]);
    
    EXPECT_EQ(1, arm_map.integer_factor_levels(3)[0]);
    EXPECT_EQ(1, arm_map.integer_factor_levels(3)[1]);

    EXPECT_EQ(2, arm_map.integer_factor_levels(4)[0]);
    EXPECT_EQ(0, arm_map.integer_factor_levels(4)[1]);
    
    EXPECT_EQ(2, arm_map.integer_factor_levels(5)[0]);
    EXPECT_EQ(1, arm_map.integer_factor_levels(5)[1]);

    // Factor level names are preceded by the factor names.
    EXPECT_EQ("Color:Red", arm_map.factor_level_names(1)[0]);
    EXPECT_EQ("Direction:Right", arm_map.factor_level_names(1)[1]);
  }

}  // namespace
