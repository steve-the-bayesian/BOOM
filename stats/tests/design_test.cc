#include "gtest/gtest.h"

#include "stats/Design.hpp"

#include "test_utils/test_utils.hpp"
#include <string>
#include <vector>

namespace {
  using namespace BOOM;

  class ExperimentStructureTest : public ::testing::Test {
   protected:
    ExperimentStructureTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(ExperimentStructureTest, StringConstructor) {
    std::vector<std::string> factor_names;
    std::vector<std::vector<std::string>> factor_levels;

    ExperimentStructure xp;
    xp.add_factor(
        "color",
        {"Red", "Green", "Blue"});

    EXPECT_EQ(1, xp.nfactors());
    EXPECT_EQ(3, xp.nlevels(0));

    EXPECT_EQ(3, xp.nlevels()[0]);
    EXPECT_EQ(1, xp.nlevels().size());

    EXPECT_EQ("Red", xp.level_name(0, 0));
    EXPECT_EQ("color:Red", xp.full_level_name(0, 0, ":"));
  }

}  // namespace
