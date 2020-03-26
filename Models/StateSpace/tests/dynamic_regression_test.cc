#include "gtest/gtest.h"

#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/StateSpace/DynamicRegression.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"

#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include "Models/StateSpace/tests/state_space_test_utils.hpp"
#include "stats/AsciiDistributionCompare.hpp"

#include <fstream>

namespace {
  using namespace BOOM;
  using namespace BOOM::StateSpace;
  using std::endl;
  using std::cout;

  class RegressionDataTimePointTest : public ::testing::Test {
   protected:
    RegressionDataTimePointTest() {}
  };

  TEST_F(RegressionDataTimePointTest, blah) {
    NEW(RegressionDataTimePoint, dp)();
    EXPECT_EQ(0, dp->sample_size());
    EXPECT_FALSE(dp->using_suf());
  }

  class DynamicRegressionModelTest : public ::testing::Test {
   protected:

  };


}  // namespace
