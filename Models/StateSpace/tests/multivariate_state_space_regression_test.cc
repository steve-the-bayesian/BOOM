#include "gtest/gtest.h"

#include "test_utils/test_utils.hpp"

#include "cpputil/math_utils.hpp"

#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/IndependentMvnVarSampler.hpp"
#include "Models/StateSpace/MultivariateStateSpaceRegressionModel.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/PosteriorSamplers/SharedLocalLevelPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/MultivariateStateSpaceModelSampler.hpp"
#include "distributions.hpp"
#include "LinAlg/Array.hpp"

namespace {

  using namespace BOOM;
  using std::endl;
  using std::cout;

  class MultivariateStateSpaceRegressionModelTest : public ::testing::Test {
   protected:
    MultivariateStateSpaceRegressionModelTest()
    {
      GlobalRng::rng.seed(8675310);
    }
  };

  //===========================================================================
  TEST_F(MultivariateStateSpaceRegressionModelTest, EmptyTest) {}

  //===========================================================================
  TEST_F(MultivariateStateSpaceRegressionModelTest, ConstructorTest) {
    MultivariateStateSpaceRegressionModel model(3, 4);
  }
  
}  // namespace
