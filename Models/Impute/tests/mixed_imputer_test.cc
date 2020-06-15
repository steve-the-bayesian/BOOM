#include "gtest/gtest.h"
#include "Models/Impute/MvRegCopulaDataImputer.hpp"
#include "Models/Glm/PosteriorSamplers/MultivariateRegressionSampler.hpp"
#include "Models/MvnModel.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  //===========================================================================
  class MixedDataImputerTest : public ::testing::Test {
   protected:
    MixedDataImputerTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(MixedDataImputerTest, Empty) {
    // This test checks if the code can be built and linked.
  }


}  // namespace
