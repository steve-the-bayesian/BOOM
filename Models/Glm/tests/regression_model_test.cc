#include "gtest/gtest.h"

#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "distributions.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/PosteriorSamplers/AdaptiveSpikeSlabRegressionSampler.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class RegressionModelTest : public ::testing::Test {
   protected:
    RegressionModelTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(RegressionModelTest, DataConstructor) {
    int nobs = 1000;
    int nvars = 10;
    double residual_sd = 1.3;
    Matrix predictors(nobs, nvars);
    predictors.randomize();
    predictors.col(0) = 1.0;
    Vector coefficients(nvars);
    coefficients.randomize();
    for (int i = 6; i < nvars; ++i) {
      coefficients[i] = 0.0;
    }
    Vector response = predictors * coefficients;
    for (int i = 0; i < nobs; ++i) {
      response[i] += rnorm(0, residual_sd); 
    }
    NEW(RegressionModel, model)(predictors, response);
  }
  
}  // namespace
