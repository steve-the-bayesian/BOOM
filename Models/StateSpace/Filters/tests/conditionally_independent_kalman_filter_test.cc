#include "gtest/gtest.h"
#include "distributions.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"
#include "Models/StateSpace/MultivariateStateSpaceModelBase.hpp"

#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"

#include "LinAlg/DiagonalMatrix.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  using Marginal = Kalman::ConditionallyIndependentMarginalDistribution;

  
  class ConditionallyIndependentKalmanFilterTest : public ::testing::Test {
   protected:
    ConditionallyIndependentKalmanFilterTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  // It would be nice to have a mock version of the Model object expected by the
  // Marginal distribution.  Unfortunately, there is so much stuff to mock out
  // that it is easier to simply use the actual model.
  TEST_F(ConditionallyIndependentKalmanFilterTest, MarginalTest) {

    // model.simulate_data(5);

    // // Compute the update directly.
    // first1.set_high_dimensional_threshold_factor(infinity());
    // Marginal first1(&model, nullptr, 0);
    // double loglike1 = first1.update(model.observation(0),
    //                                 model.observed_status(0));


    // first2.set_high_dimensional_threshold_factor(0.0);
    // Marginal first2(&model, nullptr, 0);
    // double loglike2 = first2.update(
    
    // Marginal second1(&model, &first1, 1);
    // Marginal second2(&model, &first2, 1);
  }

}  // namespace
