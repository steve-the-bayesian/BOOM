#include "gtest/gtest.h"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateModels/LocalLinearTrend.hpp"
#include "Models/HMM/GeneralHmmStateSpaceWrapper.hpp"
#include "Models/HMM/PosteriorSamplers/LiuWestParticleFilter.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  class LiuWestParticleFilterTest : public ::testing::Test {
   protected:
    LiuWestParticleFilterTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(LiuWestParticleFilterTest, Basics) {
  }

  TEST_F(LiuWestParticleFilterTest, Filtering) {
    Vector true_state(1000);
    Vector observations(true_state.size());
    std::vector<Ptr<DoubleData>> data;
    for (int i = 0; i < true_state.size(); ++i) {
      true_state[i] = (i == 0) ? rnorm(0, 3) : rnorm(true_state[i-1], .1);
      observations[i] = true_state[i] + rnorm(0, .5);
      data.push_back(new DoubleData(observations[i]));
    }

    NEW(StateSpaceModel, base_model)();
    EXPECT_TRUE(base_model->observation_model() != nullptr);
    EXPECT_DOUBLE_EQ(base_model->observation_model()->sigsq(), 1.0);
    
    NEW(LocalLinearTrendStateModel, trend)();
    base_model->add_state(trend);

    cout << "building wrapper"  << endl;
    NEW(GeneralHmmStateSpaceWrapper, model)(base_model);
    int number_of_particles = 1000;
    cout << "building filter"  << endl;
    LiuWestParticleFilter filter(model, number_of_particles);
    for (int i = 0; i < data.size(); ++i) {
      cout << "about to update for data point " << i << endl;
      filter.update(GlobalRng::rng, *data[i], i);
    }
  }
  
}  // namespace
