#include "gtest/gtest.h"
#include "distributions.hpp"

#include "Models/MvnGivenSigma.hpp"
#include "Models/WishartModel.hpp"
#include "Models/Mixtures/BetaBinomialMixture.hpp"
#include "Models/Mixtures/PosteriorSamplers/BetaBinomialMixturePosteriorSampler.hpp"
#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;

  class BetaBinomialMixtureTest : public ::testing::Test {
   protected:
    BetaBinomialMixtureTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(BetaBinomialMixtureTest, BasicModelFunctions) {

    NEW(MultinomialModel, mixing_distribution)(Vector{.3, .2, .5});

    std::vector<Ptr<BetaBinomialModel>> mixture_components;
    mixture_components.push_back(new BetaBinomialModel(1.0, 2.0));
    mixture_components.push_back(new BetaBinomialModel(2.0, 1.0));
    mixture_components.push_back(new BetaBinomialModel(15.0, 45.0));
    BetaBinomialMixtureModel model(mixture_components, mixing_distribution);

  }

}  // namespace
