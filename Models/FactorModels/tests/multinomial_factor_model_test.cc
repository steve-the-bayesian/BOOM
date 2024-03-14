#include "gtest/gtest.h"
#include "Models/FactorModels/MultinomialFactorModel.hpp"
#include "Models/FactorModels/PosteriorSamplers/MultinomialFactorModelPosteriorSampler.hpp"

#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>
#include <string>

namespace {
  using namespace BOOM;
  using Visitor = BOOM::FactorModels::MultinomialVisitor;
  using Site = BOOM::FactorModels::MultinomialSite;
  using std::endl;
  using std::cout;

  class MultinomialFactorModelTest : public ::testing::Test {
   protected:
    MultinomialFactorModelTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(MultinomialFactorModelTest, SmokeTest) {
    MultinomialFactorModel model(3);
  }
  
}  // namespace
