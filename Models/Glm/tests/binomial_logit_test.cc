#include "gtest/gtest.h"

#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "distributions.hpp"
#include "Models/Glm/BinomialLogitModel.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialLogitDataImputer.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class BinomialLogitTest : public ::testing::Test {
   protected:
    BinomialLogitTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  // BinomialLogitDataImputer triggered an ASAN error on CRAN during
  // construction.  This test builds the two types of imputers to try to
  // replicate the error, which I was unable to do.
  TEST_F(BinomialLogitTest, DataImputer) {
    BinomialLogitCltDataImputer clt_imputer;
    BinomialLogitPartialAugmentationDataImputer pa_imputer;
  }
  
}  // namespace
