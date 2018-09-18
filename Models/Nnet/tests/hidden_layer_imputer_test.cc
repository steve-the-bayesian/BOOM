#include "gtest/gtest.h"
#include "Models/Nnet/Nnet.hpp"
#include "Models/Nnet/PosteriorSamplers/HiddenLayerImputer.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class HiddenLayerImputerTest : public ::testing::Test {
   protected:
    HiddenLayerImputerTest()
        : layer0_(new HiddenLayer(5, 3)),
          layer1_(new HiddenLayer(3, 2)),
          layer2_(new HiddenLayer(2, 4)),
          imputer0_(layer0_, 0),
          imputer1_(layer1_, 1),
          imputer2_(layer2_, 2),
          predictors_{8, 6, 7, 5, 3},
          data_point_(new RegressionData(1.2, predictors_))
    {
      GlobalRng::rng.seed(8675309);
      outputs_.push_back(std::vector<bool>(3, false));
      outputs_.push_back(std::vector<bool>(2, true));
      outputs_.push_back(std::vector<bool>(4, false));
    }
    Ptr<HiddenLayer> layer0_;
    Ptr<HiddenLayer> layer1_;
    Ptr<HiddenLayer> layer2_;
    HiddenLayerImputer imputer0_;
    HiddenLayerImputer imputer1_;
    HiddenLayerImputer imputer2_;
    Nnet::HiddenNodeValues outputs_;
    Vector predictors_;
    Ptr<RegressionData> data_point_;
  };

  TEST_F(HiddenLayerImputerTest, ImputeInputsTest) {
  }

  TEST_F(HiddenLayerImputerTest, InputFullConditionalTest) {
  }
  
  TEST_F(HiddenLayerImputerTest, ClearLatentDataTest) {
  }

  TEST_F(HiddenLayerImputerTest, StoreInitialLayerLatentDataTest) {
  }
  
}  // namespace
