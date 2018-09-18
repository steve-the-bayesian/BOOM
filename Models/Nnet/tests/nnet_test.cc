#include "gtest/gtest.h"
#include "Models/Nnet/Nnet.hpp"
#include "Models/Nnet/GaussianFeedForwardNeuralNetwork.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class NnetTest : public ::testing::Test {
   protected:
    NnetTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(NnetTest, HiddenLayerTest) {
    HiddenLayer layer(3, 4);
  }

  TEST_F(NnetTest, GaussianFeedForwardNeuralNetworkTest) {
    GaussianFeedForwardNeuralNetwork model;
  }
  
}  // namespace
