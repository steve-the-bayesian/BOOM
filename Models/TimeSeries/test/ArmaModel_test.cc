#include "gtest/gtest.h"
#include "Models/TimeSeries/ArmaModel.hpp"

namespace {
  using namespace BOOM;
  
  class ArmaModelTest : public ::testing::Test {
   protected:
  };

  TEST_F(ArmaModelTest, Constructor) {
    ArmaModel model(3, 2);
    EXPECT_THROW(ArmaModel(0, 0), std::runtime_error);
  }

  TEST_F(ArmaModelTest, Stationary) {
    ArmaModel model(new GlmCoefs(Vector{.80, 067}),
                    new VectorParams(Vector{.53, .09}),
                    new UnivParams(18));
    EXPECT_TRUE(model.is_causal());
    EXPECT_TRUE(model.is_invertible());
    EXPECT_TRUE(model.is_stationary());
  }
  
}  // namespace

