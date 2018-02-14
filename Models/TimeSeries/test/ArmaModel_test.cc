#include "gtest/gtest.h"
#include <Models/TimeSeries/ArmaModel.hpp>

namespace {
  using namespace BOOM;
  
  class ArmaModelTest : public ::testing::Test {
   protected:
  };

  TEST_F(ArmaModelTest, Constructor) {
    ArmaModel model(3, 2);
    EXPECT_THROW(ArmaModel(0, 0), std::runtime_error);
  }
  
}  // namespace

