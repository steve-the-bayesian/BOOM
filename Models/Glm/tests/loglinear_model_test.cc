#include "gtest/gtest.h"

#include "Models/Glm/Encoders.hpp"
#include "Models/Glm/LoglinearModel.hpp"
#include "LinAlg/Selector.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class CategoricalEncoderTest : public ::testing::Test {
   protected:
    CategoricalEncoderTest() {
      GlobalRng::rng.seed(8675309);
      colors_.reset(new CatKey({"red", "blue", "green"}));
      sizes_.reset(new CatKey({"xs", "small", "med", "large"}));
      shapes_.reset(new CatKey({"circle", "square"}));

      red_.reset(new CategoricalData("red", colors_));
      medium_.reset(new CategoricalData("med", sizes_));
      circle_.reset(new CategoricalData("circle", shapes_));
    }
    Ptr<CatKey> colors_;
    Ptr<CatKey> sizes_;
    Ptr<CatKey> shapes_;
    Ptr<CategoricalData> red_;
    Ptr<CategoricalData> medium_;
    Ptr<CategoricalData> circle_;
  };

  // Test MultivariateCategoricalData for correctness.
  TEST_F(CategoricalEncoderTest, DataTest) {
    MultivariateCategoricalData dp;
    dp.push_back(red_);
    dp.push_back(medium_);
    dp.push_back(circle_);
    EXPECT_EQ(dp.nvars(), 3);
  }

  TEST_F(CategoricalEncoderTest, EncodingTest) {
    NEW(CategoricalMainEffect, color_encoder)(0, colors_);
    NEW(CategoricalMainEffect, size_encoder)(1, sizes_);
    NEW(CategoricalMainEffect, shape_encoder)(2, shapes_);

    // CategoricalDatasetEncoder encoder;
    // encoder.add_main_effect(color_encoder);
    // encoder.add_main_effect(size_encoder);
    // encoder.add_main_effect(shape_encoder);

  }

}  // namespace
