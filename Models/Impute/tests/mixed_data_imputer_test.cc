#include "gtest/gtest.h"
#include "Models/Impute/MixedDataImputer.hpp"
#include "Models/Glm/PosteriorSamplers/MultivariateRegressionSampler.hpp"
#include "Models/MvnModel.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using namespace BOOM::MixedImputation;
  using std::endl;
  using std::cout;


  //===========================================================================
  class MixedDataImputerTest : public ::testing::Test {
   protected:
    MixedDataImputerTest() {
      GlobalRng::rng.seed(8675309);
      colors_.reset(new CatKey(
          std::vector<std::string>{"red", "blue", "green"}));
      shapes_.reset(new CatKey(
          std::vector<std::string>{"circle", "square", "octogon"}));

      numeric_ = Vector(3);
      numeric_.randomize();

      data_.reset(new MixedMultivariateData);
      data_->add_numeric(new DoubleData(numeric_[0]));
      data_->add_categorical(new LabeledCategoricalData("blue", colors_));
      data_->add_numeric(new DoubleData(numeric_[1]));
      data_->add_categorical(new LabeledCategoricalData("circle", shapes_));
      data_->add_numeric(new DoubleData(numeric_[2]));
    }
    Ptr<CatKey> colors_;
    Ptr<CatKey> shapes_;
    Vector numeric_;
    Ptr<MixedMultivariateData> data_;
  };

  TEST_F(MixedDataImputerTest, RowModelTest) {
    NEW(RowModel, model)();
  }

  TEST_F(MixedDataImputerTest, Empty) {
    // This test checks if the code can be built and linked.
  }

  TEST_F(MixedDataImputerTest, ImputesNumericData) {

  }

  TEST_F(MixedDataImputerTest, ImputesCategoricalData) {

  }

}  // namespace
