#include "gtest/gtest.h"
#include "Models/Impute/MixedDataImputer.hpp"
#include "Models/Impute/MixedDataImputerWithErrorCorrection.hpp"
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
  class MixedDataImputerWithErrorCorrectionTest : public ::testing::Test {
   protected:
    MixedDataImputerWithErrorCorrectionTest() {
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

  TEST_F(MixedDataImputerWithErrorCorrectionTest, CompleteDataTest) {
    NEW(MixedImputation::CompleteData, complete_data)(data_);
    Vector yobs = complete_data->y_observed();
    EXPECT_TRUE(VectorEquals(numeric_, yobs));

    EXPECT_EQ(3, complete_data->y_true().size());
    EXPECT_EQ(3, complete_data->y_numeric().size());
    EXPECT_EQ(2, complete_data->true_categories().size());
    EXPECT_EQ(complete_data->observed_categories()[0]->value(), 1);
    EXPECT_EQ(complete_data->observed_categories()[1]->value(), 0);
  }

  TEST_F(MixedDataImputerWithErrorCorrectionTest, NumericErrorCorrectionModelTest) {
    NEW(NumericErrorCorrectionModel, model)(2, Vector{0.0, 999999});
  }

  TEST_F(MixedDataImputerWithErrorCorrectionTest, CategoricalErrorCorrectionModelTest) {
    NEW(CategoricalErrorCorrectionModel, model)(1, colors_);
  }

  TEST_F(MixedDataImputerWithErrorCorrectionTest, RowModelTest) {
    NEW(RowModel, model)();
  }

  TEST_F(MixedDataImputerWithErrorCorrectionTest, Empty) {
    // This test checks if the code can be built and linked.
  }

}  // namespace
