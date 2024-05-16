#include "gtest/gtest.h"

#include "Models/Glm/ChoiceData.hpp"

#include "distributions.hpp"
#include "stats/moments.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class ChoiceDataTest : public ::testing::Test {
   protected:
    ChoiceDataTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  //===========================================================================
  TEST_F(ChoiceDataTest, FullFreight) {
    int subject_xdim = 5;
    int choice_xdim = 4;
    int num_choices = 3;

    NEW(CategoricalData, response)(1, num_choices);
    NEW(VectorData, subject_predictors)(rnorm_vector(
        subject_xdim, 0, 1));
    subject_predictors->set_element(1.0, 0);

    std::vector<Ptr<VectorData>> choice_predictors;
    for (int m = 0; m < num_choices; ++m) {
      choice_predictors.push_back(
          new VectorData(rnorm_vector(choice_xdim, 0, 1)));
    }

    ChoiceData data_point(*response,
                          subject_predictors,
                          choice_predictors);

    EXPECT_TRUE(VectorEquals(
        subject_predictors->value(),
        data_point.Xsubject()));
    EXPECT_TRUE(VectorEquals(
        choice_predictors[0]->value(),
        data_point.Xchoice(0)));
    EXPECT_TRUE(VectorEquals(
        choice_predictors[1]->value(),
        data_point.Xchoice(1)));

    Matrix X_with_zeros = data_point.X();
    EXPECT_EQ(X_with_zeros.nrow(), num_choices);
    EXPECT_EQ(X_with_zeros.ncol(),
              num_choices * subject_xdim + choice_xdim);

    Matrix X_no_zeros = data_point.X(false);
    EXPECT_EQ(X_no_zeros.nrow(), num_choices);
    EXPECT_EQ(X_no_zeros.ncol(),
              choice_xdim + (num_choices - 1) * subject_xdim);

  }

  //===========================================================================
  TEST_F(ChoiceDataTest, TestPredictorMapWithExplicitZerosForChoiceZero) {
    int subject_xdim = 5;
    int choice_xdim = 4;
    int num_choices = 3;

    ChoiceDataPredictorMap pred_map(
        subject_xdim, choice_xdim, num_choices, true);

    // The first subject_xdim elements should be mapped to choice level 0.
    for (int m = 0; m < subject_xdim; ++m) {
      std::pair<int, int> indices = pred_map.subject_index(m);
      int predictor_index = indices.first;
      int choice_level = indices.second;
      EXPECT_EQ(choice_level, 0)
          << "Error with m = " << m << " of " << subject_xdim << ".\n";
      EXPECT_EQ(predictor_index, m);
    }

    // The next subject_xdim elements should be mapped to choice level 1.
    for (int m = subject_xdim; m < 2 * subject_xdim; ++m) {
      std::pair<int, int> indices = pred_map.subject_index(m);
      int predictor_index = indices.first;
      int choice_level = indices.second;
      EXPECT_EQ(choice_level, 1)
          << "Error with m = " << m << " of " << subject_xdim << ".\n";
      EXPECT_EQ(predictor_index, m - subject_xdim);
    }

    // The choice variables should start at subject_xdim + num_choices, so
    // everything before that point should show up as 'false' in the is_choice()
    // method.
    int choice_start = subject_xdim * num_choices;
    for (int m = 0; m < choice_start; ++m) {
      EXPECT_FALSE(pred_map.is_choice(m));
    }

    // The final 'choice_dim' predictors should show up as 'true' in the
    // is_choice() method, and their indices should start at 0 and count
    // to the end.
    for (int m = choice_start; m < choice_start + choice_xdim; ++m) {
      EXPECT_TRUE(pred_map.is_choice(m));
      int choice_index = pred_map.choice_index(m);
      EXPECT_EQ(choice_index, m - choice_start);
      EXPECT_EQ(m, pred_map.long_choice_index(choice_index));
    }
  }

  //===========================================================================
  TEST_F(ChoiceDataTest, TestPredictorMapWithImplicitZerosForChoiceZero) {
    int subject_xdim = 5;
    int choice_xdim = 4;
    int num_choices = 3;

    ChoiceDataPredictorMap pred_map(
        subject_xdim, choice_xdim, num_choices,
        false); // include_zeros set to false.

    // The first subject_xdim elements should be mapped to choice level 1.
    for (int m = 0; m < subject_xdim; ++m) {
      std::pair<int, int> indices = pred_map.subject_index(m);
      int predictor_index = indices.first;
      int choice_level = indices.second;
      EXPECT_EQ(choice_level, 1)
          << "Error with m = " << m << " of " << subject_xdim << ".\n";
      EXPECT_EQ(predictor_index, m);

      EXPECT_EQ(pred_map.long_subject_index(
          predictor_index, choice_level),
                m);
    }

    // The next subject_xdim elements should be mapped to choice level 2.
    for (int m = subject_xdim; m < 2 * subject_xdim; ++m) {
      std::pair<int, int> indices = pred_map.subject_index(m);
      int predictor_index = indices.first;
      int choice_level = indices.second;
      EXPECT_EQ(choice_level, 2)
          << "Error with m = " << m << " of " << subject_xdim << ".\n";
      EXPECT_EQ(predictor_index, m - subject_xdim);
      EXPECT_EQ(pred_map.long_subject_index(
          predictor_index, choice_level),
                m);
    }

    // The choice variables should start at subject_xdim + (num_choices - 1), so
    // everything before that point should show up as 'false' in the is_choice()
    // method.
    int choice_start = subject_xdim * (num_choices - 1);
    for (int m = 0; m < choice_start; ++m) {
      EXPECT_FALSE(pred_map.is_choice(m));
    }

    // The final 'choice_dim' predictors should show up as 'true' in the
    // is_choice() method, and their indices should start at 0 and count
    // to the end.
    for (int m = choice_start; m < choice_start + choice_xdim; ++m) {
      EXPECT_TRUE(pred_map.is_choice(m));
      int choice_index = pred_map.choice_index(m);
      EXPECT_EQ(choice_index, m - choice_start);
      EXPECT_EQ(m, pred_map.long_choice_index(choice_index));
    }
  }


}  // namespace
