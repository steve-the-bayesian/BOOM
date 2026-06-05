#include "gtest/gtest.h"

#include "Bandits/LinearBanditEncoder.hpp"

#include "distributions.hpp"
#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class ArmMapTest : public ::testing::Test {
   protected:
    ArmMapTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(ArmMapTest, CheckFactorsTest) {
    ExperimentStructure xp;
    xp.add_factor("Color", {"Red", "Blue", "Green"});
    xp.add_factor("Direction", {"Left", "Right"});

    ArmMap arm_map(xp);
    EXPECT_EQ(arm_map.number_of_arms(), 6);
    EXPECT_EQ(arm_map.number_of_factors(), 2);

    EXPECT_EQ(arm_map.factor_names()[0], "Color");
    EXPECT_EQ(arm_map.factor_names()[1], "Direction");
    
    EXPECT_EQ(0, arm_map.integer_factor_levels(0)[0]);
    EXPECT_EQ(0, arm_map.integer_factor_levels(0)[1]);

    // We iterate through the factors by moving the right-most element the fastest.
    EXPECT_EQ(0, arm_map.integer_factor_levels(1)[0]);
    EXPECT_EQ(1, arm_map.integer_factor_levels(1)[1]);

    EXPECT_EQ(1, arm_map.integer_factor_levels(2)[0]);
    EXPECT_EQ(0, arm_map.integer_factor_levels(2)[1]);
    
    EXPECT_EQ(1, arm_map.integer_factor_levels(3)[0]);
    EXPECT_EQ(1, arm_map.integer_factor_levels(3)[1]);

    EXPECT_EQ(2, arm_map.integer_factor_levels(4)[0]);
    EXPECT_EQ(0, arm_map.integer_factor_levels(4)[1]);
    EXPECT_EQ(2, arm_map.integer_factor_levels(5)[0]);
    EXPECT_EQ(1, arm_map.integer_factor_levels(5)[1]);

    // Factor level names are preceded by the factor names.
    EXPECT_EQ("Color:Red", arm_map.factor_level_names(1)[0]);
    EXPECT_EQ("Direction:Right", arm_map.factor_level_names(1)[1]);
  }

  class ExperimentArmEncoderTest : public ::testing::Test {
   protected:
    ExperimentArmEncoderTest() {
      GlobalRng::rng.seed(8675309);
      
    }

    DataTable create_fake_data_table(int nrow, int seed = 8675309) {
      GlobalRng::rng.seed(seed);
      
      Vector x1(nrow);
      x1.randomize();

      Vector x2(nrow);
      x2.randomize();

      Vector x3(nrow);
      x3.randomize();

      std::vector<std::string> colors = {"Red", "Blue", "Green"};
      std::vector<std::string> vcolors;
      for (int i = 0; i < nrow; ++i) {
        vcolors.push_back(colors[rmulti(0, 2)]);
      }

      DataTable ans;
      ans.append_variable(x1, "x1");
      ans.append_variable(x2, "x2");
      ans.append_variable(x3, "x3");
      ans.append_variable(
          CategoricalVariable(vcolors),
          "color");

      return ans;
    }
  };

  TEST_F(ExperimentArmEncoderTest, DimTest) {
    ExperimentStructure xp;
    xp.add_factor("ButtonPosition", {"left", "right"});
    xp.add_factor("ButtonColor", {"orange", "purple", "black"});

    NEW(ArmMap, arm_map)(xp);
    ExperimentArmEncoder enc("ButtonColor", arm_map, "black");

    EXPECT_EQ(2, enc.dim());
    EXPECT_TRUE(VectorEquals(enc.encode_level(0), Vector{1.0, 0.0}));
    EXPECT_TRUE(VectorEquals(enc.encode_level(1), Vector{0.0, 1.0}));
    EXPECT_TRUE(VectorEquals(enc.encode_level(2), Vector{-1.0, -1.0}));

    DataTable context = create_fake_data_table(100);
    enc.set_current_experiment_level(1);
    Matrix x = enc.encode_dataset(context);
    EXPECT_EQ(x.nrow(), 100);
    EXPECT_EQ(x.ncol(), 2);
    EXPECT_DOUBLE_EQ(x.col(0).sum(), 0.0);
    EXPECT_DOUBLE_EQ(x.col(1).sum(), 100.0);

    enc.set_current_experiment_level(0);
    x = enc.encode_dataset(context);
    EXPECT_DOUBLE_EQ(x.col(0).sum(), 100.0);
    EXPECT_DOUBLE_EQ(x.col(1).sum(), 0.0);
    
    enc.set_current_experiment_level(2);
    x = enc.encode_dataset(context);
    EXPECT_DOUBLE_EQ(x.col(0).sum(), -100.0);
    EXPECT_DOUBLE_EQ(x.col(1).sum(), -100.0);
  }
  
  class LinearBanditEncoderTest : public ::testing::Test {
   protected:
    LinearBanditEncoderTest() {
      GlobalRng::rng.seed(8675309);
      
      xp_.add_factor("ButtonPosition", {"Left", "Center", "Right"});
      xp_.add_factor("ButtonColor", {"Red", "Green", "Blue", "Orange"});

      arm_map_.reset(new ArmMap(xp_));
      button_position_encoder_.reset(new ExperimentArmEncoder(
          "ButtonPosition", arm_map_));
      button_color_encoder_.reset(new ExperimentArmEncoder(
          "ButtonColor", arm_map_));

      dataset_encoder_.reset(new DatasetEncoder);
      dataset_encoder_->add_encoder(button_position_encoder_);
      dataset_encoder_->add_encoder(button_color_encoder_);
    }

    ExperimentStructure xp_;
    Ptr<ArmMap> arm_map_;
    Ptr<ExperimentArmEncoder> button_position_encoder_;
    Ptr<ExperimentArmEncoder> button_color_encoder_;
    Ptr<DatasetEncoder> dataset_encoder_;
  };

  TEST_F(LinearBanditEncoderTest, ConstructionTest) {
    LinearBanditEncoder linear_encoder(arm_map_, dataset_encoder_);
  }

  TEST_F(LinearBanditEncoderTest, EncodingTest) {
    NEW(IdentityEncoder, x1_encoder)("x1");
    dataset_encoder_->add_encoder(x1_encoder);

    NEW(CatKey, stooge_key)({"Larry", "Moe", "Curly"});
    NEW(EffectsEncoder, stooge_encoder)("stooge", stooge_key);
    dataset_encoder_->add_encoder(stooge_encoder);
    
    NEW(InteractionEncoder, x1_pos)(x1_encoder, button_position_encoder_);
    dataset_encoder_->add_encoder(x1_pos);

    NEW(InteractionEncoder, x1_color)(x1_encoder, button_color_encoder_);
    dataset_encoder_->add_encoder(x1_color);

    LinearBanditEncoder encoder(arm_map_, dataset_encoder_);
    DataTable data;
    int sample_size = 10;
    Vector x1(sample_size);
    x1.randomize();
    data.append_variable(x1, "x1");

    std::vector<int> stooge_index = rmulti_vector_mt(
        GlobalRng::rng,
        sample_size,
        Vector{.3333, .3333, .3333});
    CategoricalVariable stooge_data(stooge_index, stooge_key);
    data.append_variable(stooge_data, "stooge");

    std::cout << "arm_map: \n" << *arm_map_ << std::endl;
    std::cout << "data = \n" << data;
    int which_arm = 1;

    Ptr<MixedMultivariateData> data_row(data.row(0));
    Vector encoded_row = encoder.encode_row(which_arm, *data_row);
    EXPECT_EQ(encoded_row.size(),
              1
              + button_position_encoder_->dim()
              + button_color_encoder_->dim()
              + x1_encoder->dim()
              + stooge_encoder->dim()
              + x1_pos->dim()
              + x1_color->dim());

    std::cout << encoded_row << std::endl;
    
    // Check the intercept term.
    EXPECT_DOUBLE_EQ(encoded_row[0], 1.0); 
    
    // Check that the results of each encoder appear in the correct positions.
    Vector button_encoding = button_position_encoder_->encode_row(*data_row);
    int pos = 1;
    for (int i = 0; i < button_encoding.size(); ++i, ++pos) {
      EXPECT_DOUBLE_EQ(encoded_row[pos], button_encoding[i]);
    }

    Vector button_color_encoding = button_color_encoder_->encode_row(*data_row);
    for (int i = 0; i < button_color_encoding.size(); ++i, ++pos) {
      EXPECT_DOUBLE_EQ(encoded_row[pos], button_color_encoding[i]);
    }

    EXPECT_DOUBLE_EQ(encoded_row[pos++], x1[0]);
    Vector stooge_encoding = stooge_encoder->encode_row(*data_row);
    for (int i = 0; i < stooge_encoding.size(); ++i, ++pos) {
      EXPECT_DOUBLE_EQ(encoded_row[pos], stooge_encoding[i]);
    }

    Vector x1_pos_encoding = x1_pos->encode_row(*data_row);
    for (int i = 0; i < x1_pos_encoding.size(); ++i, ++pos) {
      EXPECT_DOUBLE_EQ(encoded_row[pos], x1_pos_encoding[i]);
    }
    
    Vector x1_color_encoding = x1_color->encode_row(*data_row);
    for (int i = 0; i < x1_color_encoding.size(); ++i, ++pos) {
      EXPECT_DOUBLE_EQ(encoded_row[pos], x1_color_encoding[i]);
    }
  }

  
}  // namespace
