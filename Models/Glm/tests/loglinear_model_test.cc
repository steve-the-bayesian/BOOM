#include "gtest/gtest.h"

#include "stats/Encoders.hpp"
#include "Models/Glm/LoglinearModel.hpp"
#include "Models/Glm/PosteriorSamplers/LoglinearModelBipfSampler.hpp"
#include "LinAlg/Selector.hpp"
#include "distributions.hpp"
#include "stats/DataTable.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace BOOM {
  // Import from test library.
  DataTable minn38_data();
}

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  std::vector<Ptr<MultivariateCategoricalData>> get_minn38_data() {
    DataTable table = minn38_data();
    Vector frequency = table.getvar(4);
    CategoricalVariable hs = table.get_nominal(0);
    CategoricalVariable phs = table.get_nominal(1);
    CategoricalVariable fol = table.get_nominal(2);
    CategoricalVariable sex = table.get_nominal(3);

    std::vector<Ptr<MultivariateCategoricalData>> ans;
    ans.reserve(table.nrow());
    for (int i = 0; i < table.nrow(); ++i) {
      NEW(MultivariateCategoricalData, data_point)(
        {hs[i], phs[i], fol[i], sex[i]}, frequency[i]);
      ans.push_back(data_point);
    }
    return ans;
  }

  class MultivariateCategoricalDataTest : public ::testing::Test {
   protected:
    MultivariateCategoricalDataTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(MultivariateCategoricalDataTest, TestData) {
    std::vector<Ptr<MultivariateCategoricalData>> dataset = get_minn38_data();
    Ptr<MultivariateCategoricalData> data = dataset[0];
    EXPECT_TRUE(!!data);
    EXPECT_EQ(data->nvars(), 4);
    EXPECT_DOUBLE_EQ(data->frequency(), 87.0);
    std::ostringstream output;
    output << *data;
    EXPECT_EQ(output.str(), "L C F1 M 87");

    MultivariateCategoricalData d2;
    EXPECT_DOUBLE_EQ(1.0, d2.frequency());
    EXPECT_EQ(d2.nvars(), 0);
    d2.push_back(data->mutable_element(0));
    EXPECT_EQ(d2.nvars(), 1);
    d2.push_back(data->mutable_element(1));
    EXPECT_EQ(d2.nvars(), 2);
  }

  //===========================================================================
  class CategoricalEncoderTest : public ::testing::Test {
   protected:
    CategoricalEncoderTest()
        : table_(minn38_data())
    {
      GlobalRng::rng.seed(8675309);
    }
    DataTable table_;
  };

  TEST_F(CategoricalEncoderTest, TestMainEffect) {
    // The hs variable has 3 levels: L, M, U.
    CategoricalMainEffect hs(0, table_.get_nominal(0, 0)->key());

    EXPECT_EQ(2, hs.dim());
    EXPECT_EQ(hs.nlevels(), std::vector<int>(1, 3));
    EXPECT_EQ(hs.which_variables(), std::vector<int>(1, 0));

    NEW(MultivariateCategoricalData, observation)(table_, 0);
    Vector enc = hs.encode(*observation);
    EXPECT_EQ(observation->mutable_element(0)->value(), 0);
    EXPECT_TRUE(VectorEquals(enc, Vector{1.0, 0}));
    enc = hs.encode(std::vector<int>{0, 19, 23, 84});
    EXPECT_TRUE(VectorEquals(enc, Vector{1.0, 0}));
    enc = hs.encode(std::vector<int>{1, 19, 23, 84, 95, 42, 37});
    EXPECT_TRUE(VectorEquals(enc, Vector{0.0, 1.0}));
    enc = hs.encode(std::vector<int>{2, 19, 23, 84});
    EXPECT_TRUE(VectorEquals(enc, Vector{-1, -1}));


    // phs has 4 levels.
    CategoricalMainEffect phs(1, table_.get_nominal(0, 1)->key());
    EXPECT_EQ(3, phs.dim());
    EXPECT_EQ(phs.nlevels(), std::vector<int>(1, 4));
    EXPECT_EQ(phs.which_variables(), std::vector<int>(1, 1));

    enc = phs.encode(std::vector<int>{19, 0, 43, 12});
    EXPECT_TRUE(VectorEquals(enc,
                             Vector{1, 0, 0}));
    EXPECT_TRUE(VectorEquals(phs.encode(std::vector<int>{19, 1, 43, 12}),
                             Vector{0, 1, 0}));
    EXPECT_TRUE(VectorEquals(phs.encode(std::vector<int>{19, 2, 43, 12}),
                             Vector{0, 0, 1}));
    EXPECT_TRUE(VectorEquals(phs.encode(std::vector<int>{19, 3, 43, 12}),
                             Vector{-1, -1, -1}));
  }

  TEST_F(CategoricalEncoderTest, TestInteraction) {
    NEW(CategoricalMainEffect, hs)(0, table_.get_nominal(0, 0)->key());
    NEW(CategoricalMainEffect, phs)(1, table_.get_nominal(0, 1)->key());
    NEW(CategoricalInteraction, interaction)(hs, phs);

    EXPECT_EQ(6, interaction->dim());
    EXPECT_EQ(interaction->nlevels(), (std::vector<int>{3, 4}));
    EXPECT_EQ(interaction->which_variables(), (std::vector<int>{0, 1}));

    // Check that the index of the first variable is the one changing most rapidly.
    // The mapping looks like this:
    //      0  1  2
    //    _____________________________
    //  0 | 0  2  4
    //  1 | 1  3  5
    //
    EXPECT_TRUE(VectorEquals(interaction->encode(std::vector<int>{0, 0, 0, 0}),
                             Vector{1, 0, 0, 0, 0, 0}));
    EXPECT_TRUE(VectorEquals(interaction->encode(std::vector<int>{1, 0, 0, 0}),
                             Vector{0, 1, 0, 0, 0, 0}));
    EXPECT_TRUE(VectorEquals(interaction->encode(std::vector<int>{2, 0, 0, 0}),
                             Vector{-1, -1, 0, 0, 0, 0}));
    EXPECT_TRUE(VectorEquals(interaction->encode(std::vector<int>{0, 1, 0, 0}),
                             Vector{0, 0, 1, 0, 0, 0}));
    EXPECT_TRUE(VectorEquals(interaction->encode(std::vector<int>{1, 1, 0, 0}),
                             Vector{0, 0, 0, 1, 0, 0}));
    EXPECT_TRUE(VectorEquals(interaction->encode(std::vector<int>{2, 1, 0, 0}),
                             Vector{0, 0, -1, -1, 0, 0}));
    EXPECT_TRUE(VectorEquals(interaction->encode(std::vector<int>{0, 2, 0, 0}),
                             Vector{0, 0, 0, 0, 1, 0}));
    EXPECT_TRUE(VectorEquals(interaction->encode(std::vector<int>{1, 2, 0, 0}),
                             Vector{0, 0, 0, 0, 0, 1}));
    EXPECT_TRUE(VectorEquals(interaction->encode(std::vector<int>{2, 2, 0, 0}),
                             Vector{0, 0, 0, 0, -1, -1}));
    EXPECT_TRUE(VectorEquals(interaction->encode(std::vector<int>{0, 3, 0, 0}),
                             Vector{-1, 0, -1, 0, -1, 0}));
    EXPECT_TRUE(VectorEquals(interaction->encode(std::vector<int>{1, 3, 0, 0}),
                             Vector{0, -1, 0, -1, 0, -1}));
    EXPECT_TRUE(VectorEquals(interaction->encode(std::vector<int>{2, 3, 0, 0}),
                             Vector{1, 1, 1, 1, 1, 1}));
  }

  //===========================================================================
  class LoglinearModelTest : public ::testing::Test {
   protected:
    LoglinearModelTest() {
      data_ = get_minn38_data();
      hs_.reset(new CategoricalMainEffect(0, data_[0]->mutable_element(0)->key()));
      phs_.reset(new CategoricalMainEffect(1, data_[0]->mutable_element(1)->key()));
      fol_.reset(new CategoricalMainEffect(2, data_[0]->mutable_element(2)->key()));
      sex_.reset(new CategoricalMainEffect(3, data_[0]->mutable_element(3)->key()));
    }
    std::vector<Ptr<MultivariateCategoricalData>> data_;
    Ptr<CategoricalMainEffect> hs_;
    Ptr<CategoricalMainEffect> phs_;
    Ptr<CategoricalMainEffect> fol_;
    Ptr<CategoricalMainEffect> sex_;
  };

  TEST_F(LoglinearModelTest, construction) {
    LoglinearModel model;
    data_ = get_minn38_data();
    for (int i = 0; i < data_.size(); ++i) {
      model.add_data(data_[i]);
    }
    model.add_interaction({0, 1});
    model.add_interaction({0, 2});
    model.add_interaction({0, 3});
    model.add_interaction({1, 2});
    model.add_interaction({1, 3});
    model.add_interaction({2, 3});
    model.refresh_suf();
  }

  // NOTE: Because the sufficient statistics are stored in actual arrays, we do
  // not need to check the order of their unwinding.
  TEST_F(LoglinearModelTest, TestSufficientStats) {
    LoglinearModelSuf suf;
    suf.add_effect(hs_);
    suf.add_effect(phs_);
    suf.add_effect(fol_);
    suf.add_effect(sex_);
    suf.add_effect(new CategoricalInteraction(hs_, phs_));
    suf.add_effect(new CategoricalInteraction(phs_, fol_));

    // data_[0] is L, C, F1, M with frequency 87
    suf.update(data_[0]);
    double freq = data_[0]->frequency();
    EXPECT_DOUBLE_EQ(suf.sample_size(), freq);

    // The hs main effect.
    Array arr = suf.margin(std::vector<int>(1, 0));
    EXPECT_EQ(3, arr.size());
    EXPECT_DOUBLE_EQ(arr(0), freq);
    EXPECT_DOUBLE_EQ(arr(1), 0);
    EXPECT_DOUBLE_EQ(arr(2), 0);

    // Find the hs by phs interaction.
    arr = suf.margin(std::vector<int>{0, 1});
    EXPECT_EQ(2, arr.ndim());
    EXPECT_EQ(3, arr.dim(0));
    EXPECT_EQ(4, arr.dim(1));
    EXPECT_DOUBLE_EQ(arr(0, 0), freq);

    // Now update with an observation that has a different phs.
    suf.update(data_[7]);
    // "L","N","F1","M",3
    // Add 3 more to variable 0.
    arr = suf.margin(std::vector<int>(1, 0));
    EXPECT_DOUBLE_EQ(arr(0), 87 + 3);

    int val = (*data_[7])[1].value();
    arr = suf.margin(std::vector<int>(1, 1));
    EXPECT_DOUBLE_EQ(arr(0), 87);
    EXPECT_DOUBLE_EQ(arr(val), 3);

    arr = suf.margin(std::vector<int>{0, 1});
    EXPECT_DOUBLE_EQ(arr(0, 0), 87);
    EXPECT_DOUBLE_EQ(arr(0, val), 3);

    suf.clear();
    EXPECT_EQ(0, suf.sample_size());
    arr = suf.margin(std::vector<int>(1, 0));
    EXPECT_EQ(1, arr.ndim());
    EXPECT_EQ(3, arr.dim(0));
    EXPECT_EQ(0, arr(0));
    EXPECT_EQ(0, arr(1));
    EXPECT_EQ(0, arr(2));
  }

  TEST_F(LoglinearModelTest, TestSingleVar) {
    NEW(LoglinearModel, model)();
    data_ = get_minn38_data();
    for (int i = 0; i < data_.size(); ++i) {
      NEW(MultivariateCategoricalData, data_point)({}, data_[i]->frequency());
      data_point->push_back(data_[i]->mutable_element(0));
      //      model->add_data(data_[i]);
      model->add_data(data_point);
    }
    model->refresh_suf();

    // NEW(LoglinearModelBipfSampler, sampler)(model.get());
    // model->set_method(sampler);
    // int niter = 1000;

    // Matrix draws(niter, model->coef().nvars_possible());
    // for (int i = 0; i < niter; ++i) {
    //   model->sample_posterior();
    //   draws.row(i) = model->coef().Beta();
    // }

    // std::ofstream out("loglin_single.out");
    // out << draws;
  }

  TEST_F(LoglinearModelTest, TestMcmc) {
    // NEW(LoglinearModel, model)();
    // data_ = get_minn38_data();
    // for (int i = 0; i < data_.size(); ++i) {
    //   model->add_data(data_[i]);
    // }
    // model->add_interaction({0, 1});
    // model->add_interaction({0, 2});
    // model->add_interaction({0, 3});
    // model->add_interaction({1, 2});
    // model->add_interaction({1, 3});
    // model->add_interaction({2, 3});
    // model->refresh_suf();

    // NEW(LoglinearModelBipfSampler, sampler)(model.get());
    // model->set_method(sampler);
    // int niter = 1000;

    // Matrix draws(niter, model->coef().nvars_possible());
    // for (int i = 0; i < niter; ++i) {
    //   model->sample_posterior();
    //   draws.row(i) = model->coef().Beta();
    // }

    // std::ofstream out("loglin.out");
    // out << draws;
  }

}  // namespace
