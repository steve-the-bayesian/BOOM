#include "gtest/gtest.h"
#include "Models/Nnet/Nnet.hpp"
#include "Models/Nnet/PosteriorSamplers/HiddenLayerImputer.hpp"

#include "distributions.hpp"
#include "stats/moments.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace HiddenLayerImputerTestNamespace {
  using namespace BOOM;
  using std::endl;

  class HiddenLayerImputerTest : public ::testing::Test {
   protected:
    HiddenLayerImputerTest()
        : layer0_(new HiddenLayer(3, 12)),
          layer1_(new HiddenLayer(12, 2)),
          imputer0_(layer0_, 0),
          imputer1_(layer1_, 1)
    {
      GlobalRng::rng.seed(8675309);
      Randomize(layer0_);
      Randomize(layer1_);
    }

    void Randomize(Ptr<HiddenLayer> layer) {
      for (int i = 0; i < layer->number_of_nodes(); ++i) {
        Ptr<BinomialLogitModel> logit = layer->logistic_regression(i);
        Vector b = logit->Beta();
        for (int j = 0; j < b.size(); ++j) {
          b[j] = rnorm(0, 1);
        }
        logit->set_Beta(b);
      }
    }
    
    Ptr<HiddenLayer> layer0_;
    Ptr<HiddenLayer> layer1_;
    HiddenLayerImputer imputer0_;
    HiddenLayerImputer imputer1_;
  };

  TEST_F(HiddenLayerImputerTest, InputFullConditionalTest) {
    // Fill inputs and outputs with random coin flips.
    Vector inputs(layer1_->input_dimension());
    for (int i = 0; i < layer1_->input_dimension(); ++i) {
      inputs[i] = runif() < .5;
    }
    std::vector<bool> outputs(layer1_->output_dimension());
    for (int i = 0; i < layer1_->output_dimension(); ++i) {
      outputs[i] = runif() < .5;
    }
    Vector log_activation_probs(layer1_->input_dimension());
    Vector log_complements = log_activation_probs;
    for (int i = 0; i < layer1_->input_dimension(); ++i) {
      double prob = runif();
      log_activation_probs[i] = log(prob);
      log_complements[i] = log(1 - prob);
    }
    double logp = imputer1_.input_full_conditional(
        inputs,
        outputs,
        log_activation_probs,
        log_complements);

    EXPECT_LT(logp , 0);

    // The full conditional is p(outputs | inputs) * p(inputs)
    double logp_manual = 0;
    for (int i = 0; i < layer1_->output_dimension(); ++i) {
      // Add plogis from the lower tail if outputs[i] is true and from the upper
      // tail if false.
      logp_manual += plogis(
          layer1_->logistic_regression(i)->predict(inputs),
          0, 1, outputs[i], true);
    }
    for (int i = 0; i < layer1_->input_dimension(); ++i) {
      logp_manual +=
          (inputs[i] > .5) ? log_activation_probs[i] : log_complements[i];
    }
    EXPECT_NEAR(logp, logp_manual, 1e-7);
  }

  //===========================================================================
  TEST_F(HiddenLayerImputerTest, ImputeInputsTest) {
    Vector allocation_probs(layer1_->input_dimension());
    for (int i = 0; i < allocation_probs.size(); ++i) {
      allocation_probs[i] = rbeta(.3, .3);
    }
    Vector workspace(allocation_probs.size());
    int niter = 10000;
    Nnet::HiddenNodeValues outputs;
    outputs.push_back(std::vector<bool>(layer0_->output_dimension()));
    outputs.push_back(std::vector<bool>(layer1_->output_dimension()));
    for (int i = 0; i < outputs[0].size(); ++i) {
      outputs[0][i] = runif() < allocation_probs[i];
    }
    Vector inputs(outputs[0].size());
    Nnet::to_numeric(outputs[0], inputs);
    for (int i = 0; i < inputs.size(); ++i) {
      EXPECT_NEAR(0, outputs[0][i] - inputs[i], 1e-10);
    }

    for (int i = 0; i < outputs[1].size(); ++i) {
      double prob = plogis(layer1_->logistic_regression(i)->predict(inputs));
      outputs[1][i] = runif() < prob;
    }

    auto original_outputs = outputs;
    Matrix input_draws(niter, layer1_->input_dimension());
    for (int i = 0; i < niter; ++i) {
      Vector log_allocation_probs = allocation_probs;
      Vector log_complements = log_allocation_probs;
      imputer1_.impute_inputs(GlobalRng::rng,
                              outputs,
                              log_allocation_probs,
                              log_complements,
                              workspace);

      EXPECT_TRUE(VectorEquals(log_allocation_probs,
                               log(allocation_probs)));
      EXPECT_TRUE(VectorEquals(exp(log_complements),
                               1. - allocation_probs));
      VectorView row(input_draws.row(i));
      Nnet::to_numeric(outputs[0], row);
    }

    Vector marginal_probs = mean(input_draws);
    for (int i = 0; i < marginal_probs.size(); ++i) {
      if (original_outputs[0][i]) {
        EXPECT_TRUE(marginal_probs[i] > .5);
      } else {
        EXPECT_TRUE(marginal_probs[i] < .5);
      }
    }
  }
  
}  // namespace HiddenLayerImputerTestNamespace
