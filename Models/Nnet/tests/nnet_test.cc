#include "gtest/gtest.h"
#include "Models/Nnet/Nnet.hpp"
#include "Models/Nnet/GaussianFeedForwardNeuralNetwork.hpp"

#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using namespace BOOM::Nnet;
  using std::endl;
  using std::cout;

  class NnetTest : public ::testing::Test {
   protected:
    NnetTest()
        : layer1_(new HiddenLayer(3, 2)),
          layer2_(new HiddenLayer(2, 3))
    {
      GlobalRng::rng.seed(8675309);

      // Input dimension is 3.  Two hidden layers with 2 and 3 node each.
      network_.add_layer(layer1_);
      network_.add_layer(layer2_);
      network_.finalize_network_structure();
      randomize(network_);
      EXPECT_TRUE(VectorEquals(
        layer1_->logistic_regression(0)->Beta(),
        network_.hidden_layer(0)->logistic_regression(0)->Beta()));
    }

    //---------------------------------------------------------------------------
    // Fill all model parameters with randomly generated coefficients.  Logistic
    // and regression coefficients get N(0, 1) noise, and the residual variance
    // gets Exponential(1).
    void randomize(GaussianFeedForwardNeuralNetwork &network) {
      for (int i = 0; i < network.number_of_hidden_layers(); ++i) {
        Ptr<HiddenLayer> layer = network.hidden_layer(i);
        for (int j = 0; j < layer->number_of_nodes(); ++j) {
          Ptr<BinomialLogitModel> logit = layer->logistic_regression(j);
          Vector b = logit->Beta();
          for (int k = 0; k < b.size(); ++k) {
            b[k] = rnorm();
          }
          logit->set_Beta(b);
        }
      }

      Ptr<RegressionModel> terminal = network.terminal_layer();
      Vector b = terminal->Beta();
      for (int i = 0; i < b.size(); ++i) {
        b[i] = rnorm();
      }
      terminal->set_Beta(b);
      terminal->set_sigsq(rexp());
    }

    //---------------------------------------------------------------------------
    // Print all the parameters of the specified model to a string which can be
    // passed to the logging macros.
    std::string PrintAllParameters(
        const GaussianFeedForwardNeuralNetwork &model) {
      auto params = model.parameter_vector();
      std::ostringstream out;
      for (const auto &el : params) {
        out << *el << std::endl;
      }
      return out.str();
    }
    
    GaussianFeedForwardNeuralNetwork network_;
    Ptr<HiddenLayer> layer1_;
    Ptr<HiddenLayer> layer2_;
  };

  //===========================================================================
  TEST_F(NnetTest, HiddenLayerTest) {
    // Test for the right size inputs and outputs.
    EXPECT_EQ(3, layer1_->input_dimension());
    EXPECT_EQ(2, layer1_->output_dimension());
    EXPECT_EQ(layer1_->output_dimension(),
              layer1_->number_of_nodes());

    // Test the copy constructor.
    HiddenLayer layer1_copy(*layer1_);
    EXPECT_EQ(layer1_copy.input_dimension(),
              layer1_->input_dimension());
    EXPECT_EQ(layer1_copy.output_dimension(),
              layer1_->output_dimension());

    for (int i = 0; i < layer1_->output_dimension(); ++i) {
      EXPECT_TRUE(VectorEquals(
          layer1_->logistic_regression(i)->Beta(),
          layer1_copy.logistic_regression(i)->Beta()));
    }

    // Test the predict method (also tested implicitly below).
    Vector x(layer1_->input_dimension());
    x.randomize();
    Vector hidden(layer1_->output_dimension());
    layer1_->predict(x, hidden);

    Vector manual_hidden(hidden.size());
    for (int i = 0; i < layer1_->output_dimension(); ++i) {
      manual_hidden[i] = plogis(layer1_->logistic_regression(i)->predict(x));
    }
    EXPECT_TRUE(VectorEquals(manual_hidden, hidden));
  }

  //===========================================================================
  TEST_F(NnetTest, Construction) {
    GaussianFeedForwardNeuralNetwork empty;
    EXPECT_EQ(0, empty.number_of_hidden_layers());

    // The network "network_" was constructed as part of the test framework.
    // Make sure all the sizes are as expected.
    EXPECT_EQ(2, network_.number_of_hidden_layers());
    EXPECT_EQ(3, network_.hidden_layer(0)->input_dimension());
    EXPECT_EQ(2, network_.hidden_layer(0)->output_dimension());
    EXPECT_EQ(2, network_.hidden_layer(0)->number_of_nodes());

    EXPECT_EQ(2, network_.hidden_layer(1)->input_dimension());
    EXPECT_EQ(3, network_.hidden_layer(1)->output_dimension());
    EXPECT_EQ(3, network_.hidden_layer(1)->number_of_nodes());

    EXPECT_EQ(3, network_.terminal_layer()->xdim());
  }

  //===========================================================================
  TEST_F(NnetTest, CopyConstructor) {
    GaussianFeedForwardNeuralNetwork net2(network_);
    EXPECT_EQ(2, net2.number_of_hidden_layers());
    EXPECT_EQ(3, net2.hidden_layer(0)->input_dimension());
    EXPECT_EQ(2, net2.hidden_layer(0)->output_dimension());
    EXPECT_EQ(2, net2.hidden_layer(0)->number_of_nodes());

    EXPECT_EQ(2, net2.hidden_layer(1)->input_dimension());
    EXPECT_EQ(3, net2.hidden_layer(1)->output_dimension());
    EXPECT_EQ(3, net2.hidden_layer(1)->number_of_nodes());

    EXPECT_EQ(3, net2.terminal_layer()->xdim());

    EXPECT_EQ(network_.parameter_vector().size(),
              net2.parameter_vector().size())
        << "original network parameters: " << std::endl
        << PrintAllParameters(network_) << std::endl
        << "copied network parameters: " << std::endl
        << PrintAllParameters(net2) << std::endl;
  }
  
  //===========================================================================
  // Test that the predict method works as advertised.
  TEST_F(NnetTest, PredictTest) {
    Vector x(layer1_->input_dimension());
    x.randomize();
    double pred = network_.predict(x);

    // Now recreate the prediction manually and check that it agrees with 'pred'.
    Vector h1(layer1_->number_of_nodes());
    Vector h2(layer2_->number_of_nodes());
    for (int i = 0; i < layer1_->number_of_nodes(); ++i) {
      h1[i] = plogis(layer1_->logistic_regression(i)->predict(x));
    }
    for (int i = 0; i < layer2_->number_of_nodes(); ++i) {
      h2[i] = plogis(layer2_->logistic_regression(i)->predict(h1));
    }
    double manual_pred = network_.terminal_layer()->predict(h2);
    EXPECT_NEAR(manual_pred, pred, 1e-7);
  }

  //===========================================================================
  TEST_F(NnetTest, FillActivationProbabilities) {
    Vector x(layer1_->input_dimension());
    x.randomize();

    std::vector<Vector> activation_probs =
        network_.activation_probability_workspace();
    std::vector<Vector> manual_activation_probs = activation_probs;
        
    EXPECT_EQ(activation_probs.size(), 2);
    // There is one activation probabilty for each node in a layer.
    EXPECT_EQ(2, activation_probs[0].size());
    EXPECT_EQ(3, activation_probs[1].size());
    network_.fill_activation_probabilities(x, activation_probs);

    for (int i = 0; i < layer1_->number_of_nodes(); ++i) {
      manual_activation_probs[0][i] = plogis(
          layer1_->logistic_regression(i)->predict(x));
    }
    for (int i = 0; i < layer2_->number_of_nodes(); ++i) {
      manual_activation_probs[1][i] = plogis(
          layer2_->logistic_regression(i)->predict(
              manual_activation_probs[0]));
    }
    EXPECT_TRUE(VectorEquals(activation_probs[0], manual_activation_probs[0]));
    EXPECT_TRUE(VectorEquals(activation_probs[1], manual_activation_probs[1]));
  }
  
}  // namespace
