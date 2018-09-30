/*
  Copyright (C) 2005-2018 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/Nnet/PosteriorSamplers/GaussianFeedForwardPosteriorSampler.hpp"
#include "distributions.hpp"
#include "cpputil/lse.hpp"

namespace BOOM {

  namespace {
    using GFFPS = GaussianFeedForwardPosteriorSampler;
  }  // namespace 
  
  GFFPS::GaussianFeedForwardPosteriorSampler(
      GaussianFeedForwardNeuralNetwork *model,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model)
  {}

  double GFFPS::logpri() const {
    report_error("Not yet implemented");
    return negative_infinity();
  }

  void GFFPS::draw() {
    ensure_imputers();
    impute_hidden_layer_outputs(rng());
    draw_parameters_given_hidden_nodes();
  }
  
  // The imputation method is a "collapsed Gibbs sampler" that integrates out
  // latent data from preceding layers (i.e. preceding nodes are activated
  // probabilistically), but conditions on the latent data from the current
  // layer and the layer above.
  void GFFPS::impute_hidden_layer_outputs(RNG &rng) {
    int number_of_hidden_layers = model_->number_of_hidden_layers();
    if (number_of_hidden_layers == 0) return;
    ensure_space_for_latent_data();
    clear_latent_data();
    std::vector<Vector> allocation_probs =
        model_->activation_probability_workspace();
    std::vector<Vector> complementary_allocation_probs = allocation_probs;
    std::vector<Vector> workspace = allocation_probs;
    for (int i = 0; i < model_->dat().size(); ++i) {
      const Ptr<RegressionData> &data_point(model_->dat()[i]);
      Nnet::HiddenNodeValues &outputs(imputed_hidden_layer_outputs_[i]);
      model_->fill_activation_probabilities(data_point->x(), allocation_probs);
      impute_terminal_layer_inputs(rng, data_point->y(), outputs.back(),
                                   allocation_probs.back(),
                                   complementary_allocation_probs.back());
      for (int layer = number_of_hidden_layers - 1; layer > 0; --layer) {
        // This for-loop intentionally skips layer 0, because the inputs to the
        // first hidden layer are the observed predictors.
        imputers_[layer].impute_inputs(
            rng,
            outputs,
            allocation_probs[layer - 1],
            complementary_allocation_probs[layer - 1],
            workspace[layer - 1]);
      }
      imputers_[0].store_initial_layer_latent_data(outputs[0], data_point);
    }
  }

  std::pair<double, double> summarize_logit_data(
      const std::vector<Ptr<BinomialRegressionData>> &data) {
    std::pair<double, double> ans = {0, 0};
    for (int i = 0; i < data.size(); ++i) {
      ans.first += data[i]->y();
      ans.second += data[i]->n();
    }
    return ans;
  }
  
  // Simulate the parameters of the logistic and linear regression models,
  // conditional on sampled values of the data from the hidden nodes.
  //
  // TODO:  exploit parallelism
  void GFFPS::draw_parameters_given_hidden_nodes() {
    model_->terminal_layer()->sample_posterior();
    for (int i = 0; i < model_->number_of_hidden_layers(); ++i) {
      Ptr<HiddenLayer> layer = model_->hidden_layer(i);
      for (int node = 0; node < layer->output_dimension(); ++node) {
        layer->logistic_regression(node)->sample_posterior();
      }
    }
  }
  
  // Clear the data from the models defining the hidden and terminal layers, and
  // clear any latent data structures stored by the imputers.
  void GFFPS::clear_latent_data() {
    model_->terminal_layer()->suf()->clear();
    for (int i = 0; i < model_->number_of_hidden_layers(); ++i) {
      imputers_[i].clear_latent_data();
    }
  }

  // Args:
  //   response:   The response variable for a single observation.
  //   binary_inputs: The vector of inputs to the terminal layer.  Each element
  //     must be either 0 or 1, and it is the caller's responsibility to verify.
  //   logprob: A vector containing the log of the probabilities that each node
  //     in the terminal layer inputs is 'on'.
  //   logprob_complement: A vector containing the log of the probabilities
  //     that each node in the terminal layer inputs is 'off'.
  //
  // Returns:
  //   The log of the un-normalized full conditional distribution of the
  //   terminal layer inputs.
  double GFFPS::terminal_inputs_log_full_conditional(
      double response,
      const Vector &binary_inputs,
      const Vector &logprob,
      const Vector &logprob_complement) const {
    double ans = dnorm(
        response,
        model_->terminal_layer()->predict(binary_inputs),
        model_->terminal_layer()->sigma(),
        true);
    for (int i = 0; i < binary_inputs.size(); ++i) {
      ans += binary_inputs[i] > .5 ? logprob[i] : logprob_complement[i];
    }
    return ans;
  }

  // Set up space for storing the outputs of the hidden layers.
  void GFFPS::ensure_space_for_latent_data() {
    if (imputed_hidden_layer_outputs_.size() != model_->dat().size()) {
      imputed_hidden_layer_outputs_.clear();
      imputed_hidden_layer_outputs_.reserve(model_->dat().size());
      int number_of_hidden_layers = model_->number_of_hidden_layers();
      for (int i = 0; i < model_->dat().size(); ++i) {
        std::vector<std::vector<bool>> element;
        element.reserve(number_of_hidden_layers);
        for (int layer = 0; layer < number_of_hidden_layers; ++layer) {
          element.push_back(std::vector<bool>(
              model_->hidden_layer(layer)->output_dimension()));
        }
        imputed_hidden_layer_outputs_.push_back(element);
      }
    }
  }
  
  void GFFPS::ensure_imputers() {
    while (imputers_.size() < model_->number_of_hidden_layers()) {
      imputers_.push_back(HiddenLayerImputer(
          model_->hidden_layer(imputers_.size()), imputers_.size()));
    }
  }

  // Args:
  //   rng:  The random number generator.
  //   binary_inputs: The value of the inputs to the terminal layer (i.e. the
  //     outputs from the final hidden layer).  These will be updated by the
  //     imputation.
  //   logprob: On input this is a vector giving the marginal (un-logged)
  //     probability that each input node is active.  These values will be
  //     over-written by their logarithms.
  //   logprob_complement: On input this is any vector with size matching
  //     logprob.  On output its elements contain log(1 - exp(logprob)).
  //
  // Effects:
  //   The latent data for the terminal layer is imputed, and the sufficient
  //   statistics for the latent regression model in the terminal layer are
  //   updated to included the imputed data.
  void GFFPS::impute_terminal_layer_inputs(
      RNG &rng,
      double response,
      std::vector<bool> &binary_inputs,
      Vector &logprob,
      Vector &logprob_complement) {
    for (int i = 0; i < logprob.size(); ++i) {
      logprob_complement[i] = log(1 - logprob[i]);
      logprob[i] = log(logprob[i]);
    }
    Vector terminal_layer_inputs(binary_inputs.size());
    Nnet::to_numeric(binary_inputs, terminal_layer_inputs);
    double logp_original = terminal_inputs_log_full_conditional(
        response, terminal_layer_inputs, logprob, logprob_complement);
    for (int i = 0; i < terminal_layer_inputs.size(); ++i) {
      terminal_layer_inputs[i] = 1 - terminal_layer_inputs[i];
      double logp = terminal_inputs_log_full_conditional(
          response, terminal_layer_inputs, logprob, logprob_complement);
      double log_input_prob = logp - lse2(logp, logp_original);
      double logu = log(runif_mt(rng));
      if (logu < log_input_prob) {
        logp_original = logp;
      } else {
        terminal_layer_inputs[i] = 1 - terminal_layer_inputs[i];
      }
    }
    model_->terminal_layer()->suf()->add_mixture_data(
        response, terminal_layer_inputs, 1.0);
    Nnet::to_binary(terminal_layer_inputs, binary_inputs);
  }

}  // namespace BOOM
