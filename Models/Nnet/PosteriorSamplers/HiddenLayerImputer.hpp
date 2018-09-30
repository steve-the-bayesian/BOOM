#ifndef BOOM_MODELS_NNET_HIDDEN_LAYER_IMPUTER_HPP_
#define BOOM_MODELS_NNET_HIDDEN_LAYER_IMPUTER_HPP_
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

#include "Models/Nnet/Nnet.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

// Forward declaration of the test rig responsible for HiddenLayerImputer.
namespace HiddenLayerImputerTestNamespace {
  class HiddenLayerImputerTest;
}

namespace BOOM {
  // A HiddenLayerImputer manages the imputed data for a single hidden layer in
  // a feed forward neural network.
  class HiddenLayerImputer {
   public:
    // Args:
    //   layer:  The hidden layer to be managed by this object.
    //   layer_index: The position of 'layer' in the FeedForwardNeuralNetwork
    //     object that owns it.
    HiddenLayerImputer(const Ptr<HiddenLayer> &layer, int layer_index);

    // Perform an MCMC update on the vector of inputs to the managed layer.
    //
    // Args:
    //   rng:  The random number generator.
    //   outputs: The set of outputs for all hidden nodes in the network.  The
    //     inputs to this layer are the outputs to the preceding layer.
    //   allocation_probs: On input this contains the marginal probability that
    //     each input to the managed layer is active, conditional on network
    //     parameters and predictors for the observation being imputed.  On
    //     output the marginal probabilities are replaced by their logarithms.
    //   complementary_allocation_probs: On input this is an arbitrary vector of
    //     the same size as allocation_probs.  On output the entries are the
    //     logarithms of the probabilities complementary to allocation_probs.
    //   input_workspace: A third vector of the same size as allocation_probs
    //     for storing layer inputs as numeric values.
    //
    // Effects:
    //   * The value of outputs[layer_index_ - 1] is updated by an MCMC draw from
    //     its full conditional distribution.  If layer_index_ is zero then
    //     the call is a no-op.
    //   * The imputed latent data is added to the logistic regression models
    //     implementing the managed node.
    void impute_inputs(RNG &rng,
                       Nnet::HiddenNodeValues &outputs,
                       Vector &allocation_probs,
                       Vector &complementary_allocation_probs,
                       Vector &input_workspace);

    // The conditional distribution for the vector of inputs to this layer,
    // given the set of predictors and model parameters, and given the outputs
    // for the layer.
    //
    // Args:
    //   inputs: The vector of inputs to the hidden layer.  Each entry is
    //     assumed to be either 0 or 1.
    //   outputs: The vector of outputs for the layer, assumed to have been
    //     imputed from the layer above.
    //   logprob: Contains the log of the marginal prior probability that each
    //     input node is active, obtained by evaluating the predictors,
    //     coefficients, and activation functions.
    //   logprob_complement: The log of the marginal probability that each input
    //     node is INactive.  I.e. the log of the complementary probability to
    //     logprob.
    //
    // Returns:
    //   The un-normalized density of 'inputs'.
    double input_full_conditional(
        const Vector &inputs,
        const std::vector<bool> &outputs,
        const Vector &logprob,
        const Vector &logprob_complement) const;

    // Remove latent data from the logistic regression models in the managed
    // layer, and set all counts of latent variables to zero.
    void clear_latent_data();

    // Store the imputed outputs of the first hidden layer.
    // Args:
    //   outputs:  The imputed outputs for the first hidden layer.
    //   data_point:  The observed data point for this observation.
    void store_initial_layer_latent_data(
        const std::vector<bool>  &outputs,
        const Ptr<GlmBaseData> &data_point);

    // Store the latent data simulated from impute_inputs in the logistic
    // regression models making up the hidden layer, and in the data store
    // managed by this object.
    void store_latent_data(Nnet::HiddenNodeValues &outputs);

   private:
    // For testing.  Let the test rig access private data.
    friend class HiddenLayerImputerTestNamespace::HiddenLayerImputerTest;
    
    std::vector<Ptr<BinomialRegressionData>>
    get_initial_data(const Ptr<GlmBaseData> &data_point);
    
    // Retrieve a specified row of data from the appropriate data store,
    // creating and adding it to the store if it does not already exist.
    std::vector<Ptr<BinomialRegressionData>> get_data_row(
        const std::vector<bool> &inputs);

    // Add the data row to the active data store and add its elements to the
    // logistic regression models in the managed layer.
    void install_data_row(const std::vector<bool> &inputs,
                          const std::vector<Ptr<BinomialRegressionData>> &row);
    
    // The hidden layer managed by this object.
    Ptr<HiddenLayer> layer_;

    // The managed layer's position in the network (0 being closest to the
    // predictors).
    int layer_index_;

    // The active data store stores data that will be added to one of the
    // logistic regression models for the node.  The active data store is
    // cleared out each time you call clear_data, and each of its elements has
    // 'n' and 'y' set to zero.
    std::map<std::vector<bool>,
             std::vector<Ptr<BinomialRegressionData>>> active_data_store_;

    // Long term storage for hidden layer latent data.  This is an optimization
    // to cut down on the number of memory allocations.  Data points are stored
    // here so they can be reused later without reallocating.
    //
    // TODO: This scheme can suck up a lot of memory.  It might be useful to
    // clear this data structure once it grows more than say, 10x, times the
    // size of the data stored in the model.
    std::map<std::vector<bool>,
             std::vector<Ptr<BinomialRegressionData>>> long_term_data_store_;

    // Stores data for the initial hidden layer.
    std::map<Ptr<VectorData>,
             std::vector<Ptr<BinomialRegressionData>>> initial_data_store_;
  };
  
}  // namespace BOOM

#endif  // BOOM_MODELS_NNET_HIDDEN_LAYER_IMPUTER_HPP_

