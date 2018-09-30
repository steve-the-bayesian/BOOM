#ifndef BOOM_NNET_NNET_HPP_
#define BOOM_NNET_NNET_HPP_
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

#include <vector>
#include "Models/Glm/BinomialLogitModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "cpputil/RefCounted.hpp"

namespace BOOM {

  namespace Nnet {
    // A data structure to store imputed hidden node values for a single
    // observation.  If values is a HiddenNodeValues object, then
    // values[layer][node] stores the output for the specified node in the
    // specified layer.
    using HiddenNodeValues = std::vector<std::vector<bool>>;

    // The data augmentation algorithms used for posterior sampling involve a
    // lot of translating back and forth between binary and numeric
    // representations of the hidden node outputs.  These two functions help
    // smooth that out a bit.
    inline void to_binary(const Vector &numeric, std::vector<bool> &binary) {
      for (int i = 0; i < numeric.size(); ++i) {
        binary[i] = numeric[i] > .5;
      }
    }

    inline void to_numeric(const std::vector<bool> &binary, VectorView &numeric) {
      for (int i = 0; i < numeric.size(); ++i) {
        numeric[i] = binary[i];
      }
    }

    inline void to_numeric(const std::vector<bool> &binary, Vector &numeric) {
      VectorView view(numeric);
      to_numeric(binary, view);
    }

  }  // namespace Nnet
  
  //===========================================================================
  // The hidden layer is a concrete class because the abstraction is the same
  // regardless of the type of output being modeled.
  class HiddenLayer : private RefCounted {
   public:
    friend void intrusive_ptr_add_ref(HiddenLayer *layer) { layer->up_count(); }
    friend void intrusive_ptr_release(HiddenLayer *layer) {
      layer->down_count();
      if (layer->ref_count() == 0) delete layer;
    }

    HiddenLayer(int intput_dimension, int output_dimension);
    HiddenLayer(const HiddenLayer &rhs);
    HiddenLayer &operator=(const HiddenLayer &rhs);
    HiddenLayer(HiddenLayer &&rhs) = default;
    HiddenLayer &operator=(HiddenLayer &&rhs) = default;
    HiddenLayer *clone() const {return new HiddenLayer(*this);}
    
    int input_dimension() const;
    int output_dimension() const {return models_.size();}
    int number_of_nodes() const {return output_dimension();}

    // Args:
    //   inputs: The inputs to the hidden layer.  If this is the initial layer
    //     then the inputs are unconstrained.  Otherwise they are in [0, 1].
    //   outputs: The marginal probabilties that each output node is active.  
    void predict(const Vector &inputs, Vector &outputs) const;

    Ptr<BinomialLogitModel> logistic_regression(int node) {
      return models_[node];
    }
    const BinomialLogitModel &logistic_regression(int node) const {
      return *models_[node];
    }
    
   private:
    // There is one logistic regression model for each node in the layer.  Each
    // model corresponds to a single output.
    std::vector<Ptr<BinomialLogitModel>> models_;
  };

  //===========================================================================
  // A FeedForwardNeuralNetwork is an abstract class analogous to a GLM.  The
  // concrete class will be determined by the type of response variable.  Each
  // concrete GLM includes a terminal layer specific to that class, which is a
  // GLM for the appropriate response type.
  class FeedForwardNeuralNetwork :
      public CompositeParamPolicy,
      public PriorPolicy {
   public:
    FeedForwardNeuralNetwork();
    FeedForwardNeuralNetwork(const FeedForwardNeuralNetwork &rhs);
    FeedForwardNeuralNetwork(FeedForwardNeuralNetwork &&rhs) = default;
    FeedForwardNeuralNetwork &operator=(const FeedForwardNeuralNetwork &rhs);
    FeedForwardNeuralNetwork &operator=(FeedForwardNeuralNetwork &&rhs) = default;
    
    FeedForwardNeuralNetwork *clone() const override = 0;

    // Add a hidden layer to the network.  The caller must ensure that the input
    // dimension of 'layer' matches either the dimension of the predictors in
    // the data (if 'layer' is the first hidden layer added), or the output
    // dimension of the preceding layer.
    void add_layer(const Ptr<HiddenLayer> &layer);

    // Call this function after the last call to add_layer, so the terminal
    // layer can be adjusted to the correct number of coefficients.
    void finalize_network_structure();
    
    // The number of observations in the training data.  
    virtual int number_of_observations() const = 0;
    
    int number_of_hidden_layers() const {
      return hidden_layers_.size();
    }
    
    // Args:
    //   inputs: The data for the 'input layer'.  The observed predictors.
    //   activation_probs: Each element corresponds to the output of the
    //     corresponding hidden layer.  It is the caller's responsibility to
    //     ensure that this vector and its elements are sized correctly.  The
    //     model can allocate this data structure for you by calling
    //     activation_probability_workspace().
    //
    // Effects:
    //   Each element of activation_probs is filled with the probability that
    //   the corresponding node is active.
    void fill_activation_probabilities(
        const Vector &inputs,
        std::vector<Vector> &activation_probs) const;
    
    // Allocate a data structure that can be passed to
    // fill_activation_probabilities.
    std::vector<Vector> activation_probability_workspace() const;
    
    Ptr<HiddenLayer> hidden_layer(int i) {return hidden_layers_[i];}

   protected:
    // This flag starts off false and is set to false each time add_layer is called.
    bool finalized_;

    // To be called each time a hidden layer is added.  Concrete classes accept
    // this as a signal to resize or re-allocate the model implementing the
    // terminal layer if the dimension changes.
    virtual void restructure_terminal_layer(int dimension) = 0;

    // Fill the internal data structure 'prediction_workspace_' with the
    // activiation probabilities corresponding to 'inputs'.
    //
    // Args:
    //   inputs:  Observed predictors.
    void fill_prediction_workspace(const Vector &inputs) const {
      ensure_prediction_workspace();
      fill_activation_probabilities(inputs, prediction_workspace_);
    }

    // Access the final element of prediction_workspace_ after a call to
    // fill_prediction_workspace().  This returns the vector of inputs (in [0,
    // 1]) needed to evaluate a prediction from the terminal layer.
    const Vector &terminal_layer_inputs() const {
      return prediction_workspace_.back();
    }
    
   private:
    // This function is logically const.  It can change the mutable member
    // prediction_workspace_.
    void ensure_prediction_workspace() const;

    // The collection of hidden layers.  The input to layer zero comes from the
    // data.  The inputs to each remaining layer are the outputs of the
    // preceding layers.
    //
    // Each concrete child of a FeedForwardNeuralNetwork will define its own
    // terminal layer taking as inputs the outputs of hidden_layers_.back()
    std::vector<Ptr<HiddenLayer>> hidden_layers_;

    // A multi-threaded implementation would need a separate workspace for each
    // thread.
    mutable std::vector<Vector> prediction_workspace_;
  };
  
}  // namespace BOOM


#endif  // BOOM_NNET_NNET_HPP_


