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
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  HiddenLayer::HiddenLayer(int input_dimension, int output_dimension) {
    if (input_dimension <= 0 || output_dimension <= 0) {
      report_error("Both input_dimension and output_dimension must be "
                   "positive.");
    }
    for (int i = 0; i < output_dimension; ++i) {
      models_.push_back(new BinomialLogitModel(input_dimension));
    }
  }

  HiddenLayer::HiddenLayer(const HiddenLayer &rhs) {
    models_.reserve(rhs.models_.size());
    for (int i = 0; i < rhs.models_.size(); ++i) {
      models_.push_back(rhs.models_[i]->clone());
    }
  }

  HiddenLayer &HiddenLayer::operator=(const HiddenLayer &rhs) {
    if (&rhs != this) {
      models_.clear();
      models_.reserve(rhs.models_.size());
      for (int i = 0; i < models_.size(); ++i) {
        models_.push_back(rhs.models_[i]->clone());
      }
    }
    return *this;
  }
  
  int HiddenLayer::input_dimension() const {
    if (models_.empty()) {
      return -1;
    } else {
      return models_[0]->xdim();
    }
  }

  void HiddenLayer::predict(const Vector &inputs, Vector &outputs) const {
    if (inputs.size() != input_dimension() ||
        outputs.size() != output_dimension()) {
      report_error("Either inputs or outputs are the wrong dimension in "
                   "HiddenLayer::predict.");
    }

    for (int i = 0; i < outputs.size(); ++i) {
      outputs[i] = plogis(models_[i]->predict(inputs));
    }
  }
  
  //===========================================================================
  namespace {
    using FFNN = FeedForwardNeuralNetwork;
  }  // namespace
  
  FFNN::FeedForwardNeuralNetwork()
      : finalized_(false) {}

  FFNN::FeedForwardNeuralNetwork(const FFNN &rhs)
      : ParamPolicy(rhs),
        PriorPolicy(rhs)
  {
    for (int i = 0; i < rhs.hidden_layers_.size(); ++i) {
      add_layer(rhs.hidden_layers_[i]->clone());
    }
    finalized_ = rhs.finalized_;
  }

  FFNN & FFNN::operator=(const FFNN &rhs) {
    if (&rhs != this) {
      ParamPolicy::operator=(rhs);
      PriorPolicy::operator=(rhs);
      for (int i = 0; i < rhs.hidden_layers_.size(); ++i) {
        add_layer(rhs.hidden_layers_[i]->clone());
      }
      finalize_network_structure();
    }
    return *this;
  }
  
  void FFNN::add_layer(const Ptr<HiddenLayer> &layer) {
    if (!hidden_layers_.empty()) {
      if (hidden_layers_.back()->output_dimension() != layer->input_dimension()) {
        std::ostringstream err;
        err << "Input dimension of new layer (" << layer->input_dimension()
            << ") does not match the output dimension of the previous layer ("
            << hidden_layers_.back()->output_dimension() << ".";
        report_error(err.str());
      }
    }
    hidden_layers_.push_back(layer);
    for (int i = 0; i < layer->output_dimension(); ++i) {
      ParamPolicy::add_model(layer->logistic_regression(i));
    }
    finalized_ = false;
  }

  void FFNN::finalize_network_structure() {
    restructure_terminal_layer(hidden_layers_.back()->output_dimension());
    finalized_ = true;
  }
  
  void FFNN::fill_activation_probabilities(
      const Vector &inputs,
      std::vector<Vector> &activation_probs) const {
    const Vector *in = &inputs;
    for (int i = 0; i < hidden_layers_.size(); ++i) {
      hidden_layers_[i]->predict(*in, activation_probs[i]);
      in = &activation_probs[i];
    }
  }

  std::vector<Vector> FFNN::activation_probability_workspace() const {
    std::vector<Vector> ans;
    for (int i = 0; i < hidden_layers_.size(); ++i) {
      ans.emplace_back(hidden_layers_[i]->output_dimension());
    }
    return ans;
  }
  
  void FFNN::ensure_prediction_workspace() const {
    if (prediction_workspace_.size() != hidden_layers_.size()) {
      prediction_workspace_ = activation_probability_workspace();
    }
  }
  
}  // namespace BOOM
