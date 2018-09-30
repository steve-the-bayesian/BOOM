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

#include "Models/Nnet/PosteriorSamplers/HiddenLayerImputer.hpp"
#include "distributions.hpp"
#include "cpputil/lse.hpp"

namespace BOOM {

  HiddenLayerImputer::HiddenLayerImputer(const Ptr<HiddenLayer> &layer,
                                         int layer_index) 
      : layer_(layer),
        layer_index_(layer_index)
    {}
  
  //---------------------------------------------------------------------------
  void HiddenLayerImputer::impute_inputs(
      RNG &rng,
      Nnet::HiddenNodeValues &outputs,
      Vector &allocation_probs,
      Vector &complementary_allocation_probs,
      Vector &input_workspace) {
    if (layer_index_ <= 0) return;
    std::vector<bool> &inputs(outputs[layer_index_ - 1]);
    Nnet::to_numeric(inputs, input_workspace);
    for (int i = 0; i < allocation_probs.size(); ++i) {
      complementary_allocation_probs[i] = log(1 - allocation_probs[i]);
      allocation_probs[i] = log(allocation_probs[i]);
    }
    double logp_current = input_full_conditional(
        input_workspace,
        outputs[layer_index_],
        allocation_probs,
        complementary_allocation_probs);
    for (int i = 0; i < input_workspace.size(); ++i) {
      input_workspace[i] = 1 - input_workspace[i];
      double logp_cand = input_full_conditional(
          input_workspace,
          outputs[layer_index_],
          allocation_probs,
          complementary_allocation_probs);
      double logu = log(runif_mt(rng));
      double log_input_prob = logp_cand - lse2(logp_cand, logp_current);
      if (logu < log_input_prob) {
        // Accept the draw by keeping the candidate and updating the current
        // value of logp.
        // Note that this is the Gibbs sampling draw
        logp_current = logp_cand;
        inputs[i] = 1 - inputs[i];
      } else {
        // Reject the draw by putting input_workspace back the way it was.
        input_workspace[i] = 1 - input_workspace[i];
      }
    }
    store_latent_data(outputs);
  }

  //---------------------------------------------------------------------------
  double HiddenLayerImputer::input_full_conditional(
      const Vector &inputs,
      const std::vector<bool> &outputs,
      const Vector &logp,
      const Vector &logp_complement) const {
    double ans = 0;
    for (int node = 0; node < outputs.size(); ++node) {
      double logit = layer_->logistic_regression(node)->predict(inputs);
      ans += plogis(logit, 0, 1, outputs[node], true);
    }
    for (int i = 0; i < inputs.size(); ++i) {
      ans += inputs[i] > .5 ? logp[i] : logp_complement[i];
    }
    return ans;
  }
      
  //---------------------------------------------------------------------------
  void HiddenLayerImputer::clear_latent_data() {
    if (layer_index_ > 0) {
      for (auto &row : active_data_store_) {
        for (auto &data_point : row.second) {
          data_point->set_y(0);
          data_point->set_n(0);
        }
      }
      active_data_store_.clear();
      for (int i = 0; i < layer_->output_dimension(); ++i) {
        layer_->logistic_regression(i)->clear_data();
      }
    } else {
      for (int node = 0; node < layer_->output_dimension(); ++node) {
        std::vector<Ptr<BinomialRegressionData>> &latent_data(
            layer_->logistic_regression(node)->dat());
        for (int i = 0; i < latent_data.size(); ++i) {
          latent_data[i]->set_y(0.0);
          latent_data[i]->set_n(0.0);
        }
      }
    }
  }

  //---------------------------------------------------------------------------
  // Store the parts of the hidden layer outputs from a single observation that
  // are relevant to this layer.  That is, take the inputs and outputs from this
  // layer and use them to update the latent data sets.
  void HiddenLayerImputer::store_latent_data(Nnet::HiddenNodeValues &outputs) {
    if (layer_index_ <= 0) {
      report_error("Don't call store_latent_data for hidden layer 0.");
    }
    std::vector<bool> &inputs(outputs[layer_index_ - 1]);
    // Find the data point for the node that corresponds to 'inputs'.
    std::vector<Ptr<BinomialRegressionData>> data_row = get_data_row(inputs);
    for (int i = 0; i < data_row.size(); ++i) {
      // Each element of 'data_row' is a BinomialRegressionData with predictors
      // matching the input vector.  Each element corresponds to a different
      // node in the hidden layer.  If the corresponding output is 'on' then
      // increment the data point.  Increment the observation count in any case.
      data_row[i]->increment(outputs[layer_index_][i], 1.0);
    }
  }

  //---------------------------------------------------------------------------
  std::vector<Ptr<BinomialRegressionData>> HiddenLayerImputer::get_data_row(
      const std::vector<bool> &inputs) {
    // If inputs is in active data storage return the answer.
    auto it = active_data_store_.find(inputs);
    if (it != active_data_store_.end()) return it->second;

    // Check if inputs exists in the long term data store add it to the active
    // data store, and return the answer.
    it = long_term_data_store_.find(inputs);
    if (it != long_term_data_store_.end()) {
      install_data_row(inputs, it->second);
      return it->second;
    }

    // Otherwise, create a new vector, add it to both data stores, and return
    // the answer.
    Vector workspace(inputs.size());
    Nnet::to_numeric(inputs, workspace);
    std::vector<Ptr<BinomialRegressionData>> data_row;
    data_row.reserve(layer_->output_dimension());
    NEW(VectorData, predictors)(workspace);
    for (int i = 0; i < layer_->output_dimension(); ++i) {
      NEW(BinomialRegressionData, data_point)(0, 0, predictors);
      data_row.push_back(data_point);
    }
    long_term_data_store_[inputs] = data_row;
    install_data_row(inputs, data_row);
    return data_row;
  }

  //---------------------------------------------------------------------------
  void HiddenLayerImputer::install_data_row(
      const std::vector<bool> &inputs,
      const std::vector<Ptr<BinomialRegressionData>> &data_row) {
    active_data_store_[inputs] = data_row;
    for (int i = 0; i < layer_->output_dimension(); ++i) {
      layer_->logistic_regression(i)->add_data(data_row[i]);
    }
  }

  //---------------------------------------------------------------------------
  void HiddenLayerImputer::store_initial_layer_latent_data(
      const std::vector<bool> &outputs,
      const Ptr<GlmBaseData> &data_point) {
    if (layer_index_ != 0) {
      report_error("Only the first hidden layer can store initial layer "
                   "latent data.");
    }
    std::vector<Ptr<BinomialRegressionData>> data_row =
        get_initial_data(data_point);
    for (int i = 0; i < data_row.size(); ++i) {
      data_row[i]->set_n(1.0);
      data_row[i]->set_y(outputs[i]);
    }
  }

  //---------------------------------------------------------------------------
  std::vector<Ptr<BinomialRegressionData>>
  HiddenLayerImputer::get_initial_data(const Ptr<GlmBaseData> &data_point) {
    auto it = initial_data_store_.find(data_point->Xptr());
    if (it == initial_data_store_.end()) {
      std::vector<Ptr<BinomialRegressionData>> data_row;
      data_row.reserve(layer_->output_dimension());
      for (int i = 0; i < layer_->output_dimension(); ++i) {
        NEW(BinomialRegressionData, hidden_node_data)(0, 0, data_point->Xptr());
        data_row.push_back(hidden_node_data);
        layer_->logistic_regression(i)->add_data(hidden_node_data);
      }
      initial_data_store_[data_point->Xptr()] = data_row;
      return data_row;
    } else {
      return it->second;
    }
  }
  
}  // namespace BOOM
