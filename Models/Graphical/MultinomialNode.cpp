/*
  Copyright (C) 2005-2024 Steven L. Scott

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

#include "Models/Graphical/MultinomialNode.hpp"
#include <sstream>
#include "cpputil/report_error.hpp"
#include "stats/moments.hpp"

namespace BOOM {
  namespace Graphical {

    MultinomialNode::MultinomialNode(const DataTable &data,
                                     const std::string &variable_name)
        : Node(-1, variable_name)
    {
      const std::vector<std::string> &vnames(data.vnames());
      for (size_t i = 0; i < vnames.size(); ++i) {
        if (vnames[i] == variable_name) {
          set_variable_index(i);
          set_id(i);
          categorical_key_ = data.get_nominal(i).key();
          return;
        }
      }
      report_error("");
    }

    MultinomialNode::MultinomialNode(int node_id,
                                     const std::string &name,
                                     int variable_index,
                                     const Ptr<CatKey> &key)
        : Node(node_id, name, variable_index),
          categorical_key_(key)
    {}

    Array MultinomialNode::conditional_probability_table() const {
      ensure_models();

      std::vector<int> dims = parent_dims();
      dims.push_back(dim());
      Array ans(dims);

      for (auto it = models_.begin(); it != models_.end(); ++it) {
        std::vector<int> pos = it.position();
        pos.push_back(-1);
        ans.slice(pos) = (*it)->pi();
      }

      return ans;
    }

    double MultinomialNode::logp(const MixedMultivariateData &data_point) const {
      std::vector<Ptr<Node>> missing_parents;
      std::vector<Ptr<Node>> observed_parents;
      for (const Ptr<Node> &parent : parents()) {
        if (parent->is_missing(data_point)) {
          missing_parents.push_back(parent);
        } else {
          observed_parents.push_back(parent);
        }
      }

      // If all parents are observed, then just get the parent node data values
      // and look them up in models_.
      if (missing_parents.empty()) {
        std::vector<int> parent_values;
        for (const Ptr<Node> &parent : parents()) {
          parent_values.push_back(parent->categorical_value(data_point));
        }
        return models_[parent_values]->logpi()[
            this->categorical_value(data_point)];
      } else {
        // If any parent data are missing then (1) make sure the host model has
        // run the message passing algorithm. (2) Use the marginal distribution
        // of the parent nodes to compute the log density.
        report_error("logp for missing data is not yet implemented.");
        return 1.0;
      }
    }

    int MultinomialNode::categorical_value(
        const MixedMultivariateData &data_point) const {
      const LabeledCategoricalData &data(data_point.categorical(
          variable_index()));
      return data.missing() == Data::missing_status::observed ? data.value() : -1;
    }

    std::vector<int> MultinomialNode::parent_dims() const {
      std::vector<int> dims;
      for (const auto &parent : parents()) {
        if (parent->node_type() == NodeType::CATEGORICAL) {
          dims.push_back(parent->dim());
        } else {
          std::ostringstream err;
          err << "A MultinomialNode must have CATEGORICAL parents.";
          report_error(err.str());
        }
      }
      return dims;
    }

    void MultinomialNode::ensure_models() const {
      if (!models_current_) {
        int dim = this->dim();
        std::vector<int> parent_dims = this->parent_dims();
        std::vector<Ptr<MultinomialModel>> raw_model_storage;
        if (parent_dims.empty()) {
          parent_dims = std::vector<int>{1};
          raw_model_storage.push_back(new MultinomialModel(dim));
        } else {
          int num_models = prod(parent_dims);
          raw_model_storage.reserve(num_models);
          for (int i = 0; i < num_models; ++i) {
            raw_model_storage.push_back(new MultinomialModel(dim));
          }
        }
        models_ = GenericArray<Ptr<MultinomialModel>>(
            parent_dims, raw_model_storage);
        models_current_ = true;

        // TODO: The models still need priors and data.
      }
    }

  }  // namespace Graphical
}  // namespace BOOM
