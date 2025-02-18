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

#include "Models/MultilevelMultinomialModel.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/math_utils.hpp"


namespace BOOM {


  MultilevelMultinomialModel::MultilevelMultinomialModel(
      const Ptr<Taxonomy> &tax)
      : taxonomy_(tax),
        only_keep_suf_(false)
  {
    create_models();
  }

  MultilevelMultinomialModel::MultilevelMultinomialModel(
      const MultilevelMultinomialModel &rhs)
      : Model(rhs),
        MixtureComponent(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs)
  {}

  MultilevelMultinomialModel & MultilevelMultinomialModel::operator=(
      const MultilevelMultinomialModel &rhs) {
    if (&rhs != this) {
      report_error("NYI");
    }
    return *this;
  }
  
  MultilevelMultinomialModel *MultilevelMultinomialModel::clone() const {
    return new MultilevelMultinomialModel(*this);
  }

  
  double MultilevelMultinomialModel::logp(
      const MultilevelCategoricalData &data_point) const {
    const std::vector<int> &levels(data_point.levels());
    if (levels.empty()) {
      return negative_infinity();
    }
    double ans = top_level_model_->logp(levels[0]);
    const TaxonomyNode *node = taxonomy_->top_level_node(levels[0]);
    for (int i = 1; i < levels.size(); ++i) {
      ans += conditional_model(node)->logp(levels[i]);
      node = node->child(levels[i]);
    }
    return ans;
  }

  double MultilevelMultinomialModel::pdf(const Data *dp, bool logscale) const {
    const MultilevelCategoricalData *data_point =
        dynamic_cast<const MultilevelCategoricalData *>(dp);
    if (!data_point) {
      report_error("Failed cast in MultilevelMultinomialModel::pdf.");
    }
    double ans = logp(*data_point);
    return logscale ? ans : exp(ans);
  }

  int MultilevelMultinomialModel::number_of_observations() const {
    return top_level_model_->number_of_observations();
  }


  void MultilevelMultinomialModel::add_data(const Ptr<Data> &dp) {
    Ptr<MultilevelCategoricalData> data_point(
        dp.dcast<MultilevelCategoricalData>());
    add_data(data_point);
  }

  void MultilevelMultinomialModel::add_data(
      const Ptr<MultilevelCategoricalData> &data_point) {
    const std::vector<int> &levels(data_point->levels());
    if (levels.empty()) {
      return;
    }

    top_level_model_->suf()->update_raw(levels[0]);
    TaxonomyNode *taxonomy_node = taxonomy_->top_level_node(levels[0]);
    for (int i = 0; i < levels.size(); ++i) {
      Ptr<MultinomialModel> &model(conditional_models_[taxonomy_node]);
      model->suf()->update_raw(levels[i]);
      if (taxonomy_node->is_leaf()) {
        break;
      } else {
        taxonomy_node = taxonomy_node->child(levels[i]);
      }
    }
  }

  void MultilevelMultinomialModel::clear_data() {
    data_.clear();
    top_level_model_->clear_data();
    for (auto it = taxonomy_->begin(); it != taxonomy_->end(); ++it) {
      const Ptr<TaxonomyNode> &node(*it);
      if (!node->is_leaf()) {
        conditional_models_[node.get()]->clear_data();
      }
    }
  }

  void MultilevelMultinomialModel::combine_data(const Model &other_model,
                                                bool just_suf) {
    const MultilevelMultinomialModel &rhs(
        dynamic_cast<const MultilevelMultinomialModel &>(other_model));
    
    if (*taxonomy_ != *rhs.taxonomy_) {
      report_error("Models must have the same taxonomy.");
    }

    top_level_model_->combine_data(*rhs.top_level_model_, true);
    auto it1 = taxonomy_->begin();
    auto it2 = rhs.taxonomy_->begin();
    for (; it1 != taxonomy_->end(); ++it1, ++it2) {
      Ptr<TaxonomyNode> &node1(*it1);
      if (!node1->is_leaf()) {
        Ptr<TaxonomyNode> &node2(*it2);
        const MultinomialModel *model2(rhs.conditional_model(node2.get()));
        conditional_model(node1.get())->combine_data(*model2);
      }
    }

    if (!just_suf) {
      std::copy(rhs.data_.begin(), rhs.data_.end(), std::back_inserter(data_));
    }
    
  }
  
  const MultinomialModel *MultilevelMultinomialModel::conditional_model(
      const TaxonomyNode *node) const {
    auto it = conditional_models_.find(node);
    if (it == conditional_models_.end()) {
      report_error("Could not find model.");
    }
    return it->second.get();
  }

  MultinomialModel *MultilevelMultinomialModel::conditional_model(
      const TaxonomyNode *node) {
    return conditional_models_[node].get();
  }
  
  void MultilevelMultinomialModel::create_models() {
    top_level_model_.reset(new MultinomialModel(taxonomy_->top_level_size()));

    for (auto it = taxonomy_->begin(); it != taxonomy_->end(); ++it) {
      const TaxonomyNode *node((*it).get());
      if (!node->is_leaf()) {
        conditional_models_[node] = new MultinomialModel(
            node->number_of_children());
      }
    }

  }
}
