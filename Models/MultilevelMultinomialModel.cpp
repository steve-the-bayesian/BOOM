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


  MultilevelMultinomialModel::MultilevelMultinomialModel(const Ptr<Taxonomy> &tax)
      : taxonomy_(tax)
  {
    create_models();
  }

  MultilevelMultinomialModel *MultilevelMultinomialModel::clone() const {
    return new MultilevelMultinomialModel(*this);
  }

  int MultilevelMultinomialModel::number_of_observations() const {
    report_error("NYI");
    return -1;
  }

  double MultilevelMultinomialModel::logp(const MultilevelCategoricalData &data_point) const {
    report_error("NYI");
    return negative_infinity();
  }

  double MultilevelMultinomialModel::pdf(const Data *dp, bool logscale) const {
    report_error("NYI");
    return negative_infinity();
  }

  void MultilevelMultinomialModel::create_models() {
    top_level_model_.reset(new MultinomialModel(taxonomy_->top_level_size()));

    for (auto it = taxonomy_->begin(); it != taxonomy_->end(); ++it) {
      const Ptr<TaxonomyNode> &node(*it);
      if (!node->is_leaf()) {
        conditional_models_[node] = new MultinomialModel(node->number_of_children());
      }
    }

  }
}
