// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#ifndef BOOM_HIERARCHICAL_MODEL_HPP_
#define BOOM_HIERARCHICAL_MODEL_HPP_

#include <vector>
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  // A base class for handling common implementations of simple
  // hierarchical models.
  //   Template arguments:
  //     DATA_MODEL_TYPE: The lowest level model (closest to data)
  //       responsible for modeling the data within each group.
  //     PRIOR_TYPE: The model responsible for describing the
  //       parameters of the data_models across groups.
  template <class DATA_MODEL_TYPE, class PRIOR_TYPE>
  class HierarchicalModelBase : public CompositeParamPolicy,
                                public PriorPolicy {
   public:
    typedef HierarchicalModelBase<DATA_MODEL_TYPE, PRIOR_TYPE> HierarchicalBase;

    explicit HierarchicalModelBase(const Ptr<PRIOR_TYPE> &prior)
        : prior_(prior) {
      initialize_model_structure();
    }

    HierarchicalModelBase(const HierarchicalModelBase &rhs)
        : Model(rhs),
          ParamPolicy(rhs),
          PriorPolicy(rhs),
          prior_(rhs.prior_->clone()) {
      initialize_model_structure();
      for (int i = 0; i < rhs.data_level_models_.size(); ++i) {
        add_data_level_model(rhs.data_level_models_[i]->clone());
      }
    }

    HierarchicalModelBase(HierarchicalModelBase &&rhs) = default;

    HierarchicalModelBase *clone() const override = 0;

    void add_data_level_model(const Ptr<DATA_MODEL_TYPE> &data_model) {
      data_level_models_.push_back(data_model);
      ParamPolicy::add_model(data_model);
    }

    void clear_data() override {
      data_level_models_.clear();
      ParamPolicy::clear();
      initialize_model_structure();
      prior_->clear_data();
    }

    void clear_client_data() {
      prior_->clear_data();
      for (int i = 0; i < data_level_models_.size(); ++i) {
        data_level_models_[i]->clear_data();
      }
    }

    void clear_methods() override {
      prior_->clear_methods();
      for (int i = 0; i < data_level_models_.size(); ++i) {
        data_level_models_[i]->clear_methods();
      }
    }

    void combine_data(const Model &rhs, bool) override {
      const HierarchicalBase &rhs_model(
          dynamic_cast<const HierarchicalBase &>(rhs));
      for (int i = 0; i < rhs_model.number_of_groups(); ++i) {
        add_data_level_model(rhs_model.data_level_models_[i]);
      }
    }

    int number_of_groups() const { return data_level_models_.size(); }

    DATA_MODEL_TYPE *data_model(int which_group) {
      return data_level_models_[which_group].get();
    }
    const DATA_MODEL_TYPE *data_model(int which_group) const {
      return data_level_models_[which_group].get();
    }

    PRIOR_TYPE *prior_model() { return prior_.get(); }

    const PRIOR_TYPE *prior_model() const { return prior_.get(); }

   private:
    void initialize_model_structure() {
      ParamPolicy::add_model(prior_);
      for (int i = 0; i < data_level_models_.size(); ++i) {
        ParamPolicy::add_model(data_level_models_[i]);
      }
    }

    std::vector<Ptr<DATA_MODEL_TYPE> > data_level_models_;
    Ptr<PRIOR_TYPE> prior_;
  };

}  // namespace BOOM

#endif  //  BOOM_HIERARCHICAL_MODEL_HPP_
