// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include "Models/Glm/HierarchicalPoissonRegression.hpp"

namespace BOOM {

  HierarchicalPoissonRegressionModel::HierarchicalPoissonRegressionModel(
      const Ptr<MvnModel> &prior)
      : data_parent_model_(prior) {
    data_parent_model_->only_keep_sufstats();
    ParamPolicy::add_model(data_parent_model_);
  }

  HierarchicalPoissonRegressionModel::HierarchicalPoissonRegressionModel(
      const HierarchicalPoissonRegressionModel &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        PriorPolicy(rhs),
        data_parent_model_(rhs.data_parent_model_->clone()) {
    for (int i = 0; i < rhs.data_level_models_.size(); ++i) {
      add_data_level_model(rhs.data_level_models_[i]->clone());
    }
    data_parent_model_->only_keep_sufstats();
    ParamPolicy::add_model(data_parent_model_);
  }

  HierarchicalPoissonRegressionModel *
  HierarchicalPoissonRegressionModel::clone() const {
    return new HierarchicalPoissonRegressionModel(*this);
  }

  void HierarchicalPoissonRegressionModel::add_data_level_model(
      const Ptr<PoissonRegressionModel> &data_model) {
    if (data_model->xdim() != data_parent_model_->dim()) {
      ostringstream err;
      err << "Error when adding data_level_model to "
          << "HierarchicalPoissonRegression." << endl
          << "Dimension of data_model is " << data_model->xdim() << "." << endl
          << "Dimension of prior is " << data_parent_model_->dim() << "."
          << endl;
      report_error(err.str());
    }
    data_level_models_.push_back(data_model);
    ParamPolicy::add_model(data_model);
  }

  void HierarchicalPoissonRegressionModel::clear_data() {
    for (int i = 0; i < data_level_models_.size(); ++i) {
      data_level_models_[i]->clear_data();
    }
  }

  void HierarchicalPoissonRegressionModel::combine_data(const Model &rhs,
                                                        bool just_suf) {
    report_error(
        "HierarchicalPoissonRegressionModel::combine_data:"
        "  not yet implemented");
  }

  void HierarchicalPoissonRegressionModel::add_data(const Ptr<Data> &dp) {
    report_error(
        "HierarchicalPoissonRegressionModel::add_data:"
        "  not yet implemented");
  }

  int HierarchicalPoissonRegressionModel::xdim() const {
    return data_parent_model_->dim();
  }

  int HierarchicalPoissonRegressionModel::number_of_groups() const {
    return data_level_models_.size();
  }

  PoissonRegressionModel *HierarchicalPoissonRegressionModel::data_model(
      int which_group) {
    return data_level_models_[which_group].get();
  }

  MvnModel *HierarchicalPoissonRegressionModel::data_parent_model() {
    return data_parent_model_.get();
  }

}  // namespace BOOM
