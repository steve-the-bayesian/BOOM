// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#include "Models/Hierarchical/HierarchicalGammaModel.hpp"
#include <vector>

namespace BOOM {
  HierarchicalGammaModel::HierarchicalGammaModel(
      const std::vector<int> &number_of_observations_per_group,
      const std::vector<double> &sum_of_observations_per_group,
      const std::vector<double> &sum_of_log_observations_per_group)
      : prior_for_mean_parameters_(new GammaModel(1, 1)),
        prior_for_shape_parameters_(new GammaModel(1, 1)) {
    int n = number_of_observations_per_group.size();
    initialize();
    if ((sum_of_observations_per_group.size() != n) ||
        (sum_of_log_observations_per_group.size() != n)) {
      report_error(
          "Vectors must be the same size in HierarchicalGammaModel "
          "constructor.");
    }
    data_models_.reserve(n);
    for (int i = 0; i < n; ++i) {
      NEW(GammaModel, data_model)(1, 1);
      data_model->suf()->set(sum_of_observations_per_group[i],
                             sum_of_log_observations_per_group[i],
                             number_of_observations_per_group[i]);
      get_initial_parameter_estimates(data_model);
      add_data_level_model(data_model);
    }
  }

  HierarchicalGammaModel::HierarchicalGammaModel(
      const HierarchicalGammaModel &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        PriorPolicy(rhs),
        prior_for_mean_parameters_(rhs.prior_for_mean_parameters_->clone()),
        prior_for_shape_parameters_(rhs.prior_for_shape_parameters_->clone()) {
    initialize();
    for (int i = 0; i < rhs.data_models_.size(); ++i) {
      add_data_level_model(rhs.data_models_[i]->clone());
    }
  }

  HierarchicalGammaModel *HierarchicalGammaModel::clone() const {
    return new HierarchicalGammaModel(*this);
  }

  void HierarchicalGammaModel::clear_methods() {
    PriorPolicy::clear_methods();
    prior_for_mean_parameters_->clear_methods();
    prior_for_shape_parameters_->clear_methods();
    for (int i = 0; i < data_models_.size(); ++i) {
      data_models_[i]->clear_methods();
    }
  }

  void HierarchicalGammaModel::clear_data() {
    data_models_.clear();
    ParamPolicy::clear();
    initialize();
  }

  void HierarchicalGammaModel::combine_data(const Model &rhs, bool just_suf) {
    const HierarchicalGammaModel &that(
        dynamic_cast<const HierarchicalGammaModel &>(rhs));
    for (int i = 0; i < that.number_of_groups(); ++i) {
      add_data_level_model(that.data_models_[i]);
    }
  }

  void HierarchicalGammaModel::add_data(const Ptr<Data> &dp) {
    NEW(GammaModel, data_model)(1, 1);
    Ptr<HierarchicalGammaData> d(dp.dcast<HierarchicalGammaData>());
    data_model->suf()->combine(d->suf());
    get_initial_parameter_estimates(data_model);
    add_data_level_model(data_model);
  }

  void HierarchicalGammaModel::add_data_level_model(
      const Ptr<GammaModel> &model) {
    data_models_.push_back(model);
    ParamPolicy::add_model(model);
  }

  int HierarchicalGammaModel::number_of_groups() const {
    return data_models_.size();
  }

  GammaModel *HierarchicalGammaModel::prior_for_mean_parameters() {
    return prior_for_mean_parameters_.get();
  }

  GammaModel *HierarchicalGammaModel::prior_for_shape_parameters() {
    return prior_for_shape_parameters_.get();
  }

  GammaModel *HierarchicalGammaModel::data_model(int group) {
    return data_models_[group].get();
  }

  double HierarchicalGammaModel::mean_parameter_prior_mean() const {
    return prior_for_mean_parameters_->mean();
  }

  double HierarchicalGammaModel::mean_parameter_prior_shape() const {
    return prior_for_mean_parameters_->alpha();
  }

  double HierarchicalGammaModel::shape_parameter_prior_mean() const {
    return prior_for_shape_parameters_->mean();
  }

  double HierarchicalGammaModel::shape_parameter_prior_shape() const {
    return prior_for_shape_parameters_->alpha();
  }

  void HierarchicalGammaModel::initialize() {
    ParamPolicy::add_model(prior_for_mean_parameters_);
    ParamPolicy::add_model(prior_for_shape_parameters_);
  }

  void HierarchicalGammaModel::get_initial_parameter_estimates(
      const Ptr<GammaModel> &data_model) const {
    try {
      data_model->mle();
    } catch (...) {
      double a = 1;
      double b = 1;
      Ptr<GammaSuf> suf(data_model->suf());
      if (suf->n() > 0) {
        double mean = suf->sum() / suf->n();
        b = 1.0 / mean;
      }
      data_model->set_shape_and_scale(a, b);
    }
  }

}  // namespace BOOM
