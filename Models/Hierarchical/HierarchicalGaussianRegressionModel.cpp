// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#include "Models/Hierarchical/HierarchicalGaussianRegressionModel.hpp"

namespace BOOM {
  namespace {
    typedef HierarchicalGaussianRegressionModel HGRM;
  }  // namespace

  HGRM::HierarchicalGaussianRegressionModel(const Ptr<MvnModel> &prior,
                                            const Ptr<UnivParams> &sigsq)
      : prior_(prior), residual_variance_(sigsq) {
    initialize_param_policy();
  }

  HGRM::HierarchicalGaussianRegressionModel(const HGRM &rhs)
      : prior_(rhs.prior_->clone()),
        residual_variance_(rhs.residual_variance_->clone()) {
    initialize_param_policy();
  }

  HGRM *HGRM::clone() const { return new HGRM(*this); }

  void HGRM::add_model(const Ptr<RegressionModel> &model) {
    if (!groups_.empty()) {
      if (model->xdim() != groups_[0]->xdim()) {
        report_error(
            "Different sized group models in "
            "HierarchicalGaussianRegressionModel.");
      }
    }
    // Set the shared residual variance parameter.
    model->set_params(model->coef_prm(), residual_variance_);
    ParamPolicy::add_params(model->coef_prm());
    prior_->add_data(Ptr<VectorData>(model->coef_prm()));
    groups_.push_back(model);
  }

  void HGRM::add_data(const Ptr<Data> &dp) {
    Ptr<RegSuf> suf = dp.dcast<RegSuf>();
    if (!suf) {
      report_error(
          "Wrong data type in "
          "HierarchicalGaussianRegressionModel::add_data");
    }
    add_data(suf);
  }

  void HGRM::add_data(const Ptr<RegSuf> &suf) {
    NEW(RegressionModel, model)(suf->size());
    model->set_suf(suf);
    add_model(model);
  }

  void HGRM::add_regression_data(const Ptr<RegressionData> &dp, int group) {
    groups_[group]->add_data(dp);
  }

  void HGRM::clear_data() {
    groups_.clear();
    prior_->clear_data();
    initialize_param_policy();
  }

  void HGRM::clear_data_keep_models() {
    for (int i = 0; i < groups_.size(); ++i) {
      groups_[i]->clear_data();
    }
    prior_->clear_data();
  }

  void HGRM::combine_data(const Model &rhs, bool) {
    const HierarchicalGaussianRegressionModel *other_model =
        dynamic_cast<const HierarchicalGaussianRegressionModel *>(&rhs);
    if (!other_model) {
      report_error(
          "Could not convert the argument of 'combine_data' to "
          "HierarchicalGaussianRegressionModel.");
    }
    for (int i = 0; i < other_model->groups_.size(); ++i) {
      add_data(Ptr<RegSuf>(other_model->groups_[i]->suf()->clone()));
    }
  }

  void HGRM::initialize_param_policy() {
    ParamPolicy::clear();
    ParamPolicy::add_model(prior_);
    ParamPolicy::add_params(residual_variance_);
  }

}  // namespace BOOM
