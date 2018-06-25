// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#include "Models/StateSpace/StateModels/RegressionStateModel.hpp"
#include "Models/StateSpace/DynamicInterceptRegression.hpp"

namespace BOOM {

  RegressionStateModel::RegressionStateModel(const Ptr<RegressionModel> &rm)
      : regression_(rm),
        transition_matrix_(new IdentityMatrix(1)),
        error_variance_(new ZeroMatrix(1)),
        state_error_expander_(new IdentityMatrix(1)),
        state_error_variance_(new ZeroMatrix(1)) {}

  // The copy constructor copies pointers to private data.  Only regression_
  // is controversial, as all the others are the same across all
  // classes.  They could easily be static members.
  RegressionStateModel::RegressionStateModel(const RegressionStateModel &rhs)
      : StateModel(rhs),
        regression_(rhs.regression_->clone()),
        transition_matrix_(rhs.transition_matrix_->clone()),
        error_variance_(rhs.error_variance_->clone()),
        state_error_expander_(rhs.state_error_expander_->clone()),
        state_error_variance_(rhs.state_error_variance_->clone()),
        predictors_(rhs.predictors_) {}

  RegressionStateModel *RegressionStateModel::clone() const {
    return new RegressionStateModel(*this);
  }

  void RegressionStateModel::clear_data() { regression_->suf()->clear(); }

  // This function is a no-op.  The responsibility for observing state
  // lies with the state space model that owns it.
  void RegressionStateModel::observe_state(const ConstVectorView &then,
                                           const ConstVectorView &now,
                                           int time_now) {
  }

  uint RegressionStateModel::state_dimension() const { return 1; }

  void RegressionStateModel::update_complete_data_sufficient_statistics(
      int t, const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    report_error("RegressionStateModel cannot be part of an EM algorithm.");
  }

  void RegressionStateModel::simulate_state_error(RNG &, VectorView eta,
                                                  int t) const {
    eta[0] = 0;
  }

  void RegressionStateModel::simulate_initial_state(RNG &,
                                                    VectorView eta) const {
    eta[0] = 1;
  }

  Ptr<SparseMatrixBlock> RegressionStateModel::state_transition_matrix(
      int t) const {
    return transition_matrix_;
  }

  Ptr<SparseMatrixBlock> RegressionStateModel::state_variance_matrix(
      int) const {
    return error_variance_;
  }

  Ptr<SparseMatrixBlock> RegressionStateModel::state_error_expander(int) const {
    return state_error_expander_;
  }

  Ptr<SparseMatrixBlock> RegressionStateModel::state_error_variance(int) const {
    return state_error_variance_;
  }

  SparseVector RegressionStateModel::observation_matrix(int t) const {
    ConstVectorView x(predictors_.empty()
                          ? ConstVectorView(regression_->dat()[t]->x())
                          : predictors_[t].row(0));
    SparseVector ans(1);
    ans[0] = regression_->predict(x);
    return ans;
  }

  Vector RegressionStateModel::initial_state_mean() const {
    return Vector(1, 1.0);
  }

  SpdMatrix RegressionStateModel::initial_state_variance() const {
    return SpdMatrix(1, 0.0);
  }

  void RegressionStateModel::add_predictor_data(
      const std::vector<Matrix> &predictors) {
    if (!regression_) {
      report_error("Set the regression model first, before adding data.");
    }
    predictors_.reserve(predictors_.size() + predictors.size());
    for (int i = 0; i < predictors.size(); ++i) {
      if (predictors[i].ncol() != regression_->xdim()) {
        report_error(
            "The number of columns in predictor matrix does not match "
            "the dimension of regression model.");
      }
      predictors_.push_back(predictors[i]);
    }
  }

  Ptr<SparseMatrixBlock>
  RegressionDynamicInterceptStateModel::observation_coefficients(
      int t, const StateSpace::TimeSeriesRegressionData &data_point) const {
    return new DenseMatrix(Matrix(
        data_point.sample_size(),
        1,
        regression()->coef().predict(data_point.predictors())));
  }
  
}  // namespace BOOM
