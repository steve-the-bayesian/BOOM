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

#include "Models/StateSpace/StateModels/DynamicRegressionArStateModel.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    typedef DynamicRegressionArStateModel DRASM;
  }  // namespace

  DRASM::DynamicRegressionArStateModel(const Matrix &predictors, int lags)
      : DynamicRegressionArStateModel(split_predictors(predictors), lags) {}

  DRASM::DynamicRegressionArStateModel(const std::vector<Matrix> &predictors,
                                       int lags)
      : transition_matrix_(new BlockDiagonalMatrixBlock),
        state_error_expander_(new StackedMatrixBlock),
        state_error_variance_(new DiagonalMatrixParamView),
        state_variance_matrix_(new SparseDiagonalMatrixBlockParamView(
            compute_state_dimension(predictors, lags))),
        initial_state_mean_(compute_state_dimension(predictors, lags), 0.0),
        initial_state_variance_(compute_state_dimension(predictors, lags),
                                1.0) {
    if (lags < 1) {
      report_error("An AR model must have a lag of at least 1.");
    }
    int xdim = ncol(predictors[0]);
    if (xdim < 1) {
      report_error("Dynamic regression model has no data.");
    }

    // Set up the models.  This needs to be done before calling
    // add_to_predictors.
    for (int i = 0; i < xdim; ++i) {
      add_model(new ArModel(lags), xdim);
    }

    // Set up the predictors.
    add_to_predictors(predictors);

    xnames_.reserve(xdim);
    for (int i = 0; i < xdim; ++i) {
      std::ostringstream xname;
      xname << "X." << i + 1;
      xnames_.push_back(xname.str());
    }
  }

  DRASM::DynamicRegressionArStateModel(const DRASM &rhs) { operator=(rhs); }

  DRASM &DRASM::operator=(const DRASM &rhs) {
    if (&rhs != this) {
      coefficient_transition_model_.clear();
      transition_components_.clear();
      expanded_predictors_.clear();
      for (int i = 0; i < rhs.expanded_predictors_.size(); ++i) {
        expanded_predictors_.push_back(rhs.expanded_predictors_[i]->clone());
      }
      transition_matrix_.reset(new BlockDiagonalMatrixBlock);
      state_error_expander_.reset(new StackedMatrixBlock);
      state_error_variance_.reset(new DiagonalMatrixParamView);
      state_variance_matrix_.reset(new SparseDiagonalMatrixBlockParamView(
          expanded_predictors_[0]->ncol()));
      int xdim = rhs.coefficient_transition_model_.size();
      for (int i = 0; i < xdim; ++i) {
        add_model(rhs.coefficient_transition_model_[i]->clone(), xdim);
      }
      initial_state_mean_ = rhs.initial_state_mean_;
      initial_state_variance_ = rhs.initial_state_variance_;
    }
    return *this;
  }

  void DRASM::clear_data() {
    for (int i = 0; i < coefficient_transition_model_.size(); ++i) {
      coefficient_transition_model_[i]->clear_data();
    }
  }

  void DRASM::observe_state(const ConstVectorView &then,
                            const ConstVectorView &now, int time_now) {
    int pos = 0;
    for (int i = 0; i < xdim(); ++i) {
      double y = now[pos];
      ConstVectorView x(then, pos, number_of_lags());
      // TODO: There is an implicit conversion from ConstVectorView
      // to Vector in this function call.  Run this in a profiler, and see if
      // the Vector allocation is a bottleneck.
      coefficient_transition_model_[i]->suf()->add_mixture_data(y, x, 1.0);
      pos += number_of_lags();
    }
  }

  void DRASM::observe_initial_state(const ConstVectorView &state) {
    // Nothing to do here.
  }

  void DRASM::update_complete_data_sufficient_statistics(
      int t, const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    report_error(
        "MAP estimation is not supported for DynamicRegressionAr"
        "state models.");
  }

  void DRASM::simulate_state_error(RNG &rng, VectorView eta, int t) const {
    assert(eta.size() == state_dimension());
    int position = 0;
    for (int i = 0; i < coefficient_transition_model_.size(); ++i) {
      eta[position++] =
          rnorm_mt(rng, 0, coefficient_transition_model_[i]->sigma());
      for (int j = 1; j < number_of_lags(); ++j) {
        eta[position++] = 0;
      }
    }
  }

  Vector DRASM::initial_state_mean() const { return initial_state_mean_; }

  void DRASM::set_initial_state_mean(const Vector &mu) {
    check_state_dimension_size(mu.size());
    initial_state_mean_ = mu;
  }

  SpdMatrix DRASM::initial_state_variance() const {
    return initial_state_variance_;
  }

  void DRASM::set_initial_state_variance(const SpdMatrix &sigma) {
    check_state_dimension_size(sigma.nrow());
    initial_state_variance_ = sigma;
  }

  void DRASM::add_forecast_data(const Matrix &predictors) {
    add_multiplexed_forecast_data(split_predictors(predictors));
  }

  void DRASM::add_multiplexed_forecast_data(
      const std::vector<Matrix> &predictors) {
    add_to_predictors(predictors);
  }

  void DRASM::increment_expected_gradient(
      VectorView gradient, int t, const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    // TODO:  implement this one day.
    report_error(
        "MAP estimation is not supported for dynamic regression "
        "AR(p) state models.");
  }

  void DRASM::set_xnames(const std::vector<std::string> &names) {
    if (names.size() != xdim()) {
      std::ostringstream err;
      err << "set_xnames was called with a vector of " << names.size()
          << " elements, but there are " << xdim()
          << " predictors in the model.";
      report_error(err.str());
    }
    xnames_ = names;
  }

  Matrix DRASM::predictors() const {
    int number_of_time_points = expanded_predictors_.size();
    int number_of_observations = 0;
    for (int t = 0; t < number_of_time_points; ++t) {
      number_of_observations += expanded_predictors_[t]->nrow();
    }
    Matrix ans(number_of_observations, xdim());
    int row = 0;
    for (int t = 0; t < number_of_time_points; ++t) {
      for (int r = 0; r < expanded_predictors_[t]->nrow(); ++r) {
        state_error_expander_->Tmult(ans.row(row++),
                                     expanded_predictors_[t]->row(r).dense());
      }
    }
    return ans;
  }

  //---------------------------------------------------------------------------
  // Private methods implemented below this line.

  void DRASM::add_to_predictors(const std::vector<Matrix> &predictors) {
    if (predictors.empty()) {
      report_error("Empty predictor set.");
    }
    int xdim = predictors[0].ncol();
    for (int t = 0; t < predictors.size(); ++t) {
      NEW(GenericSparseMatrixBlock, predictor_matrix)
      (predictors[t].nrow(), xdim * number_of_lags());
      for (int i = 0; i < predictors[t].nrow(); ++i) {
        predictor_matrix->set_row(expand_predictor(predictors[t].row(i)), i);
      }
      if (!expanded_predictors_.empty() &&
          (expanded_predictors_[0]->ncol() != predictor_matrix->ncol())) {
        report_error("All predictors must be the same dimension.");
      }
      expanded_predictors_.push_back(predictor_matrix);
    }
  }

  std::vector<Matrix> DRASM::split_predictors(const Matrix &predictors) const {
    std::vector<Matrix> ans;
    ans.reserve(predictors.nrow());
    for (int i = 0; i < predictors.nrow(); ++i) {
      ans.push_back(Matrix(1, predictors.ncol(), predictors.row(i)));
    }
    return ans;
  }

  SparseVector DRASM::expand_predictor(const ConstVectorView &x) const {
    SparseVector ans(state_dimension());
    int pos = 0;
    int lags = coefficient_transition_model_[0]->number_of_lags();
    for (int i = 0; i < x.size(); ++i) {
      ans[pos] = x[i];
      pos += lags;
    }
    return ans;
  }

  void DRASM::add_model(ArModel *coefficient_model, int xdim) {
    coefficient_transition_model_.push_back(coefficient_model);
    transition_components_.push_back(new AutoRegressionTransitionMatrix(
        coefficient_transition_model_.back()->coef_prm()));
    transition_matrix_->add_block(transition_components_.back());
    int index = coefficient_transition_model_.size() - 1;
    state_error_expander_->add_block(new SingleElementInFirstRow(
        coefficient_model->number_of_lags(), xdim, index, 1.0));
    Ptr<UnivParams> residual_variance =
        coefficient_transition_model_.back()->Sigsq_prm();
    state_error_variance_->add_variance(residual_variance);
    state_variance_matrix_->add_element(
        residual_variance, coefficient_model->number_of_lags() * index);
  }

  void DRASM::check_state_dimension_size(int state_dimension) const {
    if (state_dimension != this->state_dimension()) {
      report_error("Size does not match state dimension.");
    }
  }

  
}  // namespace BOOM
