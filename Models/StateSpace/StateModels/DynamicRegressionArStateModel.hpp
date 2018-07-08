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

#ifndef BOOM_DYNAMIC_REGRESSION_AR_STATE_MODEL_HPP_
#define BOOM_DYNAMIC_REGRESSION_AR_STATE_MODEL_HPP_

#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include "Models/StateSpace/Filters/SparseVector.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/TimeSeries/ArModel.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  // A dynamic regression model where the regression coefficients evolve
  // according to an AR(p) model.
  //
  // Note that the state of this model is p times the dimension of beta, and
  // that high dimensional state models can slow the Kalman filter.
  //
  // The state of this model at time t is
  // beta0[t], beta0[t-1], ... beta0[t-p+1], beta1[t], beta1[t-1], ...,
  //
  // where beta0, beta1, beta2, ... are the coefficients of the dynamic
  // regression model.
  //
  // The observation matrix is x(t, 0), 0, 0, 0, ..., x(t, 1), 0, 0, 0, ...,
  // x(t, 2), ...
  class DynamicRegressionArStateModel : virtual public StateModel,
                                        public CompositeParamPolicy,
                                        public NullDataPolicy,
                                        public PriorPolicy {
   public:
    // A convenience constructor to use for the typical case of one observation
    // per time point.
    // Args:
    //   predictors:  The matrix of predictors.
    //   lag: The number of lags to consider in each coefficient's time series
    //     model.
    DynamicRegressionArStateModel(const Matrix &predictors, int lags);

    DynamicRegressionArStateModel(const DynamicRegressionArStateModel &rhs);
    DynamicRegressionArStateModel(DynamicRegressionArStateModel &&) = default;
    DynamicRegressionArStateModel *clone() const override {
      return new DynamicRegressionArStateModel(*this);
    }
    DynamicRegressionArStateModel &operator=(
        const DynamicRegressionArStateModel &rhs);
    DynamicRegressionArStateModel &operator=(DynamicRegressionArStateModel &&) =
        default;

    // The number of coefficients.
    int xdim() const { return coefficient_transition_model_.size(); }

    // The number of lags.
    int number_of_lags() const {
      return coefficient_transition_model_[0]->number_of_lags();
    }

    Ptr<ArModel> coefficient_model(int i) {
      return coefficient_transition_model_[i];
    }

    ArModel const *coefficient_model(int i) const {
      return coefficient_transition_model_[i].get();
    }

    void clear_data() override;

    void observe_state(const ConstVectorView &then, const ConstVectorView &now,
                       int time_now) override;

    void observe_initial_state(const ConstVectorView &state) override;

    uint state_dimension() const override { return transition_matrix_->nrow(); }

    uint state_error_dimension() const override { return xdim(); }

    void update_complete_data_sufficient_statistics(
        int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override {
      return transition_matrix_;
    }

    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override {
      return state_variance_matrix_;
    }

    Ptr<SparseMatrixBlock> state_error_expander(int t) const override {
      return state_error_expander_;
    }
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override {
      return state_error_variance_;
    }

    // The observation matrix is row t of the design matrix.
    SparseVector observation_matrix(int t) const override {
      if (t >= expanded_predictors_.size()) {
        report_error(
            "A DynamicRegressionArStateModel cannot be used outside "
            "the range of its predictor data.");
      }
      return expanded_predictors_[t]->row(0);
    }

    // The initial state is the value of the regression coefficients
    // at time 0.  Zero with a big variance is a good guess.
    Vector initial_state_mean() const override;
    void set_initial_state_mean(const Vector &mu);

    SpdMatrix initial_state_variance() const override;
    void set_initial_state_variance(const SpdMatrix &sigma);

    void add_forecast_data(const Matrix &predictors);
    void add_multiplexed_forecast_data(const std::vector<Matrix> &predictors);

    void increment_expected_gradient(
        VectorView gradient, int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    // The matrix of predictors used to initialize the model.
    Matrix predictors() const;

    void set_xnames(const std::vector<std::string> &names);
    const std::vector<std::string> &xnames() const { return xnames_; }

   protected:
    // For use with multiplexed data.
    // Args:
    //   predictors: Element t is the predictor matrix for time t.  Each row of
    //     predictors[t] is an observation at time t.
    //   lags: The number of lags to consider in each coefficient's time series
    //     model.
    DynamicRegressionArStateModel(const std::vector<Matrix> &predictors,
                                  int lags);

    Ptr<GenericSparseMatrixBlock> expanded_predictors(int t) const {
      return expanded_predictors_[t];
    }
    
   private:
    // Compute the state dimension from arguments passed to the constructor.
    int compute_state_dimension(const std::vector<Matrix> &predictors,
                                int lags) const {
      if (predictors.empty()) {
        report_error("Empty predictor vector.");
      }
      return ncol(predictors[0]) * lags;
    }

    // Expand the supplied predictors (by passing each through
    // expand_predictor()), and add them to the expanded_predictors_ data
    // element.
    //
    // Args:
    //   predictors: A sequence of predictor matrices, each representing a time
    //     point.  The number of rows in each matrix is the number of
    //     observations at that time point.  The matrix columns represent
    //     variables, and all matrices must have the same number of columns.
    void add_to_predictors(const std::vector<Matrix> &predictors);

    // Args:
    //   predictors: A predictor matrix (rows are observations, columns are
    //     variables).
    //
    // Returns:
    //   A vector of matrices.  Each row in the argument is a single-row matrix
    //   in the output.
    std::vector<Matrix> split_predictors(const Matrix &predictors) const;

    // Expand the elements of x by putting 'lags - 1' zeros after each element.
    // Thus the return value has size number_of_lags() * x.size().
    SparseVector expand_predictor(const ConstVectorView &x) const;

    // An implementation helper for the constructors.  Add the coefficient model
    // to the set of models for the coefficients, and update the structural
    // model matrices accordingly.
    // Args:
    //   coefficient_model:  The model to be added.
    //   xdim: The dimension of the (unexpanded) predictors.  This is the total
    //     number of models to be added.
    void add_model(ArModel *coefficient_model, int xdim);

    // Reports an error if the argument does not equal state_dimension();
    void check_state_dimension_size(int state_dimension) const;

    // Names of the predictor variables associated with each coefficient.
    std::vector<std::string> xnames_;

    // Each coefficient follows an AR model.
    std::vector<Ptr<ArModel>> coefficient_transition_model_;

    // The (transposed) state of the model is
    // beta[0, t], beta[0, t-1], beta[0, t-2], ... beta[1, t], beta[1, t-1], ...
    //
    // The transition matrix is a block diagonal matrix where each block is the
    // transition matrix for an AR model.
    //
    //    [ rho_1 rho_2 ... rho_p ]
    //    [     1     0         0 ]
    //    [     0     1         0 ]
    //    [     ...               ]
    //    [     0            1  0 ]
    Ptr<BlockDiagonalMatrixBlock> transition_matrix_;
    std::vector<Ptr<AutoRegressionTransitionMatrix>> transition_components_;

    // The observation matrix is Z[t] = Expand(x[t]), where 'expand' puts p-1
    // zeros between successive elements of x.  Element t of
    // expanded_predictors_ is the t'th matrix of predictors, expanded with p-1
    // columns of zeros between each 'real' predictor column.
    // x[t, 0] 0 0 0 ... x[t, 1] 0 0 0 ... x[t, 2] ...
    std::vector<Ptr<GenericSparseMatrixBlock>> expanded_predictors_;

    // The error expander matrix R_t =
    // [1 0 0 .... 0]
    // [0 0 0 .... 0]
    // ...
    // [0 1 0  ... 0]
    // [0 0 0 .... 0]
    // ...
    // [0 0 1 ...  0]
    // [0 0 0 .... 0]
    // ...
    // This matrix has k columns and p * k rows.
    // Where row p * k + j has all elements equal to zero, unless j == 0, in
    // which case element k = 1.
    Ptr<StackedMatrixBlock> state_error_expander_;

    // The state errors are independent of one another. Their variance matrix is
    // diagonal.  The diagonal elements are the residual variance parameters
    // from coefficient_transition_model_.
    Ptr<DiagonalMatrixParamView> state_error_variance_;

    // This is the same diagonal matrix as state_error_variance_, but with zeros
    // interspersed to account for the deterministic portions of state (the
    // lags).
    Ptr<SparseDiagonalMatrixBlockParamView> state_variance_matrix_;

    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;
  };

}  // namespace BOOM

#endif  //  BOOM_DYNAMIC_REGRESSION_AR_STATE_MODEL_HPP_
