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

#ifndef BOOM_LOCAL_LINEAR_TREND_STATE_MODEL_HPP_
#define BOOM_LOCAL_LINEAR_TREND_STATE_MODEL_HPP_

#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/ZeroMeanMvnModel.hpp"

namespace BOOM {

  //  mu[t+1] = mu[t] + delta[t] + u[t]
  //  delta[t+1] = delta[t] + v[t]
  class LocalLinearTrendStateModel : public ZeroMeanMvnModel,
                                     virtual public StateModel {
   public:
    LocalLinearTrendStateModel();
    LocalLinearTrendStateModel(const LocalLinearTrendStateModel &rhs);
    LocalLinearTrendStateModel *clone() const override;

    void observe_state(const ConstVectorView &then, const ConstVectorView &now,
                       int time_now) override;

    uint state_dimension() const override { return 2; }
    uint state_error_dimension() const override { return 2; }

    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_error_expander(int t) const override;
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override;

    SparseVector observation_matrix(int t) const override;

    Vector initial_state_mean() const override;
    void set_initial_state_mean(const Vector &v);
    SpdMatrix initial_state_variance() const override;
    void set_initial_state_variance(const SpdMatrix &V);

    void update_complete_data_sufficient_statistics(
        int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;
    void increment_expected_gradient(
        VectorView gradient, int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

   private:
    void check_dim(const ConstVectorView &) const;

    SparseVector observation_matrix_;
    Ptr<LocalLinearTrendMatrix> state_transition_matrix_;
    Ptr<DenseSpdParamView> state_variance_matrix_;
    Ptr<IdentityMatrix> state_error_expander_;
    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;
  };

  //===========================================================================
  class LocalLinearTrendDynamicInterceptModel
      : public DynamicInterceptStateModel,
        public LocalLinearTrendStateModel {
   public:
    LocalLinearTrendDynamicInterceptModel *clone() const override;
    Ptr<SparseMatrixBlock> observation_coefficients(
        int t,
        const StateSpace::TimeSeriesRegressionData &data_point) const override;

    bool is_pure_function_of_time() const override { return true; }
  };

}  // namespace BOOM
#endif  // BOOM_LOCAL_LINEAR_TREND_STATE_MODEL_HPP_
