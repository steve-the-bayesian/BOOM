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

#ifndef BOOM_REGRESSION_STATE_MODEL_HPP_
#define BOOM_REGRESSION_STATE_MODEL_HPP_

#include "Models/Glm/RegressionModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"

namespace BOOM {

  // A StateModel for a static (non-time-varying) regression component.  The
  // 'state' is a constant '1' with zero error, and a [1x1] identity matrix for
  // the state transition matrix.

  class RegressionStateModel : virtual public StateModel,
                               public CompositeParamPolicy,
                               public NullDataPolicy,
                               public PriorPolicy {
   public:
    explicit RegressionStateModel(const Ptr<RegressionModel> &rm);
    RegressionStateModel(const RegressionStateModel &rhs);
    RegressionStateModel(RegressionStateModel &&rhs) = default;
    RegressionStateModel *clone() const override;

    // clears sufficient statistics, but does not erase pointers to data.
    void clear_data() override;

    // 'observe_state' is a no-op for this class because the state
    // model needs too much information in order to make the necessary
    // observations.  A class that contains a RegressionStateModel
    // should update an externally held pointer to regression_ each time a
    // state vector is observed.
    void observe_state(const ConstVectorView &then, const ConstVectorView &now,
                       int time_now) override;

    uint state_dimension() const override;
    uint state_error_dimension() const override { return 1; }

    // Implementation throws, because this model cannot be part of an
    // EM algorithm.
    void update_complete_data_sufficient_statistics(
        int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;
    void simulate_initial_state(RNG &rng, VectorView eta) const override;

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_error_expander(int t) const override;
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override;

    // The single entry in the observation matrix is x[t] * beta, the predicted
    // value from the regression at time t.  Because the state is a constant
    // '1', this adds the regression effect to the observation at time t.
    SparseVector observation_matrix(int t) const override;

    Vector initial_state_mean() const override;
    SpdMatrix initial_state_variance() const override;

    RegressionModel *regression() { return regression_.get(); }
    const RegressionModel *regression() const { return regression_.get(); }

    void add_predictor_data(const std::vector<Matrix> &predictors);

   private:
    Ptr<RegressionModel> regression_;
    Ptr<IdentityMatrix> transition_matrix_;
    Ptr<ZeroMatrix> error_variance_;
    Ptr<IdentityMatrix> state_error_expander_;
    Ptr<ZeroMatrix> state_error_variance_;

    std::vector<Matrix> predictors_;
  };
  
  class RegressionDynamicInterceptStateModel
      : public RegressionStateModel,
        public DynamicInterceptStateModel {
   public:
    RegressionDynamicInterceptStateModel *clone() const override {
      return new RegressionDynamicInterceptStateModel(*this);
    }
    
    explicit RegressionDynamicInterceptStateModel(
        const Ptr<RegressionModel> &regression)
        : RegressionStateModel(regression) {}

    // The observation coefficients at time t form a column vector with elements
    // x[t, i] * beta.  Because the state is a scalar '1' this product adds the
    // regression effect at time 't' to each element of the observation.
    Ptr<SparseMatrixBlock> observation_coefficients(
        int t, const StateSpace::TimeSeriesRegressionData &data_point) const override;

    bool is_pure_function_of_time() const override {
      return false;
    }
  };
  
}  // namespace BOOM
#endif  // BOOM_REGRESSION_STATE_MODEL_HPP_
