#ifndef BOOM_STATE_SPACE_SCALAR_STATE_MODEL_ADAPTER_HPP_
#define BOOM_STATE_SPACE_SCALAR_STATE_MODEL_ADAPTER_HPP_

/*
  Copyright (C) 2022 Steven L. Scott

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

#include "Models/ZeroMeanGaussianModel.hpp"
#include "Models/StateSpace/Multivariate/MultivariateStateSpaceModelBase.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

#include "Models/Glm/RegressionModel.hpp"

#include "Models/StateSpace/Multivariate/StateModels/SharedStateModel.hpp"

namespace BOOM {

  //===========================================================================
  // Adapts a collection of one or more StateModel objects (designed for use
  // with scalar time series) for use as a SharedStateModel.
  //
  // The model matrices that are specific to the state are all determined by the
  // base StateModel.  The observation coefficients are determined by a
  // collection of linear regressions, with one regression model assigned to
  // each element of the response vector.
  //
  // In notation, the observation equation is
  //   y[t, j] = beta[j] * [Z[t] * alpha[j]] + error[t, j].
  // and the state equation is
  //   alpha[t+1] = T[t] * alpha[t] + R[t] * innovation[t].
  class ScalarStateModelMultivariateAdapter
      : virtual public SharedStateModel,
        public CompositeParamPolicy,
        public NullDataPolicy,
        public PriorPolicy
  {
   public:
    // Args:
    //   number_of_factors: The number of independent random walks to use in
    //     this state model.  The number of factors is the state dimension.
    //   host:  The model in which this object is a component of state.
    //   nseries:  The number of observed time series being modeled.
    ScalarStateModelMultivariateAdapter(
        MultivariateStateSpaceModelBase *host, int nseries);
    ~ScalarStateModelMultivariateAdapter();
    ScalarStateModelMultivariateAdapter(
        const ScalarStateModelMultivariateAdapter &rhs);
    ScalarStateModelMultivariateAdapter(
        ScalarStateModelMultivariateAdapter &&rhs) = default;
    ScalarStateModelMultivariateAdapter &operator=(
        const ScalarStateModelMultivariateAdapter &rhs);
    ScalarStateModelMultivariateAdapter &operator=(
        ScalarStateModelMultivariateAdapter &&rhs) = default;
    ScalarStateModelMultivariateAdapter *clone() const override;

    void add_state(const Ptr<StateModel> &state) {
      component_models_.add_state(state);
    }

    void clear_data() override {
      component_models_.clear_data();
    }

    void observe_state(const ConstVectorView &then, const ConstVectorView &now,
                       int time_now) override;

    //----------------------------------------------------------------------
    // Sizes of things.  The state dimension and the state_error_dimension are
    // both just the number of factors.
    uint state_dimension() const override {
      return component_models_.state_dimension();}
    uint state_error_dimension() const override {
      return component_models_.state_error_dimension();}

    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;
    void simulate_initial_state(RNG &rng, VectorView eta) const override;

    //--------------------------------------------------------------------------
    // Model matrices.
    Ptr<SparseMatrixBlock> observation_coefficients(
        int t, const Selector &observed) const override;

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override {
      Ptr<SparseMatrixBlock> ans = component_models_.state_transition_matrix(t);
      return ans;
    }

    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override {
      return component_models_.state_variance_matrix(t);
    }

    // The state error expander matrix is an identity matrix of the same
    // dimension as the state_transition_matrix, so we just return that matrix.
    Ptr<SparseMatrixBlock> state_error_expander(int t) const override {
      return component_models_.state_error_expander(t);
    }

    // Because the error expander is the identity, the state variance matrix and
    // the state error variance are the same thing.
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override {
      return component_models_.state_error_variance(t);
    }

    //--------------------------------------------------------------------------
    // Initial state mean and variance.  These should have been set by the
    // component models.
    Vector initial_state_mean() const override;
    SpdMatrix initial_state_variance() const override;

    //--------------------------------------------------------------------------
    // Tools for working with the EM algorithm and numerical optimization.
    // These are not currently implemented.
    void update_complete_data_sufficient_statistics(
        int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    void increment_expected_gradient(
        VectorView gradient, int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    const Vector &regression_coefficients() const;

   private:
    // The observation coefficients that would be produced by the
    // component_models_ if they were being used in a scalar model.
    SparseVector component_observation_coefficients(int t) const;

    // Remove any parameter observers that were set by the constructor.
    void remove_observers();

    //-------------------------------------------------------------------------
    // Data section
    //-------------------------------------------------------------------------

    // The host is the model object in which *this is a state component.  The
    // host is needed for this model to properly implement observe_state,
    // because the coefficient models needs to subtract away the contributions
    // from other state models.
    MultivariateStateSpaceModelBase *host_;

    // The 1-d regression models that relate the state to the observed data.
    // Each model has only a single scalar coefficient.
    std::vector<Ptr<RegressionModel>> observation_models_;

    // The individual elements of state (e.g. local linear trend, seasonality,
    // etc).
    StateSpaceUtils::StateModelVector<StateModel> component_models_;

    //----------------------------------------------------------------------
    // This section contains workspace needed to implement
    // "observation_coefficients()".
    //----------------------------------------------------------------------

    // A flag indicating whether the contents of the regression_coefficients_
    // workspace vector currently reflect the contents of the coefficients held
    // in observation_models_.  This flag is set by observers, which are set by
    // the constructor.
    mutable bool regression_coefficients_current_;

    // A copy of the regression coefficients stored in observation_models_.
    mutable Vector regression_coefficients_;

    // The observation coefficients in the form expected by the multivariate
    // Kalman filter.
    mutable Ptr<DenseSparseRankOneMatrixBlock> observation_coefficients_;
  };


}  // namespace BOOM

#endif  //  BOOM_STATE_SPACE_SCALAR_STATE_MODEL_ADAPTER_HPP_
