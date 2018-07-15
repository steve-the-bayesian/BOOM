/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "Models/StateSpace/Filters/MultivariateKalmanFilterBase.hpp"
#include "Models/StateSpace/MultivariateStateSpaceModelBase.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  using std::cout;
  using std::cerr;

  namespace Kalman {
    namespace {
      using Marginal = MultivariateMarginalDistributionBase;
    }
    
    Vector Marginal::contemporaneous_state_mean() const {
      if (!previous()) {
        return model()->initial_state_mean()
            + model()->initial_state_variance()
            * model()->observation_coefficients(0)->Tmult(scaled_state_error());
      }
      return previous()->state_mean()
          + previous()->state_variance()
          * model()->observation_coefficients(time_index())->Tmult(
              scaled_state_error());
    }

    SpdMatrix Marginal::contemporaneous_state_variance() const {
      SpdMatrix P = previous() ? model()->initial_state_variance()
          : previous()->state_variance();
      const SparseKalmanMatrix *observation_coefficients(
          model()->observation_coefficients(time_index()));
      return P - sandwich(
          P, observation_coefficients->sandwich_transpose(forecast_precision()));
    }
     
  }  // namespace Kalman

  MultivariateKalmanFilterBase::MultivariateKalmanFilterBase(
      MultivariateStateSpaceModelBase *model)
      : model_(model) {}
  
  void MultivariateKalmanFilterBase::update() {
    if (!model_) {
      report_error("Model must be set before calling update().");
    }
    clear();
    ensure_size(0);
    node(0).set_state_mean(model_->initial_state_mean());
    node(0).set_state_variance(model_->initial_state_variance());
    for (int t = 0; t < model_->time_dimension(); ++t) {
      if (t > 0) {
        ensure_size(t);
        node(t).set_state_mean(node(t - 1).state_mean());
        node(t).set_state_variance(node(t - 1).state_variance());
      }
      increment_log_likelihood(node(t).update(
          model_->observation(t), model_->observed_status(t)));
      if (!std::isfinite(log_likelihood())) {
        set_status(NOT_CURRENT);
        return;
      }
    }
    set_status(CURRENT);
  }

  void MultivariateKalmanFilterBase::update_single_observation(
      const Vector &y,
      const Selector &observed,
      int t) {
    if (!model_) {
      report_error("Model must be set before calling update().");
    }
    ensure_size(t);
    if (t == 0) {
      node(t).set_state_mean(model_->initial_state_mean());
      node(t).set_state_variance(model_->initial_state_variance());
    } else {
      node(t).set_state_mean(node(t - 1).state_mean());
      node(t).set_state_variance(node(t - 1).state_variance());
    }
    increment_log_likelihood(node(t).update(y, observed));
  }


  // Disturbance smoother replaces Durbin and Koopman's K[t] with r[t].  The
  // disturbance smoother is equation (5) in Durbin and Koopman (2002,
  // Biometrika).
  //
  // Returns:
  //   Durbin and Koopman's r0.  Saves r[t] in node(t).scaled_state_error().
  void MultivariateKalmanFilterBase::fast_disturbance_smooth() {
    if (!model_) {
      report_error("Model must be set before calling fast_disturbance_smooth().");
    }

    int n = model_->time_dimension();
    Vector r(model_->state_dimension(), 0.0);
    for (int t = n - 1; t >= 0; --t) {
      // Currently r is r[t].  This step of the loop turns it into r[t-1].
      // 
      // The disturbance smoother is defined by the following formula:
      // r[t-1] = Z^T * Finv * v   +   (T^T - Z^T * K^T) * r[t]
      //        = T^T * r[t]       -   Z^T * (K^T * r[t] - Finv * v)
      //
      // Note that Durbin and Koopman (2002) is missing the transpose on Z in
      // their equation (5).  The transpose is required to get the dimensions to
      // match.
      //
      // If we stored (Z' * K') that would only be SxS.  Maybe put some smarts
      // in depending on whether m or S is larger.
      //
      // K = TPZ'Finv
      // Z' K' = Z' Finv Z P T'
      //
      // Dimensions:
      //   T: S x S
      //   K: S x m
      //   Z: m x S
      //   Finv: m x m
      //   v: m x 1
      //   r: S x 1
      //
      node(t).set_scaled_state_error(r);
      r = model_->state_transition_matrix(t)->Tmult(r)
          - model_->observation_coefficients(t)->Tmult(
              node(t).kalman_gain().Tmult(r) - node(t).scaled_prediction_error());
    }
    set_initial_scaled_state_error(r);
  }

}  // namespace BOOM
