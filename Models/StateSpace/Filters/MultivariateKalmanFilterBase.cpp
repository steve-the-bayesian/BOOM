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
  //  namespace Kalman {
  //   }  // namespace Kalman

  void MultivariateKalmanFilterBase::set_model(
      MultivariateStateSpaceModelBase *model) {
    if (model_ != model) {
      model_ = model;
      if (model_) {
        observe_model_parameters(model_);
      }
    }
  }
  
  void MultivariateKalmanFilterBase::update() {
    if (!model_) {
      report_error("Model must be set before calling update().");
    }
    ensure_size(model_->time_dimension());
    clear();
    node(0).set_state_mean(model_->initial_state_mean());
    node(0).set_state_variance(model_->initial_state_variance());

    for (int t = 0; t < model_->time_dimension(); ++t) {
      if (t > 0) {
        node(t).set_state_mean(node(t - 1).state_mean());
        node(t).set_state_variance(node(t - 1).state_variance());
      }
      increment_log_likelihood(node(t).update(
          model_->observation(t), model_->observed_status(t), t));
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
    increment_log_likelihood(node(t).update(y, observed, t));
  }


  // Disturbance smoother replaces Durbin and Koopman's K[t] with r[t].  The
  // disturbance smoother is equation (5) in Durbin and Koopman (2002).
  //
  // Returns:
  //   Durbin and Koopman's r0.
  Vector MultivariateKalmanFilterBase::fast_disturbance_smooth() {
    if (!model_) {
      report_error("Model must be set before calling fast_disturbance_smooth().");
    }

    int n = model_->time_dimension();
    Vector r(model_->state_dimension(), 0.0);
    for (int t = n - 1; t >= 0; --t) {
      node(t).set_scaled_state_error(r);
      // Currently r is r[t].  This step of the loop turns it into r[t-1].

      // The disturbance smoother is defined by the following formula:
      // r[t-1] = Z[t] * Finv[t] * v[t] + (T[t]^T - Z[t]^T * K[t]^T)r[t]
      //        = T[t]^T * r
      //        + Z[t]^T * (Finv[t] * v[t] - K^T * r)

      // Dimensions:
      //   T: S x S
      //   K: S x m
      //   Z: m x S
      //   Finv: m x m
      //   v: m x 1
      //   r: S x 1
      
      // Some syntactic sugar makes the formulas easier to match up
      // with Durbin and Koopman.
      Vector scaled_prediction_error = node(t).scaled_prediction_error();
      Vector coefficient = scaled_prediction_error
          - (node(t).kalman_gain().Tmult(r));

      // Now produce r[t-1]
      r = model_->state_transition_matrix(t)->Tmult(r)
          + model_->observation_coefficients(t)->Tmult(coefficient);
    }
    return r;
  }

}  // namespace BOOM
