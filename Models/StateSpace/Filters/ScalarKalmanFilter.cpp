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

#include "Models/StateSpace/Filters/ScalarKalmanFilter.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace Kalman {
    namespace {
      // Shorten the name.
      using Marginal = ScalarMarginalDistribution;
    }  // namespace

    Marginal::ScalarMarginalDistribution(
        const ScalarStateSpaceModelBase *model,
        ScalarKalmanFilter *filter,
        int time_index)
        : MarginalDistributionBase(model->state_dimension(), time_index),
          model_(model),
          filter_(filter),
          prediction_error_(0),
          prediction_variance_(0),
          kalman_gain_(model_->state_dimension(), 0) {}

    double Marginal::update(double y, bool missing, int t,
                            double observation_variance_scale_factor) {
      const SparseVector observation_coefficients = model_->observation_matrix(t);
      Vector PZ = state_variance() * observation_coefficients;

      prediction_variance_ =
          observation_coefficients.dot(PZ) +
          model_->observation_variance(t) * observation_variance_scale_factor;
      if (prediction_variance_ <= 0) {
        std::ostringstream err;
        err << "Found a zero (or negative) forecast variance!";
        report_error(err.str());
      }
      const SparseKalmanMatrix &state_transition_matrix(
          *model_->state_transition_matrix(t));
      Vector TPZ = state_transition_matrix * PZ;

      double loglike = 0;
      if (!missing) {
        kalman_gain_ = TPZ / prediction_variance_;
        double mu = observation_coefficients.dot(state_mean());
        prediction_error_ = y - mu;
        loglike = dnorm(y, mu, sqrt(prediction_variance_), true);
      } else {
        kalman_gain_ = 0.0;
        prediction_error_ = 0;
      }

      if (!missing) {
        set_state_mean(state_transition_matrix * state_mean()
                       + kalman_gain_ * prediction_error_);
      } else {
        set_state_mean(state_transition_matrix * state_mean());
      }

      state_transition_matrix.sandwich_inplace(mutable_state_variance());
      if (!missing) {
        mutable_state_variance().Matrix::add_outer(TPZ, kalman_gain_, -1);
      }
      model_->state_variance_matrix(t)->add_to(mutable_state_variance());
      mutable_state_variance().fix_near_symmetry();
      return loglike;
    }

    const Marginal *Marginal::previous() const {
      if (time_index() < 1) {
        return nullptr;
      } else {
        return &((*filter_)[time_index() - 1]);
      }
    }

    Marginal *Marginal::previous() {
      if (time_index() < 1) {
        return nullptr;
      } else {
        return &((*filter_)[time_index() - 1]);
      }
    }

    Vector Marginal::contemporaneous_state_mean() const {
      const Marginal *prev = previous();
      if (!prev) {
        // This marginal distribution is the initial distribution.
        return model_->initial_state_mean()
            + (model_->initial_state_variance() *
               model_->observation_matrix(time_index()))
            * prediction_error_ / prediction_variance_;
      } else {
        // a[t] + P[t] * Z[t]' Finv * v
        return prev->state_mean() +
            (prev->state_variance()
             * model_->observation_matrix(time_index()))
            * prediction_error_ / prediction_variance_;
      }
    }

    SpdMatrix Marginal::contemporaneous_state_variance() const {
      const Marginal *prev = previous();
      SpdMatrix P = prev ? prev->state_variance() :
          model_->initial_state_variance();
      SparseVector Z(model_->observation_matrix(time_index()));
      return P - (P * Z).outer() / prediction_variance_;
    }

  }  // namespace Kalman

  ScalarKalmanFilter::ScalarKalmanFilter(ScalarStateSpaceModelBase *model)
      : model_(model)
  {}

  void ScalarKalmanFilter::update() {
    if (!model_) {
      report_error("Model must be set before calling update().");
    }
    while (nodes_.size() <= model_->time_dimension()) {
      nodes_.push_back(Kalman::ScalarMarginalDistribution(
          model_, this, nodes_.size()));
    }
    clear_loglikelihood();
    if (nodes_.size() > 0) {
      nodes_[0].set_state_mean(model_->initial_state_mean());
      nodes_[0].set_state_variance(model_->initial_state_variance());
    }

    for (int t = 0; t < model_->time_dimension(); ++t) {
      if (t > 0) {
        nodes_[t].set_state_mean(nodes_[t-1].state_mean());
        nodes_[t].set_state_variance(nodes_[t-1].state_variance());
      }
      increment_log_likelihood(nodes_[t].update(
          model_->adjusted_observation(t),
          model_->is_missing_observation(t),
          t));
      if (!std::isfinite(log_likelihood())) {
        set_status(NOT_CURRENT);
        return;
      }
    }
    set_status(CURRENT);
  }

  // Disturbance smoother replaces Durbin and Koopman's K[t] with r[t].  The
  // disturbance smoother is equation (5) in Durbin and Koopman (2002).
  //
  // Returns:
  //   Durbin and Koopman's r0.
  void ScalarKalmanFilter::fast_disturbance_smooth() {
    if (!model_) {
      report_error("Model must be set before calling fast_disturbance_smooth().");
    }

    int n = model_->time_dimension();
    Vector r(model_->state_dimension(), 0.0);
    for (int t = n - 1; t >= 0; --t) {
      // Upon entry r is r[t].
      // On exit, r is r[t-1] and filter[t].K is r[t]

      // The disturbance smoother is defined by the following formula:
      // r[t-1] = Z[t] * v[t]/F[t] + (T[t]^T - Z[t] * K[t]^T)r[t]
      //        = T[t]^T * r + Z[t] * (v[t]/F[t] - K.dot(r))

      // Some syntactic sugar makes the formulas easier to match up
      // with Durbin and Koopman.
      double v = nodes_[t].prediction_error();
      double F = nodes_[t].prediction_variance();
      double coefficient = (v / F) - nodes_[t].kalman_gain().dot(r);

      // Now produce r[t-1]
      Vector rt_1 = model_->state_transition_matrix(t)->Tmult(r);
      model_->observation_matrix(t).add_this_to(rt_1, coefficient);
      nodes_[t].set_scaled_state_error(r);
      r = rt_1;
    }
    set_initial_scaled_state_error(r);
  }

  void ScalarKalmanFilter::update(double y, int t, bool missing) {
    if (!model_) {
      report_error("Model must be set before calling update().");
    }

    while (nodes_.size() <= t) {
      nodes_.push_back(Kalman::ScalarMarginalDistribution(
          model_, this, nodes_.size()));
    }
    if (t == 0) {
      nodes_[t].set_state_mean(model_->initial_state_mean());
      nodes_[t].set_state_variance(model_->initial_state_variance());
    } else {
      nodes_[t].set_state_mean(nodes_[t-1].state_mean());
      nodes_[t].set_state_variance(nodes_[t-1].state_variance());
    }
    increment_log_likelihood(nodes_[t].update(y, missing, t));
  }

  double ScalarKalmanFilter::prediction_error(int t, bool standardize) const {
    double ans = nodes_[t].prediction_error();
    if (standardize) {
      ans /= sqrt(nodes_[t].prediction_variance());
    }
    return ans;
  }

  const Kalman::ScalarMarginalDistribution &ScalarKalmanFilter::back() const {
    if (!model_) {
      report_error("Model must be set before calling back().");
    }
    int n = model_->time_dimension();
    if (n == 0) {
      report_error("Time dimension is zero.");
    }
    return nodes_[n - 1];
  }

}  // namespace BOOM
