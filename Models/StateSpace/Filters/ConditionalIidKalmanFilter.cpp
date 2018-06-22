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

#include "Models/StateSpace/MultivariateStateSpaceModelBase.hpp"
#include "Models/StateSpace/Filters/ConditionalIidKalmanFilter.hpp"
#include "Models/StateSpace/MultivariateStateSpaceModelBase.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
#include "LinAlg/QR.hpp"
#include "cpputil/Constants.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace Kalman {

    ConditionalIidMarginalDistribution::ConditionalIidMarginalDistribution(
        int state_dimension)
        : MultivariateMarginalDistributionBase(state_dimension) {}
    
    // TODO: Test this function against the update function in the other
    // multivariate marginal distributions to make sure they give the same
    // answer.

    double ConditionalIidMarginalDistribution::update(
        const Vector &observation,
        const Selector &observed,
        int t) {
      if (observed.nvars() == 0) {
        return fully_missing_update(t);
      }
      double observation_variance = model_->observation_variance(t);
      const SparseKalmanMatrix &observation_coefficients(
          *model_->observation_coefficients(t));
      Matrix dense_observation_coefficients =
          observed.select_rows(observation_coefficients.dense());
      // Make a new member function called partial_observation_coefficients

      Vector observation_mean = observed.select(
          observation_coefficients * state_mean());
      Vector observed_data = observed.select(observation);
      set_prediction_error(observed_data - observation_mean);

      // At this point the Kalman recursions compute the forecast precision Finv
      // and its log determinant.  However, we can get rid of the forecast
      // precision matrix, and replace it with the scaled error = Finv *
      // prediction_error.
      //
      // To evaluate the normal likelihood, we need the quadratic form:
      //   error * Finv * error == error.dot(scaled_error).
      // We also need the log determinant of Finv.
      //
      // The forecast_precision can be computed using the binomial inverse
      // theorem:
      //  (A + UBV).inv =
      //    A.inv - A.inv * U * (I + B * V * Ainv * U).inv * B * V * Ainv.
      //
      // When applied to F = H + Z P Z' the theorem gives
      //
      //   Hinv - Hinv * Z * (I + P Z' Hinv Z).inv * P * Z' * Hinv
      //
      // We don't compute this directly, we compute Finv * prediction_error.
      SpdMatrix Z_inner = dense_observation_coefficients.inner();
      Matrix inner_matrix = state_variance() * Z_inner / observation_variance;
      inner_matrix.diag() += 1.0;
      QR inner_qr(inner_matrix);
      
      Vector scaled_error =
          state_variance() * (observation_coefficients.Tmult(prediction_error()));
      scaled_error = inner_qr.solve(scaled_error);
      //      scaled_error = inner_matrix.solve(scaled_error);
      scaled_error = observation_coefficients * scaled_error;
      scaled_error /= observation_variance;
      scaled_error -= prediction_error();
      scaled_error /= -1 * observation_variance;
      set_scaled_prediction_error(scaled_error);

      // SpdMatrix forecast_variance =
      //     DiagonalMatrix(observed.nvars(), model_->observation_variance(t))
      //     + sandwich(dense_observation_coefficients, state_variance());
      // SpdMatrix forecast_precision = forecast_variance.inv();
      // if ((scaled_error - forecast_precision * prediction_error()).max_abs()
      //     > 1e-4) {
      //   report_error("bad scaled error");
      // }
      
      // The log determinant of F.inverse is the negative log of det(H + ZPZ').
      // That determinant can be computed using the "matrix determinant lemma,"
      // which says det(A + UV') = det(I + V' * A.inv * U) * det(A)
      //
      // Let A = H, U = Z, V' = PZ'.  Then det(F) = det(I + PZ'Z / v) * det(H)
      double forecast_precision_log_determinant =
          -1 * (inner_qr.logdet()
                + observed.nvars() * log(observation_variance));
      
      double log_likelihood = -.5 * observed.nvars() * Constants::log_root_2pi
          + .5 * forecast_precision_log_determinant - .5 * prediction_error().dot(scaled_error);

      // Compute the one-step prediction error and log likelihood contribution.
      const SparseKalmanMatrix &transition(*model_->state_transition_matrix(t));

      // Kalman gain = TPZ'Finv = 
      //
      // TPZ' * (Hinv - Hinv * Z * (I + P Z' Hinv Z).inv * P * Z' * Hinv) = 
      //
      // T * PZ'Hinv * (I - Z * (I + P Z' Hinv Z).inv * PZ'Hinv) = 
      Matrix PZprimeHinv = (observation_coefficients * state_variance()).transpose()
          / observation_variance;
      Matrix gain = -1 * (observation_coefficients * inner_qr.solve(PZprimeHinv));
      gain.diag() += 1.0;
      set_kalman_gain(transition * PZprimeHinv * gain);
      
      // Update the state mean from a[t] = E(state_t | Y[t-1]) to a[t+1] =
      // E(state[t+1] | Y[t]).
      set_state_mean(transition * state_mean() + kalman_gain() * prediction_error());

      // Update the state variance from P[t] = Var(state_t | Y[t-1]) to P[t+1] =
      // Var(state[t+1} | Y[t]).
      //
      // The update formula is
      //
      // P[t+1] = T[t] * P[t] * T[t]'
      //          - T[t] * P[t] * Z[t]' * K[t]'
      //          + R[t] * Q[t] * R[t]'
      //
      // Need to define TPZprime before modifying P (known here as
      // state_variance).
      Matrix TPZprime = (dense_observation_coefficients *
                         (transition * state_variance()).transpose()).transpose();

      // Step 1:  Set P = T * P * T.transpose()
      transition.sandwich_inplace(mutable_state_variance());

      // Step 2: 
      // Decrement P by T*P*Z.transpose()*K.transpose().  This step can be
      // skipped if y is missing, because K is zero.
      mutable_state_variance() -= TPZprime.multT(kalman_gain());

      // Step 3: P += RQR
      model_->state_variance_matrix(t)->add_to(
          mutable_state_variance());
      mutable_state_variance().fix_near_symmetry();
      return log_likelihood;
    }  // update

    //===========================================================================
    double ConditionalIidMarginalDistribution::fully_missing_update(int t) {
      // Compute the one-step prediction error and log likelihood contribution.
      const SparseKalmanMatrix  &transition(*model_->state_transition_matrix(t));
      double log_likelihood = 0;
      set_prediction_error(Vector(0));

      // Update the state mean from a[t] = E(state_t | Y[t-1]) to a[t+1] =
      // E(state[t+1] | Y[t]).
      set_state_mean(transition * state_mean());

      // Update the state variance from P[t] = Var(state_t | Y[t-1]) to P[t+1] =
      // Var(state[t+1} | Y[t]).
      //
      // The update formula is
      //
      // P[t+1] = T[t] * P[t] * T[t]' + R[t] * Q[t] * R[t]'
      //
      // Step 1:  Set P = T * P * T.transpose()
      transition.sandwich_inplace(mutable_state_variance());

      // Step 3: P += RQR
      model_->state_variance_matrix(t)->add_to(mutable_state_variance());
      mutable_state_variance().fix_near_symmetry();
      return log_likelihood;
    }  

  }  // namespace Kalman

  //===========================================================================
  ConditionalIidKalmanFilter::ConditionalIidKalmanFilter(ModelType *model) {
    set_model(model);
  }

  void ConditionalIidKalmanFilter::set_model(ModelType *model) {
    model_ = model;
    MultivariateKalmanFilterBase::set_model(model);
    for (int i = 0; i < nodes_.size(); ++i) {
      nodes_[i].set_model(model_);
    }
  }
  
  void ConditionalIidKalmanFilter::ensure_size(int t) {
    while(nodes_.size() <=  t) {
      nodes_.push_back(Kalman::ConditionalIidMarginalDistribution(
          model()->state_dimension()));
      nodes_.back().set_model(model_);
    }
  }
  
}  // namespace BOOM

