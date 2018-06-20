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
      time_index_ = t;
      if (observed.nvars() == 0) {
        return fully_missing_update(time_index_);
      }
      root_state_conditional_variance_.decompose(state_variance());
      observation_variance_ = model_->observation_variance(time_index_);
      const SparseKalmanMatrix &observation_coefficients(
          *model_->observation_coefficients(time_index_));

      // Compute the forecast precision and its log determinant.
      //
      // TODO: replace forecast_precision using the formula from the binomial
      // inverse theorem:
      // Finv = H.inv - H.inv * Z * P * (P + P*Z'*H.inv*Z*P).inv * P * Z' * H.inv
      Matrix dense_observation_coefficients =
          observed.select_rows(observation_coefficients.dense());
      SpdMatrix forecast_variance =
          DiagonalMatrix(observed.nvars(), observation_variance_)
          + sandwich(dense_observation_coefficients, state_variance());
      SpdMatrix forecast_precision = forecast_variance.inv();
      
      // SpdMatrix forecast_precision(observed.nvars(), 1.0 / observation_variance_);
      // SpdMatrix increment =
      //     (state_variance() +
      //      sandwich(state_variance(), ZP.inner() / observation_variance_)).inv();
      // increment /= square(observation_variance_);
      // forecast_precision -= sandwich(ZP, increment);
      
      // The log determinant of F.inverse is the negative log of det(H + ZPZ').
      // That determinant can be computed using the "matrix determinant lemma,"
      // which says det(A + UV') = det(I + V' * A.inv * U) * det(A)
      //
      // Let L = chol(P)
      // Set U = Z * L and V' = U'
      //
      // Then  det(F) = det(I + L'Z' H.inv * ZL) * det(H)
      //--------------------------------------------------
      // Matrix ZL = observed.select_rows(
      //     observation_coefficients * root_state_conditional_variance_.getL());
      // SpdMatrix ZL_inner = ZL.inner();
      // ZL_inner /= observation_variance_;
      // ZL_inner.diag() += 1.0;
      // forecast_precision_log_determinant_ =
      //     -observed.nvars() * log(observation_variance_)
      //     -ZL_inner.logdet();

      forecast_precision_log_determinant_ = forecast_precision.logdet();
      
      // Compute the one-step prediction error and log likelihood contribution.
      const SparseKalmanMatrix &transition(
          *model_->state_transition_matrix(time_index_));

      // Kalman gain = TPZ'Finv
      set_kalman_gain(transition * state_variance()
                      * matTmult(dense_observation_coefficients, forecast_precision));
      Vector observation_mean = observed.select(
          observation_coefficients * state_mean());
      Vector observed_data = observed.select(observation);
      double log_likelihood = dmvn(observation,
                                   observation_mean,
                                   forecast_precision,
                                   forecast_precision_log_determinant_,
                                   true);
      set_prediction_error(observed_data - observation_mean);
      set_scaled_prediction_error(forecast_precision * prediction_error());

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
      model_->state_variance_matrix(time_index_)->add_to(
          mutable_state_variance());
      mutable_state_variance().fix_near_symmetry();
      return log_likelihood;
    }  // update

    //===========================================================================
    double ConditionalIidMarginalDistribution::fully_missing_update(int t) {
      time_index_ = t;
      root_state_conditional_variance_.decompose(SpdMatrix());
      observation_variance_ = model_->observation_variance(time_index_);

      forecast_precision_log_determinant_ = negative_infinity();

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

    //---------------------------------------------------------------------------
    SpdMatrix ConditionalIidMarginalDistribution::forecast_precision() const {
      SpdMatrix Pinv = root_state_conditional_variance_.inv();
      SpdMatrix Z_inner = model_->observed_status(time_index_).select_rows(
          model_->observation_coefficients(time_index_)->dense()).inner();
      SpdMatrix precision(model_->observed_status(time_index_).nvars(), 1.0 /
                          observation_variance_);
      precision -= model_->observation_coefficients(time_index_)->sandwich(
          (Pinv + Z_inner / observation_variance_).inv())
          / square(observation_variance_);
      return precision;
    }

  }  // namespace Kalman

  //===========================================================================
  ConditionalIidKalmanFilter::ConditionalIidKalmanFilter(ModelType *model) {
    set_model(model);
  }

  void ConditionalIidKalmanFilter::set_model(ModelType *model) {
    model_ = model;
    MultivariateKalmanFilterBase::set_model(model);
  }
  
  void ConditionalIidKalmanFilter::ensure_size(int t) {
    while(nodes_.size() <=  t) {
      nodes_.push_back(Kalman::ConditionalIidMarginalDistribution(
          model()->state_dimension()));
      nodes_.back().set_model(model_);
      nodes_.back().set_time_index(nodes_.size() - 1);
    }
  }
  
}  // namespace BOOM

