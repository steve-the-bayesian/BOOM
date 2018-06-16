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
#include "distributions.hpp"

namespace BOOM {
  namespace Kalman {

    // TODO: Test this function against the update function in the other
    // multivariate marginal distributions to make sure they give the same
    // answer.

    double ConditionalIidMarginalDistribution::update(
        const Vector &observation,
        const Selector &observed,
        int t,
        const ModelType *model) {
      if (observed.nvars() == 0) {
        return update(t, model);
      }
      root_state_conditional_variance_.decompose(state_variance());
      double observation_variance = model->observation_variance(t);
      const SparseKalmanMatrix &observation_coefficients(
          *model->observation_coefficients(t));

      Matrix dense_observation_coefficients =
          observed.select_rows(observation_coefficients.dense());
      SpdMatrix Z_inner = dense_observation_coefficients.inner();
      SpdMatrix Pinv = root_state_conditional_variance_.inv();
      
      //----------------------------------------------------------------------
      // Compute the forecast precision and its log determinant.
      
      // The forecast precision is determined by the Woodbury formula.  The
      // forecast variance is F = H + ZPZ', where H is a constant times the
      // identity and P is low dimensional.
      //
      // The Woodbury formula tells us that F.inverse is
      //
      // Finv =H.inv - H.inv * Z * (P.inv + Z' * H.inv * Z).inv * Z' * H.inv
      // Because H is a constant, this simplifies to.
      //
      // Hinv - Z * (P.inv + Z'Z / sigsq).inv * Z' / sigsq^2
      SpdMatrix forecast_precision(observed.nvars(), 1.0 / observation_variance);
      forecast_precision -= observation_coefficients.sandwich(
          (Pinv + Z_inner / observation_variance).inv())
          / square(observation_variance);
      
      // This function also computes the log determinant of F.inverse, which is the
      // negative log of det(H + ZPZ').  That determinant can be computed using the
      // "matrix determinant lemma,"  which states det(A + UWV') =
      // det(W) * det(A) * det(W.inv + V'*A.inv*U).
      //
      // Thus the determinant of Finv is the negative log of 
      //     det(H) * det(P) * det(P.inv + Z'HZ)
      //     = det(H) * det(P) * det(P.inv + Z'Z * sigsq)
      forecast_precision_log_determinant_ =
          -(observed.nvars()) * std::log(observation_variance)
          -root_state_conditional_variance_.logdet();
      SpdMatrix inner =  Pinv + Z_inner * observation_variance;
      forecast_precision_log_determinant_ -= inner.logdet();

      //----------------------------------------------------------------------
      // Compute the one-step prediction error and log likelihood contribution.
      const SparseKalmanMatrix  &transition(*model->state_transition_matrix(t));
      Matrix kalman_gain = (transition * state_variance()) *
          (dense_observation_coefficients.Tmult(forecast_precision));
      Vector observation_mean = observed.select(
          observation_coefficients * state_mean());
      Vector observed_data = observed.select(observation);
      double log_likelihood = dmvn(observation,
                                   observation_mean,
                                   forecast_precision,
                                   forecast_precision_log_determinant_,
                                   true);
      Vector forecast_error = observed_data - observation_mean;

      //----------------------------------------------------------------------
      // Update the state mean from a[t] = E(state_t | Y[t-1]) to a[t+1] =
      // E(state[t+1] | Y[t]).
      set_state_mean(transition * state_mean() + kalman_gain * forecast_error);

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
      Matrix TPZprime = (transition * state_variance()).multT(
          dense_observation_coefficients);

      // Step 1:  Set P = T * P * T.transpose()
      transition.sandwich_inplace(mutable_state_variance());

      // Step 2: 
      // Decrement P by T*P*Z.transpose()*K.transpose().  This step can be
      // skipped if y is missing, because K is zero.
      mutable_state_variance() -= TPZprime.multT(kalman_gain);

      // Step 3: P += RQR
      model->state_variance_matrix(t)->add_to(mutable_state_variance());
      mutable_state_variance().fix_near_symmetry();
      return log_likelihood;
    }  // update
  }  // namespace Kalman
}  // namespace BOOM

