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
    namespace {
      using CIMD = ConditionalIidMarginalDistribution;
    }  // namespace 

    double CIMD::high_dimensional_threshold_factor_(1.0);
    
    CIMD::ConditionalIidMarginalDistribution(
        ConditionalIidMultivariateStateSpaceModelBase *model,
        CIMD *previous,
        int time_index)
        : MultivariateMarginalDistributionBase(
              model->state_dimension(), time_index),
          model_(model),
          previous_(previous)
    {}
    
    //---------------------------------------------------------------------------
    void CIMD::low_dimensional_update(
        const Vector &observation,
        const Selector &observed,
        const SparseKalmanMatrix &transition,
        const SparseKalmanMatrix &observation_coefficients) {
      set_prediction_error(
          observation - observation_coefficients * state_mean());
      SpdMatrix forecast_variance =
          DiagonalMatrix(observed.nvars(),
                         model_->observation_variance(time_index()))
          + observation_coefficients.sandwich(state_variance());
      SpdMatrix forecast_precision = forecast_variance.inv();
      set_forecast_precision_log_determinant(forecast_precision.logdet());
      set_scaled_prediction_error(forecast_precision * prediction_error());
      set_kalman_gain(transition * state_variance() *
                      observation_coefficients.Tmult(forecast_precision));
    }

    //---------------------------------------------------------------------------
    void CIMD::high_dimensional_update(
        const Vector &observation,
        const Selector &observed,
        const SparseKalmanMatrix &transition,
        const SparseKalmanMatrix &observation_coefficients) {
      double observation_variance = model_->observation_variance(time_index());

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
      //   Finv = Hinv - Hinv * Z * (I + P Z' Hinv Z).inv * P * Z' * Hinv
      //
      // We don't compute Finv directly, we compute Finv * prediction_error.
      SpdMatrix Z_inner = observation_coefficients.inner();
      Matrix inner_matrix = state_variance() * Z_inner / observation_variance;
      inner_matrix.diag() += 1.0;
      QR inner_qr(inner_matrix);
      
      Vector scaled_error =
          state_variance() * (observation_coefficients.Tmult(prediction_error()));
      scaled_error = inner_qr.solve(scaled_error);
      scaled_error = observation_coefficients * scaled_error;
      scaled_error /= observation_variance;
      scaled_error -= prediction_error();
      scaled_error /= -1 * observation_variance;
      set_scaled_prediction_error(scaled_error);
      
      // The log determinant of F.inverse is the negative log of det(H + ZPZ').
      // That determinant can be computed using the "matrix determinant lemma,"
      // which says det(A + UV') = det(I + V' * A.inv * U) * det(A)
      //
      // Let A = H, U = Z, V' = PZ'.  Then det(F) = det(I + PZ'Z / v) * det(H)
      set_forecast_precision_log_determinant(
          -1 * (inner_qr.logdet()
                + observed.nvars() * log(observation_variance)));

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
    }

    const MultivariateStateSpaceModelBase *CIMD::model() const {
      return model_;
    }

    SpdMatrix CIMD::forecast_precision() const {
      if (high_dimensional(model_->observed_status(time_index()))) {
        return large_scale_forecast_precision();
      }  else {
        return direct_forecast_precision();
      }
    }
    
    SpdMatrix CIMD::direct_forecast_precision() const {
      SpdMatrix ans = model_->observation_coefficients(
          time_index(), model_->observed_status(time_index()))->sandwich(
              previous()->state_variance());
      ans.diag() += model_->observation_variance(time_index());
      return ans.inv();
    }

    SpdMatrix CIMD::large_scale_forecast_precision() const {
      double observation_variance = model_->observation_variance(time_index());
      const SparseKalmanMatrix *observation_coefficients =
          model_->observation_coefficients(
              time_index(), model_->observed_status(time_index()));
      Matrix inner = previous()->state_variance() *
          observation_coefficients->inner() / observation_variance;
      inner.diag() += 1.0;

      SpdMatrix ans = observation_coefficients->sandwich(
          inner.solve(previous()->state_variance()));
      ans /= -square(observation_variance);
      ans.diag() += 1.0 / observation_variance;
      return ans;
    }
  }  // namespace Kalman
  
}  // namespace BOOM

