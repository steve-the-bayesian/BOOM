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

#include "Models/StateSpace/Filters/ConditionallyIndependentKalmanFilter.hpp"
#include "Models/StateSpace/Multivariate/MultivariateStateSpaceModelBase.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
#include "LinAlg/LU.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {

  namespace Kalman {
    namespace {
      using Marginal = ConditionallyIndependentMarginalDistribution;
    } // namespace

    Marginal::ConditionallyIndependentMarginalDistribution(
        ModelType *model, FilterType *filter, int time_index)
        : MultivariateMarginalDistributionBase(
              model->state_dimension(), time_index),
          model_(model),
          filter_(filter)
    {}

    const MultivariateStateSpaceModelBase * Marginal::model() const {
      return model_;
    }

    //---------------------------------------------------------------------------
    Vector Marginal::scaled_prediction_error() const {
      return *sparse_forecast_precision() * prediction_error();
    }

    Ptr<SparseBinomialInverse> Marginal::sparse_forecast_precision() const {
      SpdMatrix variance = previous() ? previous()->state_variance() :
          model_->initial_state_variance();
      const Selector &observed(model_->observed_status(time_index()));
      NEW(DiagonalMatrixBlock, observation_precision)(
          1.0 / model_->observation_variance(time_index(), observed).diag());
      Ptr<SparseKalmanMatrix> observation_coefficients =
          model_->observation_coefficients(time_index(), observed);
      return new SparseBinomialInverse(
          observation_precision,
          observation_coefficients,
          variance,
          forecast_precision_inner_matrix_,
          forecast_precision_log_determinant_);
    }

    double Marginal::forecast_precision_log_determinant() const {
      return forecast_precision_log_determinant_;
    }

    Marginal *Marginal::previous() {
      if (time_index() == 0) {
        return nullptr;
      } else {
        return &((*filter_)[time_index() - 1]);
      }
    }

    const Marginal *Marginal::previous() const {
      if (time_index() == 0) {
        return nullptr;
      } else {
        return &((*filter_)[time_index() - 1]);
      }
    }

    // To be called by the base class during the forward update portion of the
    // Kalman filter.
    void Marginal::update_sparse_forecast_precision(
        const Selector &observed) {
      // Ensure the the 'state_variance' we're using is P[t] and not P[t+1].  In
      // the Kalman filter update step these are the same thing.  In smoothing
      // they are not the same.
      SpdMatrix variance = previous() ? previous()->state_variance() :
          model_->initial_state_variance();
      int t = time_index();
      NEW(DiagonalMatrixBlock, observation_precision)(
          1.0 / model_->observation_variance(time_index(), observed).diag());
      Ptr<SparseKalmanMatrix> observation_coefficients =
          model_->observation_coefficients(t, observed);
      double sumlog_precision = 0;
      for (double scalar_precision : observation_precision->diagonal_elements()) {
        sumlog_precision += log(scalar_precision);
      }
      SparseBinomialInverse forecast_precision(
          observation_precision,
          observation_coefficients,
          variance,
          sumlog_precision);
      forecast_precision_inner_matrix_ = forecast_precision.inner_matrix();
      forecast_precision_log_determinant_ = forecast_precision.logdet();
    }

    //---------------------------------------------------------------------------
    SpdMatrix Marginal::direct_forecast_precision() const {
      // Ensure the the 'state_variance' we're using is P[t] and not P[t+1].
      SpdMatrix variance = previous() ? previous()->state_variance() :
          model_->initial_state_variance();
      const Selector &observed(model_->observed_status(time_index()));
      SpdMatrix ans = model_->observation_coefficients(
          time_index(), observed)->sandwich(variance);
      ans.diag() += model_->observation_variance(time_index()).diag();
      return ans.inv();
    }

  }  // namespace Kalman
}  // namespace BOOM
