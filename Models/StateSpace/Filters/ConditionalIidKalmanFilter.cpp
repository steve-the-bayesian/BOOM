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

#include "Models/StateSpace/Multivariate/MultivariateStateSpaceModelBase.hpp"
#include "Models/StateSpace/Filters/ConditionalIidKalmanFilter.hpp"

#include "LinAlg/DiagonalMatrix.hpp"
#include "LinAlg/QR.hpp"
#include "cpputil/Constants.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace Kalman {
    namespace {
      using CIMD = ConditionalIidMarginalDistribution;
    }  // namespace

    CIMD::ConditionalIidMarginalDistribution(
        ConditionalIidMultivariateStateSpaceModelBase *model,
        FilterType *filter,
        int time_index)
        : MultivariateMarginalDistributionBase(
              model->state_dimension(), time_index),
          model_(model),
          filter_(filter)
    {}

    CIMD * CIMD::previous() {
      if (time_index() == 0) {
        return nullptr;
      } else {
        return &((*filter_)[time_index() - 1]);
      }
    }

    const CIMD * CIMD::previous() const {
      if (time_index() == 0) {
        return nullptr;
      } else {
        return &((*filter_)[time_index() - 1]);
      }
    }

    //---------------------------------------------------------------------------
    const MultivariateStateSpaceModelBase *CIMD::model() const {
      return model_;
    }

    double CIMD::forecast_precision_log_determinant() const {
      return forecast_precision_log_determinant_;
    }

    Ptr<SparseBinomialInverse> CIMD::sparse_forecast_precision() const {
      SpdMatrix variance = previous() ? previous()->state_variance() :
          model_->initial_state_variance();
      const Selector &observed(model_->observed_status(time_index()));
      NEW(ConstantMatrix, observation_precision)(
          observed.nvars(),
          1.0 / model_->observation_variance(time_index()));
      Ptr<SparseKalmanMatrix> observation_coefficients =
          model_->observation_coefficients(time_index(), observed);
      return new SparseBinomialInverse(
          observation_precision,
          observation_coefficients,
          variance,
          forecast_precision_inner_matrix_,
          forecast_precision_log_determinant_);
    }

    SpdMatrix CIMD::direct_forecast_precision() const {
      SpdMatrix ans = model_->observation_coefficients(
          time_index(), model_->observed_status(time_index()))->sandwich(
              previous()->state_variance());
      ans.diag() += model_->observation_variance(time_index());
      return ans.inv();
    }

    void CIMD::update_sparse_forecast_precision(const Selector &observed) {
      int t = time_index();
      SpdMatrix variance = previous() ? previous()->state_variance() :
          model_->initial_state_variance();
      NEW(ConstantMatrix, observation_precision)(
          observed.nvars(),
          1.0 / model_->observation_variance(t));
      Ptr<SparseKalmanMatrix> observation_coefficients =
          model_->observation_coefficients(t, observed);
      double sumlog_precision =
          observed.nvars() * log(observation_precision->value());
      SparseBinomialInverse forecast_precision(
          observation_precision,
          observation_coefficients,
          variance,
          sumlog_precision);
      forecast_precision_log_determinant_ = forecast_precision.logdet();
      forecast_precision_inner_matrix_ = forecast_precision.inner_matrix();
    }
  }  // namespace Kalman

}  // namespace BOOM
