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
          filter_(filter),
          forecast_precision_log_determinant_(negative_infinity()),
          forecast_precision_inner_condition_number_(negative_infinity()),
          forecast_precision_implementation_(
              ForecastPrecisionImplementation::Woodbury)
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

    Ptr<SparseBinomialInverse> CIMD::bi_sparse_forecast_precision() const {
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
          forecast_precision_log_determinant_,
          forecast_precision_inner_condition_number_);
    }

    Ptr<SparseWoodburyInverse>
    CIMD::woodbury_sparse_forecast_precision() const {
      SpdMatrix variance = previous() ? previous()->state_variance() :
          model_->initial_state_variance();
      Cholesky state_variance_chol(variance);

      const Selector &observed(model_->observed_status(time_index()));
      NEW(ConstantMatrix, observation_precision)(
          observed.nvars(),
          1.0 / model_->observation_variance(time_index()));
      Ptr<SparseKalmanMatrix> observation_coefficients =
          model_->observation_coefficients(time_index(), observed);
      NEW(SparseMatrixProduct, U)();
      U->add_term(observation_coefficients);
      U->add_term(new DenseMatrix(state_variance_chol.getL(false)));

      return new SparseWoodburyInverse(
          observation_precision,
          U,
          forecast_precision_inner_matrix_,
          forecast_precision_log_determinant_,
          forecast_precision_inner_condition_number_);
    }

    Ptr<SparseKalmanMatrix> CIMD::sparse_forecast_precision() const {
      switch (forecast_precision_implementation_) {
        case BinomialInverse:
          return bi_sparse_forecast_precision();
          break;
        case Woodbury:
          return woodbury_sparse_forecast_precision();
          break;
        case Dense:
          return new DenseSpd(direct_forecast_precision());
          break;
        default:
          report_error("Unknown value of forecast_precision_implementation_");
          return new NullMatrix(1);
      }
    }

    SpdMatrix CIMD::direct_forecast_precision() const {
      SpdMatrix variance = previous() ? previous()->state_variance()
          : model_->initial_state_variance();
      SpdMatrix ans = model_->observation_coefficients(
          time_index(), model_->observed_status(time_index()))->sandwich(
              variance);
      ans.diag() += model_->observation_variance(time_index());
      return ans.inv();
    }

    void CIMD::update_sparse_forecast_precision(const Selector &observed) {
      int t = time_index();
      SpdMatrix variance = previous() ? previous()->state_variance() :
          model_->initial_state_variance();
      Cholesky state_variance_chol(variance);

      NEW(ConstantMatrix, observation_precision)(
          observed.nvars(),
          1.0 / model_->observation_variance(t));
      Ptr<SparseKalmanMatrix> observation_coefficients =
          model_->observation_coefficients(t, observed);
      double sumlog_precision =
          observed.nvars() * log(observation_precision->value());

      NEW(SparseMatrixProduct, U)();
      U->add_term(observation_coefficients);
      U->add_term(new DenseMatrix(state_variance_chol.getL(false)));

      SparseWoodburyInverse woodbury_precision(
          observation_precision, sumlog_precision, U);
      if (woodbury_precision.inner_matrix_condition_number() < 1e+8) {
        forecast_precision_implementation_ =
            ForecastPrecisionImplementation::Woodbury;
        forecast_precision_inner_matrix_ = woodbury_precision.inner_matrix();
        forecast_precision_log_determinant_ = woodbury_precision.logdet();
        forecast_precision_inner_condition_number_ =
            woodbury_precision.inner_matrix_condition_number();
      } else {
        SparseBinomialInverse bi_precision(
            observation_precision,
            observation_coefficients,
            variance,
            sumlog_precision);
        if (bi_precision.okay()) {
          forecast_precision_log_determinant_ = bi_precision.logdet();
          forecast_precision_inner_matrix_ = bi_precision.inner_matrix();
          forecast_precision_inner_condition_number_ =
              bi_precision.inner_matrix_condition_number();
          forecast_precision_implementation_ =
              ForecastPrecisionImplementation::BinomialInverse;
        } else {
          // You land here if both the Woodbury and BinomialInverse methods
          // fail.  Fall back to dense.
          forecast_precision_inner_matrix_ = SpdMatrix();
          SpdMatrix Finv = direct_forecast_precision();
          forecast_precision_log_determinant_ = Finv.logdet();
          forecast_precision_implementation_ =
              ForecastPrecisionImplementation::Dense;
        }
      }
    }

  }  // namespace Kalman

}  // namespace BOOM
