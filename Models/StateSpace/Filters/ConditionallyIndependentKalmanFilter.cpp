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
#include "LinAlg/Cholesky.hpp"
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
          filter_(filter),
          forecast_precision_log_determinant_(negative_infinity()),
          forecast_precision_inner_condition_number_(negative_infinity()),
          forecast_precision_implementation_(Woodbury)
    {}

    const MultivariateStateSpaceModelBase * Marginal::model() const {
      return model_;
    }

    //---------------------------------------------------------------------------
    Ptr<SparseBinomialInverse> Marginal::bi_sparse_forecast_precision() const {
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
          forecast_precision_log_determinant_,
          forecast_precision_inner_condition_number_);
    }

    // Return the inverse of F = H + ZPZ'.  Here H is a diagonal residual
    // variance matrix.  We decompose P = LL', so F =H + ZLL'Z'.  Note taht P
    // may not be full rank if there are deterministic components.  That's okay,
    // because the Cholesky algorihm can handle positive semidefinite matrices.
    Ptr<SparseWoodburyInverse>
    Marginal::woodbury_sparse_forecast_precision() const {

      if (forecast_precision_inner_matrix_.nrow() == 0
          || forecast_precision_inner_matrix_.ncol() == 0) {
        report_error("Error rebuilding woodbury matrix.  "
                     "inner_matrix must have positive dimension.");
      }

      SpdMatrix variance = previous() ? previous()->state_variance() :
          model_->initial_state_variance();
      Cholesky state_variance_chol(variance);
      Matrix lower_triangle = state_variance_chol.getL(false);

      const Selector &observed(model_->observed_status(time_index()));
      NEW(DiagonalMatrixBlock, observation_precision)(
          1.0 / model_->observation_variance(time_index(), observed).diag());
      Ptr<SparseKalmanMatrix> observation_coefficients =
          model_->observation_coefficients(time_index(), observed);
      NEW(SparseMatrixProduct, U)();
      U->add_term(observation_coefficients);
      U->add_term(new DenseMatrix(lower_triangle));

      return new SparseWoodburyInverse(
          observation_precision, U, forecast_precision_inner_matrix_,
          forecast_precision_log_determinant_,
          forecast_precision_inner_condition_number_);
    }

    Ptr<SparseKalmanMatrix> Marginal::sparse_forecast_precision() const {
      switch (forecast_precision_implementation_) {
        case Woodbury:
          return woodbury_sparse_forecast_precision();
          break;
        case BinomialInverse:
          return bi_sparse_forecast_precision();
          break;
        case Dense:
          return new DenseSpd(direct_forecast_precision());
          break;
        default:
          report_error("Unrecognized value of forecast_precision_implementation_");
          return new NullMatrix(0);
      }
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
    // Kalman filter.  Determine the strategy to be used when implementing
    // sparse_forecast_precision(), and precompute some relevant quantities.
    void Marginal::update_sparse_forecast_precision(
        const Selector &observed) {
      // Ensure the the 'state_variance' we're using is P[t] and not P[t+1].  In
      // the Kalman filter update step these are the same thing.  In smoothing
      // they are not the same.
      SpdMatrix variance = previous() ? previous()->state_variance() :
          model_->initial_state_variance();

      // Determine the strategy to use for implmementing the forecast precision
      // matrix.  The default, sure but slow, strategy is to use dense matrices.
      forecast_precision_implementation_ = ForecastPrecisionImplementation::Dense;

      int t = time_index();
      NEW(DiagonalMatrixBlock, observation_precision)(
          1.0 / model_->observation_variance(time_index(), observed).diag());
      Ptr<SparseKalmanMatrix> observation_coefficients =
          model_->observation_coefficients(t, observed);
      double sumlog_precision = 0;
      for (double scalar_precision : observation_precision->diagonal_elements()) {
        sumlog_precision += log(scalar_precision);
      }
      Cholesky state_variance_chol(variance);
      Matrix state_variance_lower_triangle(state_variance_chol.getL(false));
      NEW(SparseMatrixProduct, U)();
      U->add_term(observation_coefficients);
      U->add_term(new DenseMatrix(state_variance_lower_triangle));

      SparseWoodburyInverse woodbury_precision(
          observation_precision, sumlog_precision, U);

      double max_condition_number = 1e+8;

      if (woodbury_precision.inner_matrix_condition_number() <
          max_condition_number) {
        forecast_precision_inner_matrix_ = woodbury_precision.inner_matrix();
        forecast_precision_inner_condition_number_ =
            woodbury_precision.inner_matrix_condition_number();
        forecast_precision_log_determinant_ = woodbury_precision.logdet();
        forecast_precision_implementation_ = ForecastPrecisionImplementation::Woodbury;
      } else {
        SparseBinomialInverse forecast_precision(
            observation_precision,
            observation_coefficients,
            variance,
            sumlog_precision);
        if (forecast_precision.inner_matrix_condition_number() < max_condition_number) {
          forecast_precision_inner_matrix_ = forecast_precision.inner_matrix();
          forecast_precision_log_determinant_ = forecast_precision.logdet();
          forecast_precision_inner_condition_number_ =
              forecast_precision.inner_matrix_condition_number();
          forecast_precision_implementation_ =
              ForecastPrecisionImplementation::BinomialInverse;
        } else {
          forecast_precision_inner_matrix_ = SpdMatrix();
          forecast_precision_inner_condition_number_ = negative_infinity();
          SpdMatrix Finv = direct_forecast_precision();
          forecast_precision_log_determinant_ = Finv.logdet();
        }
      }

      if (!forecast_precision_inner_matrix_.all_finite()) {
        report_error("Some infinite values or nan's found when computing "
                     "sparse_forecast_precision.");
      }
    }

    //---------------------------------------------------------------------------
    SpdMatrix Marginal::direct_forecast_precision() const {
      // Ensure the the 'state_variance' we're using is P[t] and not P[t+1].
      SpdMatrix variance = previous() ? previous()->state_variance() :
          model_->initial_state_variance();
      const Selector &observed(model_->observed_status(time_index()));
      SpdMatrix ans = model_->observation_coefficients(
          time_index(), observed)->sandwich(variance);
      ans.diag() += model_->observation_variance(time_index(), observed).diag();
      return ans.inv();
    }

  }  // namespace Kalman
}  // namespace BOOM
