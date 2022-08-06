#ifndef BOOM_STATE_SPACE_CONDITIONALLY_INDEPENDENT_KALMAN_FILTER_HPP_
#define BOOM_STATE_SPACE_CONDITIONALLY_INDEPENDENT_KALMAN_FILTER_HPP_
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
#include "Models/StateSpace/Filters/SparseMatrix.hpp"

namespace BOOM {
  class ConditionallyIndependentMultivariateStateSpaceModelBase;

  namespace Kalman {

    // Models the marginal distribution of the state variables and related
    // quantities from the Kalman filter for multivariate data where the
    // variance matrix is diagonal.  (If the variance matrix is a scalar
    // constant times the identity then ConditionalIidMarginalDistribution
    // should be used instead).
    class ConditionallyIndependentMarginalDistribution
        : public MultivariateMarginalDistributionBase {
     public:
      using ModelType = ConditionallyIndependentMultivariateStateSpaceModelBase;
      using MarginalType = ConditionallyIndependentMarginalDistribution;
      using FilterType = MultivariateKalmanFilter<MarginalType>;

      // Args:
      //   model: The model describing the time series containing this
      //     marginal distribution.
      //   filter: The filter object managing this node.  Note that the model
      //     has two filters - the primary filter and the simulation filter.
      //   time_index: The index of the time period described by this marginal
      //     distribution.
      ConditionallyIndependentMarginalDistribution(
          ModelType *model,
          FilterType *filter,
          int time_index);

      // The precision matrix (inverse of the variance matrix) describing the
      // conditional distribution of the prediction error at this time point,
      // given all past data.  Be careful calling this function if the dimension
      // of Y is large.  The resulting matrix will be large^2.
      SpdMatrix direct_forecast_precision() const;

      Ptr<SparseKalmanMatrix> sparse_forecast_precision() const override;
      double forecast_precision_log_determinant() const override;

      // The marginal distribution describing the previous time point, or
      // nullptr if this is time point zero.
      MarginalType *previous() override;
      const MarginalType *previous() const override;

      // The model() method must be handled in the .cpp file, because we can't
      // know here that ModelType is derived from
      // MultivariateStateSpaceModelBase.
      const MultivariateStateSpaceModelBase *model() const override;

     private:
      // Called as part of the 'update' method in the base class.
      void update_sparse_forecast_precision(const Selector &observed) override;

      Ptr<SparseBinomialInverse> bi_sparse_forecast_precision() const;
      Ptr<SparseWoodburyInverse> woodbury_sparse_forecast_precision() const;

      ModelType *model_;
      FilterType *filter_;

      // Implementation details for sparse_forecast_precision().
      Matrix forecast_precision_inner_matrix_;
      double forecast_precision_log_determinant_;
      double forecast_precision_inner_condition_number_;
      enum ForecastPrecisionImplementation {BinomialInverse, Woodbury, Dense};
      ForecastPrecisionImplementation forecast_precision_implementation_;
    };
  }  // namespace Kalman

  using ConditionallyIndependentKalmanFilter = MultivariateKalmanFilter<
    Kalman::ConditionallyIndependentMarginalDistribution>;

}  // namespace BOOM

#endif //  BOOM_STATE_SPACE_CONDITIONALLY_INDEPENDENT_KALMAN_FILTER_HPP_
