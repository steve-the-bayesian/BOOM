#ifndef BOOM_STATE_SPACE_CONDITIONAL_IID_KALMAN_FILTER_HPP_
#define BOOM_STATE_SPACE_CONDITIONAL_IID_KALMAN_FILTER_HPP_

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
#include "LinAlg/Vector.hpp"
#include "LinAlg/Selector.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  class ConditionalIidMultivariateStateSpaceModelBase;

  namespace Kalman {
    // Marginal distribution for a multivariate state space model with
    // observation error variance equal to a constant times the identity matrix.
    // The constant need not be the same for all t, but it must be the same for
    // all observations at the same time point.
    class ConditionalIidMarginalDistribution
        : public MultivariateMarginalDistributionBase {
     public:
      using ModelType = ConditionalIidMultivariateStateSpaceModelBase;
      using MarginalType = ConditionalIidMarginalDistribution;
      using FilterType = MultivariateKalmanFilter<MarginalType>;

      explicit ConditionalIidMarginalDistribution(
          ConditionalIidMultivariateStateSpaceModelBase *model,
          FilterType *filter,
          int time_index);

      ConditionalIidMarginalDistribution * previous() override;
      const ConditionalIidMarginalDistribution * previous() const override;

      // It would be preferable to return the exact type of model_ here, but
      // doing so requires a covariant return, which we can't have without
      // declaring the full model type.  That can't happen because it would
      // create a cycle in the include graph.
      const MultivariateStateSpaceModelBase *model() const override;

      Ptr<SparseKalmanMatrix> sparse_forecast_precision() const override;
      double forecast_precision_log_determinant() const override;

     private:
      // Compute the forecast precision matrix using the definition.
      SpdMatrix direct_forecast_precision() const;

      void update_sparse_forecast_precision(const Selector &observed) override;

      Ptr<SparseBinomialInverse> bi_sparse_forecast_precision() const;
      Ptr<SparseWoodburyInverse> woodbury_sparse_forecast_precision() const;

      //---------------------------------------------------------------------------
      // Data section
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

  using ConditionalIidKalmanFilter =
      MultivariateKalmanFilter<Kalman::ConditionalIidMarginalDistribution>;

}  // namespace BOOM


#endif  //  BOOM_STATE_SPACE_CONDITIONAL_IID_KALMAN_FILTER_HPP_
