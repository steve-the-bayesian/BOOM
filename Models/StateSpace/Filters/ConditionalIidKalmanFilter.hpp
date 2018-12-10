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
      
      explicit ConditionalIidMarginalDistribution(
          ConditionalIidMultivariateStateSpaceModelBase *model,
          ConditionalIidMarginalDistribution *previous,
          int time_index);
      
      ConditionalIidMarginalDistribution * previous() override {
        return previous_;
      }
      const ConditionalIidMarginalDistribution * previous() const override {
        return previous_;
      }

      // It would be preferable to return the exact type of model_ here, but
      // doing so requires a covariant return, which we can't have without
      // declaring the full model type.  That can't happen because it would
      // create a cycle in the include graph.
      const MultivariateStateSpaceModelBase *model() const override;
      
      SpdMatrix forecast_precision() const override;
      
      // This class uses dense matrix algebra if the number of observations in
      // this time period is less than some multiple times the dimension of the
      // state.  By default the multiple is 1, but it can be changed using this
      // function.
      static void set_high_dimensional_threshold_factor(double value) {
        high_dimensional_threshold_factor_ = value;
      }
      double high_dimensional_threshold_factor() const override {
        return high_dimensional_threshold_factor_;
      }
      
     private:
      // Compute prediction_error, scaled_prediction_error_,
      // forecast_precision_log_determinant_, and kalman gain using the dense
      // forecast variance matrix.
      void low_dimensional_update(
          const Vector &observation,
          const Selector &observed,
          const SparseKalmanMatrix &transition,
          const SparseKalmanMatrix &observation_coefficients) override;

      // Compute prediction_error, scaled_prediction_error_,
      // forecast_precision_log_determinant_, and kalman gain _WITHOUT_
      // computing the dense forecast variance matrix.
      void high_dimensional_update(
          const Vector &observation,
          const Selector &observed,
          const SparseKalmanMatrix &transition,
          const SparseKalmanMatrix &observation_coefficients) override;

      // Compute the forecast precision matrix using the definition.
      SpdMatrix direct_forecast_precision() const;

      // Compute the forecast precision matrix using the binomial inverse
      // theorem.
      SpdMatrix large_scale_forecast_precision() const;

      //---------------------------------------------------------------------------
      // Data section
      ModelType *model_;
      ConditionalIidMarginalDistribution *previous_;
      static double high_dimensional_threshold_factor_;
    };
   
  }  // namespace Kalman

  using ConditionalIidKalmanFilter =
      MultivariateKalmanFilter<Kalman::ConditionalIidMarginalDistribution>;
  
}  // namespace BOOM


#endif  //  BOOM_STATE_SPACE_CONDITIONAL_IID_KALMAN_FILTER_HPP_
