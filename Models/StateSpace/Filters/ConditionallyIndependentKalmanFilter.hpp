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

      // Args:
      //   model: The model describing the time series containing this
      //     marginal distribution.
      //   previous: The marginal distribution for the previous time period.
      //     Use nullptr if this is time period 0.
      //   time_index: The index of the time period described by this marginal
      //     distribution.
      ConditionallyIndependentMarginalDistribution(
          ModelType *model, MarginalType *previous, int time_index);

      // The precision matrix (inverse of the variance matrix) describing the
      // conditional distribution of the prediction error at this time point,
      // given all past data.
      SpdMatrix forecast_precision() const override;
      SpdMatrix direct_forecast_precision() const;
      
      // An observation is considered to be high dimensional if its dimension is
      // at least 'threshold' * state_dimension.  This function sets
      // 'threshold'.
      static void set_high_dimensional_threshold_factor(double threshold = 1.0) {
        high_dimensional_threshold_factor_ = threshold;
      }
      double high_dimensional_threshold_factor() const override {
        return high_dimensional_threshold_factor_;
      }
      
      // The marginal distribution describing the previous time point, or
      // nullptr if this is time point zero.
      MarginalType *previous() override {return previous_;}
      const MarginalType *previous() const override {return previous_;}

      // The model() method must be handled in the .cpp file, because we can't
      // know here that ModelType is derived from
      // MultivariateMarginalDistributionBase.
      const MultivariateStateSpaceModelBase *model() const override;

     private:
      void high_dimensional_update(
          const Vector &observation,
          const Selector &observed,
          const SparseKalmanMatrix &transition,
          const SparseKalmanMatrix &observation_coefficient_subset) override;

      void low_dimensional_update(
          const Vector &observation,
          const Selector &observed,
          const SparseKalmanMatrix &transition,
          const SparseKalmanMatrix &observation_coefficient_subset) override;

      ModelType *model_;
      MarginalType *previous_;

      static double high_dimensional_threshold_factor_;
    };
  }  // namespace Kalman

  using ConditionallyIndependentKalmanFilter = MultivariateKalmanFilter<
    Kalman::ConditionallyIndependentMarginalDistribution>;
  
}  // namespace BOOM

#endif //  BOOM_STATE_SPACE_CONDITIONALLY_INDEPENDENT_KALMAN_FILTER_HPP_
