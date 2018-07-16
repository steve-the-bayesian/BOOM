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
#include "LinAlg/Cholesky.hpp"
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
      
      // Args:
      //   observation: The vector of observed data at time t.  Note that some
      //     elements of 'observation' might be missing.
      //   observed: Indicates which elements of 'observation' are actually
      //     observed, with observed[i] == true indicating that observation[i] is
      //     observed data.  If observed[i] == false then observation[i] should be
      //     viewed as meaningless.
      // Returns: 
      //   The contribution that observation makes to the log likelihood.
      double update(const Vector &y, const Selector &observed) override;

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
      
      // A Kalman filter update when the vector y is entirely missing.
      double fully_missing_update();
      
      // The prediction error is y[t] - E(y[t] | Y[t-1]).  The scaled prediction
      // error is forecast_precision() * prediction_error().
      Vector scaled_prediction_error() const override {
        return scaled_prediction_error_;
      }
      void set_scaled_prediction_error(const Vector &err) {
        scaled_prediction_error_ = err;
      }

      SpdMatrix forecast_precision() const override;
      
      void set_forecast_precision_log_determinant(double logdet) {
        forecast_precision_log_determinant_ = logdet;
      }
      double forecast_precision_log_determinant() const {
        return forecast_precision_log_determinant_;
      }

      // This class uses dense matrix algebra if the number of observations in
      // this time period is less than some multiple times the dimension of the
      // state.  By default the multiple is 1, but it can be changed using this
      // function.
      static void set_high_dimensional_threshold_factor(double value) {
        high_dimensional_threshold_factor = value;
      }

      // Check whether the vector of observations qualifies as high dimensional,
      // which depends on both the dimension of the observation and the
      // dimension of the state.
      bool high_dimensional(const Selector &observed) const;
      
     private:
      // Compute prediction_error, scaled_prediction_error_,
      // forecast_precision_log_determinant_, and kalman gain using the dense
      // forecast variance matrix.
      void small_sample_update(const Vector &observation,
                               const Selector &observed,
                               const SparseKalmanMatrix &transition,
                               const SparseKalmanMatrix &observation_coefficients);
      // Compute prediction_error, scaled_prediction_error_,
      // forecast_precision_log_determinant_, and kalman gain _WITHOUT_
      // computing the dense forecast variance matrix.
      void large_sample_update(const Vector &observation,
                               const Selector &observed,
                               const SparseKalmanMatrix &transition,
                               const SparseKalmanMatrix &observation_coefficients);

      // Compute the forecast precision matrix using the definition.
      SpdMatrix direct_forecast_precision() const;

      // Compute the forecast precision matrix using the binomial inverse
      // theorem.
      SpdMatrix large_scale_forecast_precision() const;
      
      ModelType *model_;
      ConditionalIidMarginalDistribution *previous_;
      Vector scaled_prediction_error_;
      double forecast_precision_log_determinant_;

      static double high_dimensional_threshold_factor;
    };
   
  }  // namespace Kalman

  //===========================================================================
  class ConditionalIidKalmanFilter
      : public MultivariateKalmanFilterBase {
   public:
    using MarginalType = Kalman::ConditionalIidMarginalDistribution;
    using ModelType = MarginalType::ModelType;

    explicit ConditionalIidKalmanFilter(ModelType *model);
    
    MarginalType &operator[](size_t pos) override { return node(pos); }
    const MarginalType &operator[](size_t pos) const override {
      return node(pos);
    }

    // The number of time points in the model being filtered.
    int size() const override { return nodes_.size(); }

    // Ensure space for at least t marginal distributions.
    void ensure_size(int t) override;

    const MarginalType &back() const {
      return nodes_.back();
    }
    
   private:
    std::vector<Kalman::ConditionalIidMarginalDistribution> nodes_;

    MarginalType & node(size_t pos) override {
      if (pos >= nodes_.size()) {
        report_error("Asking for a node past the end.");
      }
      return nodes_[pos];
    }

    const MarginalType & node(size_t pos) const override {
      if (pos >= nodes_.size()) {
        report_error("Asking for a const node past the end.");
      }
      return nodes_[pos];
    }

    ModelType *model_;
  };
  
}  // namespace BOOM


#endif  //  BOOM_STATE_SPACE_CONDITIONAL_IID_KALMAN_FILTER_HPP_
