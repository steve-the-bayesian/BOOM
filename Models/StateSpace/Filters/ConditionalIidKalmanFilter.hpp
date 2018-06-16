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

namespace BOOM {

  class ConditionalIidMultivariateStateSpaceModelBase;
  
  namespace Kalman {
    // Marginal distribution representing the case where the observation error
    // variance is a constant times the identity matrix.  The constant need not
    // be the same for all t, but it must be the same for all observations at
    // the same time point.
    class ConditionalIidMarginalDistribution
        : public MultivariateMarginalDistributionBase {
     public:
      using ModelType = ConditionalIidMultivariateStateSpaceModelBase;
      
      ConditionalIidMarginalDistribution(int state_dimension);

      void set_model(ConditionalIidMultivariateStateSpaceModelBase *model) {
        model_ = model;
      }
      
      // Args:
      //   observation: The vector of observed data at time t.  Note that some
      //     elements of 'observation' might be missing.
      //   observed: Indicates which elements of 'observation' are actually
      //     observed, with observed[i] == true indicating that observation[i] is
      //     observed data.  If observed[i] == false then observation[i] should be
      //     viewed as meaningless.
      //   t:  The time index that this marginal distribution describes.
      //   model: The model responsible for supplying the model matrices defining
      //     the Kalman filter updates.
      //
      // Returns:
      //   The contribution that observation makes to the log likelihood.
      double update(const Vector &y, const Selector &observed, int t) override;

      // A Kalman filter update when the vector y is entirely missing.
      double fully_missing_update(int t);
      
      // Compute the forecast precision matrix: Var(y[t+1] | Y[t]).inverse().
      // The inputs required to make this computation are stored during a call
      // to update().  The actual matrix is potentially very large, so it is not
      // stored.  This function re-computes it.
      SpdMatrix forecast_precision() const override;

      // The log determinant fo the forecast precision matrix.  This is computed
      // and stored during a call to update().  This function is just an
      // accessor.
      double forecast_precision_log_determinant() const override {
        return forecast_precision_log_determinant_;
      }

      // The prediction error is y[t] - E(y[t] | Y[t-1]).  The scaled prediction
      // error is forecast_precision() * prediction_error().
      Vector scaled_prediction_error() const override;

     private:
      double forecast_precision_log_determinant_;
      double observation_variance_;
      int time_index_;
      ModelType *model_;

      // The Cholesky root of the state conditional variance, before updating.
      // After updating state_variance is Durbin and Koopman's P[t+1], while
      // this remains the Cholesky root of P[t].
      Chol root_state_conditional_variance_;
    };
   
  }  // namespace Kalman

  //===========================================================================
  class ConditionalIidKalmanFilter
      : public MultivariateKalmanFilterBase {
   public:
    using MarginalType = Kalman::ConditionalIidMarginalDistribution;
    using ModelType = MarginalType::ModelType;
    ConditionalIidKalmanFilter(ModelType *model = nullptr);
    void set_model(ModelType *model);
    
    MarginalType & operator[](size_t pos) override {return node(pos);}
    const MarginalType & operator[](size_t pos) const override {
      return node(pos);
    }

    void ensure_size(int t) override;
    
   private:
    std::vector<Kalman::ConditionalIidMarginalDistribution> nodes_;

    MarginalType & node(size_t pos) override { return nodes_[pos]; }
    const MarginalType & node(size_t pos) const override { return nodes_[pos]; }

    ModelType *model_;
    
  };
  
}  // namespace BOOM


#endif  //  BOOM_STATE_SPACE_CONDITIONAL_IID_KALMAN_FILTER_HPP_
