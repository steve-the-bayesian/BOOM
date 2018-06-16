#ifndef BOOM_STATE_SPACE_MULTIVARIATE_KALMAN_FILTER_BASE_HPP_
#define BOOM_STATE_SPACE_MULTIVARIATE_KALMAN_FILTER_BASE_HPP_

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

#include "LinAlg/Vector.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "Models/StateSpace/Filters/KalmanFilterBase.hpp"

namespace BOOM {

  class MultivariateStateSpaceModelBase;

  namespace Kalman {
    // A base class to handle quantities common to all models with multivariate
    // outcomes.  Note that the dimension of the observed data need not be the
    // same at each time point.
    class MultivariateMarginalDistributionBase
        : public MarginalDistributionBase {
     public:
      MultivariateMarginalDistributionBase(int state_dim, int observation_dim)
          : MarginalDistributionBase(state_dim),
            prediction_error_(observation_dim),
            kalman_gain_(state_dim, observation_dim),
            r_(state_dim)
      {}

      const Vector &prediction_error() const {
        return prediction_error_;
      }
      void set_prediction_error(const Vector &err) {
        prediction_error_ = err;
      }

      // Inverse of Var(y[t] | Y[t-1]).
      virtual SpdMatrix forecast_precision() const = 0;

      // The log determinant of forecast_precision().
      virtual double forecast_precision_log_determinant() const = 0;

      // Returns forecast_precision() * prediction_error().  This is Finv * v in
      // Durbin and Koopman notation.
      virtual Vector scaled_prediction_error() const = 0;

      const Matrix &kalman_gain() const {return kalman_gain_;}
      void set_kalman_gain(const Matrix &gain) {kalman_gain_ = gain;}
      
      // Durbin and Koopman's r[t].  Recall that eta[t] is the error term for
      // moving from state t to state t+1.  The conditional mean of eta[t] given
      // all observed data is hat(eta[t]) = Q[t] * R[t]' * r[t].
      //
      // Where Q[t] is the error variance at time t, and R[t] is the error
      // expander.
      // 
      // In this equation R[t]' is a contractor (moving from the state dimension
      // to the error dimension).
      const Vector &scaled_state_error() const {
        return r_;
      }

      void set_scaled_state_error(const Vector &r) {
        r_  = r;
      }
      
     private:
      // y[t] - E(y[t] | Y[t-1]).  The dimension matches y[t], which might vary
      // across t.
      Vector prediction_error_;

      // The Kalman gain K[t] shows up in the updating equation:
      //       a[t+1] = T[t] * a[t] + K[t] * v[t].
      // Rows correspond to states and columns to observation elements, so the
      // dimension is S x m.
      Matrix kalman_gain_;

      // Computed from the Durbin-Koopman disturbance smoother.  DK do a poor
      // job of explaining what r_ is, but it is a scaled version of the state
      // error (see note above).  It is produced by smooth_disturbances_fast()
      // and used by propagate_disturbances().
      Vector r_;
    };
  }  // namespace Kalman

  //===========================================================================
  // A base class for handling the parts of the multivariate Kalman filter that
  // don't depend on the observation variance.
  class MultivariateKalmanFilterBase : public KalmanFilterBase {
   public:
    Vector fast_disturbance_smooth() override;
    Vector prediction_error(int t, bool standardize = false) const;

    Kalman::MultivariateMarginalDistributionBase & operator[](size_t pos)
        override = 0;
    const Kalman::MultivariateMarginalDistributionBase & operator[](size_t pos)
        const override = 0;
    
   protected:
    virtual Kalman::MultivariateMarginalDistributionBase &node(size_t t) = 0;
    virtual const Kalman::MultivariateMarginalDistributionBase &node(
        size_t t) const = 0;

   private:
    MultivariateStateSpaceModelBase *model_;
  };
  
}  // namespace BOOM

    


#endif  // BOOM_STATE_SPACE_MULTIVARIATE_KALMAN_FILTER_BASE_HPP_
