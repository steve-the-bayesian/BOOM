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
      MultivariateMarginalDistributionBase(int state_dim)
          : MarginalDistributionBase(state_dim)
      {}

      const Vector &prediction_error() const {
        return prediction_error_;
      }
      void set_prediction_error(const Vector &err) {
        prediction_error_ = err;
      }

      virtual double update(const Vector &observation,
                            const Selector &observed,
                            int t) = 0;

      // Returns forecast_precision() * prediction_error().  This is Finv * v in
      // Durbin and Koopman notation.
      virtual Vector scaled_prediction_error() const = 0;

      const Matrix &kalman_gain() const {return kalman_gain_;}
      void set_kalman_gain(const Matrix &gain) {kalman_gain_ = gain;}
      
      
     private:
      // y[t] - E(y[t] | Y[t-1]).  The dimension matches y[t], which might vary
      // across t.
      Vector prediction_error_;

      // The Kalman gain K[t] shows up in the updating equation:
      //       a[t+1] = T[t] * a[t] + K[t] * v[t].
      // Rows correspond to states and columns to observation elements, so the
      // dimension is S x m.
      Matrix kalman_gain_;

    };
  }  // namespace Kalman

  //===========================================================================
  // A base class for handling the parts of the multivariate Kalman filter that
  // don't depend on the observation variance.
  class MultivariateKalmanFilterBase : public KalmanFilterBase {
   public:
    MultivariateKalmanFilterBase(MultivariateStateSpaceModelBase *model = nullptr) {
      set_model(model);
    }
    void set_model(MultivariateStateSpaceModelBase *model);
    
    void update() override;
    void update_single_observation(
        const Vector &observation, const Selector &observed, int t);
    
    void fast_disturbance_smooth() override;
    Vector prediction_error(int t, bool standardize = false) const;

    Kalman::MultivariateMarginalDistributionBase & operator[](size_t pos)
        override = 0;
    const Kalman::MultivariateMarginalDistributionBase & operator[](size_t pos)
        const override = 0;

    // Add nodes to the collection of marginal distributions until it is large
    // enough to hold t elements.
    virtual void ensure_size(int t) = 0;

    const MultivariateStateSpaceModelBase *model() const {
      return model_;
    }
    
   protected:
    virtual Kalman::MultivariateMarginalDistributionBase &node(size_t t) = 0;
    virtual const Kalman::MultivariateMarginalDistributionBase &node(
        size_t t) const = 0;

   private:
    MultivariateStateSpaceModelBase *model_;
  };
  
}  // namespace BOOM

    


#endif  // BOOM_STATE_SPACE_MULTIVARIATE_KALMAN_FILTER_BASE_HPP_
