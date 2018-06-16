#ifndef BOOM_STATE_SPACE_SCALAR_KALMAN_FILTER_HPP_
#define BOOM_STATE_SPACE_SCALAR_KALMAN_FILTER_HPP_

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

#include "Models/StateSpace/Filters/KalmanFilterBase.hpp"
#include "LinAlg/Vector.hpp"

namespace BOOM {
  class ScalarStateSpaceModelBase;
  namespace Kalman {
    // A marginal distribution for the case of univariate data.
    class ScalarMarginalDistribution
        : public MarginalDistributionBase {
     public:
      ScalarMarginalDistribution(int state_dimension)
          : MarginalDistributionBase(state_dimension), prediction_error_(0),
            prediction_variance_(0), kalman_gain_(state_dimension, 0) {}

      double update(double y,
                    bool missing,
                    int t,
                    const ScalarStateSpaceModelBase *model,
                    double observation_variance_scale_factor = 1.0);
      
      double prediction_error() const {return prediction_error_;}
      void set_prediction_error(double err) {prediction_error_ = err;}

      double prediction_variance() const {return prediction_variance_;}
      void set_prediction_variance(double var) {prediction_variance_ = var;}

      const Vector &kalman_gain() const {return kalman_gain_;}
      void set_kalman_gain(const Vector &gain) {kalman_gain_ = gain;}
      
     private:
      double prediction_error_;
      double prediction_variance_;
      Vector kalman_gain_;
    };
  }  // namespace Kalman

  // A Kalman filter for state space models with scalar outcomes.
  class ScalarKalmanFilter : public KalmanFilterBase {
   public:
    ScalarKalmanFilter(ScalarStateSpaceModelBase *model = nullptr);

    void set_model(ScalarStateSpaceModelBase *model);
    
    // Fun the full Kalman filter over all the data held by the model.
    void update() override;

    // Update the Kalman filter at time t given observation y, which might be
    // different than y[t] held by the model (e.g. when doing posterior
    // simulation).
    void update(double y, int t, bool missing = false);

    Vector fast_disturbance_smooth() override;

    // Return the one-step prediction error held by the filter at time t.  If
    // 'standardize' is true then divide the prediction error by the square
    // root of the prediction variance.
    double prediction_error(int t, bool standardize = false) const;

    Kalman::ScalarMarginalDistribution &operator[](size_t pos) override {
      return nodes_[pos];
    }

    const Kalman::ScalarMarginalDistribution &operator[](
        size_t pos) const override {
      return nodes_[pos];
    }
      
    const Kalman::ScalarMarginalDistribution &back() const;
      
   private:
    ScalarStateSpaceModelBase *model_;
    std::vector<Kalman::ScalarMarginalDistribution> nodes_;
  };
  
  
}  // namespace BOOM

#endif  //  BOOM_STATE_SPACE_SCALAR_KALMAN_FILTER_HPP_
