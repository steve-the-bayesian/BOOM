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
      // A marginal distribution for a single time point in which scalar data
      // was (possibly) observed.
      //
      // Args:
      //   model:  The model describing the data for this marginal distribution.
      //   previous:  The marginal distribution for the preceding time point.
      //   time_index:  The time index associated with this time period.
      explicit ScalarMarginalDistribution(const ScalarStateSpaceModelBase *model,
                                          ScalarMarginalDistribution *previous,
                                          int time_index);

      // Update this marginal distribution to reflect the observed data at this
      // time point, and the marginal information from the preceding time point.
      //
      // Args:
      //    y:  The observed data point.
      //    missing: If true then the value of y is ignored, and the state is
      //      simply propagated forward.
      //    t:  The time index associated with this marginal distribution.
      //    model:
      double update(double y,
                    bool missing,
                    int t,
                    double observation_variance_scale_factor = 1.0);

      // After the call to update(), state_mean() and state_variance() refer to
      // the predictive mean and variance of the state at time_dimension() + 1
      // given data to time_dimension().
      //
      // contemporaneous_state_XXX refers to the moments at the current time,
      // given data to the current time.
      Vector contemporaneous_state_mean() const override;
      SpdMatrix contemporaneous_state_variance() const override;

      // The difference between the observed data at this time point, and the
      // expected value given previous data.
      double prediction_error() const {return prediction_error_;}
      void set_prediction_error(double err) {prediction_error_ = err;}

      // The variance of prediction_error() conditional on previous data.
      double prediction_variance() const {return prediction_variance_;}
      void set_prediction_variance(double var) {prediction_variance_ = var;}

      const Vector &kalman_gain() const {return kalman_gain_;}
      void set_kalman_gain(const Vector &gain) {kalman_gain_ = gain;}
      
     private:
      const ScalarStateSpaceModelBase *model_;
      ScalarMarginalDistribution *previous_;
      double prediction_error_;
      double prediction_variance_;
      Vector kalman_gain_;
    };
  }  // namespace Kalman

  // A Kalman filter for state space models with scalar outcomes.
  class ScalarKalmanFilter : public KalmanFilterBase {
   public:
    explicit ScalarKalmanFilter(ScalarStateSpaceModelBase *model);

    // Run the full Kalman filter over all the data held by the model.
    void update() override;

    // Update the Kalman filter at time t given observation y, which might be
    // different than y[t] held by the model (e.g. when doing posterior
    // simulation).
    void update(double y, int t, bool missing = false);

    void fast_disturbance_smooth() override;

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
    int size() const override {return nodes_.size();}
    
   private:
    ScalarStateSpaceModelBase *model_;
    std::vector<Kalman::ScalarMarginalDistribution> nodes_;
  };

}  // namespace BOOM

#endif  //  BOOM_STATE_SPACE_SCALAR_KALMAN_FILTER_HPP_
