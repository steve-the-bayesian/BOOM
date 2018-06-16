#ifndef BOOM_MULTIVARIATE_STATE_SPACE_MODEL_BASE_HPP_
#define BOOM_MULTIVARIATE_STATE_SPACE_MODEL_BASE_HPP_
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

#include <memory>
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include "Models/StateSpace/Filters/SparseVector.hpp"
#include "Models/StateSpace/Filters/ConditionalIidKalmanFilter.hpp"

#include "Models/StateSpace/PosteriorSamplers/SufstatManager.hpp"

#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "cpputil/ThreadTools.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {
  //===========================================================================
  class MultivariateStateSpaceModelBase : public StateSpaceModelBase {
   public:
    MultivariateStateSpaceModelBase *clone() const override = 0;

    // ----- virtual functions required by the base class: ----------
    // virtual int time_dimension() const = 0;
    // virtual bool is_missing_observation(int t) const = 0;
    // virtual PosteriorModeModel *observation_model() = 0;
    // virtual const PosteriorModeModel *observation_model() const = 0;
    // virtual void kalman_filter() = 0;
    // virtual void observe_state(int t) = 0;
    // virtual void observe_data_given_state(int t) = 0;
    // virtual void update_observation_model(Vector &r, SpdMatrix &N, int t,
    //     bool save_state_distributions, bool update_sufficient_statistics,
    //     Vector *gradient) = 0;
    // virtual void simulate_forward(RNG &rng) = 0;
    // virtual void smooth_observed_disturbances() = 0;
    // virtual void propagate_disturbances() = 0;
    
    //---------------- Parameters for structural equations. -------------------
    // Durbin and Koopman's Z[t].  Defined as Y[t] = Z[t] * state[t] + error.
    // Note the lack of transpose on Z[t].
    virtual const SparseKalmanMatrix *observation_coefficients(int t) const = 0;

    // Return the KalmanFilter object responsible for filtering the data.
    MultivariateKalmanFilterBase & get_filter() override = 0;
    const MultivariateKalmanFilterBase & get_filter() const override = 0;
    MultivariateKalmanFilterBase & get_simulation_filter() override = 0;
    const MultivariateKalmanFilterBase & get_simulation_filter() const override = 0;
    
    //----------------- Access to data -----------------
    // Returns the value of y observed at time t.
    virtual const Vector &observation(int t) const = 0;

    // Elements of the returned value indicate which elements of observation(t)
    // are actually observed.  In the typical case all elements will be true.
    virtual const Selector &observed_status(int t) const = 0;
    
    std::vector<Vector> state_contributions() const;
    
   protected:
    // Implements part of a single step of the E-step in the EM algorithm or
    // gradient computation for the gradient of the observed data log
    // likelihood.
    //
    // Args:
    //   r: Durbin and Koopman's r vector, which is a scaled version of the
    //     smoothed state mean.  On entry r is r[t].  On exit it is r[t-1].
    //   N: Durbin and Koopman's N matrix, which is a scaled version of the
    //     smoothed state variance. On entry N is N[t].  On exit it is N[t-1].
    //   t:  The time index for the update.
    //   save_state_distributions: If true then the observation error mean and
    //     variance (if y is univariate) or precision (if y is multivariate)
    //     will be saved in the Kalman filter.
    //   update_sufficient_statistics: If true then the complete data sufficient
    //     statistics for the observation model will be updated as in the E-step
    //     of the EM algorithm.
    //   gradient: If non-NULL then the observation model portion of the
    //     gradient will be incremented to reflect information at time t.
    //
    // Side effects:
    //   r and N are "downdated" to time t-1 throug a call to the disturbance
    //   smoother.  The Kalman filter is updated by the smoothing recursions.
    void update_observation_model(Vector &r, SpdMatrix &N, int t,
                                  bool save_state_distributions,
                                  bool update_sufficient_statistics,
                                  Vector *gradient) override;

    // Update the complete data sufficient statistics for the observation model
    // based on the posterior distribution of the observation model error term
    // at time t.
    //
    // Args:
    //   t: The time of the observation.
    //   observation_error_mean: Mean of the observation error given model
    //     parameters and all observed y's.
    //   observation_error_variance: Variance of the observation error given
    //     model parameters and all observed y's.
    virtual void update_observation_model_complete_data_sufficient_statistics(
        int t, const Vector &observation_error_mean,
        const SpdMatrix &observation_error_variance) = 0;

    // Increment the portion of the log-likelihood gradient pertaining to the
    // parameters of the observation model.
    //
    // Args:
    //   gradient: The subset of the log likelihood gradient pertaining to the
    //     observation model.  The gradient will be incremented by the
    //     derivatives of log likelihood with respect to the observation model
    //     parameters.
    //   t:  The time index of the observation error.
    //   observation_error_mean: The posterior mean of the observation error at
    //     time t.
    //   observation_error_variance: The posterior variance of the observation
    //     error at time t.
    virtual void update_observation_model_gradient(
        VectorView gradient, int t, const Vector &observation_error_mean,
        const SpdMatrix &observation_error_variance) = 0;

   private:
    void simulate_forward(RNG &rng) override;
    void propagate_disturbances() override;
    virtual Vector simulate_observation(RNG &rng, int t) = 0;

    // Workspace for disturbance smoothing.
    Vector r0_sim_;
    Vector r0_obs_;
  };

  //===========================================================================
  class ConditionalIidMultivariateStateSpaceModelBase
      : public MultivariateStateSpaceModelBase {
   public:
    // All observations at time t have this variance.
    virtual double observation_variance(int t) const = 0;

    // Run the full Kalman filter over the observed data, saving the information
    // in the filter_ object.  The log likelihood is computed as a by-product.
    void kalman_filter() override {
      filter_.update();
    }

    ConditionalIidKalmanFilter &get_filter() override;
    const ConditionalIidKalmanFilter &get_filter() const override;
    ConditionalIidKalmanFilter &get_simulation_filter() override;
    const ConditionalIidKalmanFilter &get_simulation_filter() const override;

    Vector simulate_observation(RNG &rng, int t) override;
    
   private:
    ConditionalIidKalmanFilter filter_;
    ConditionalIidKalmanFilter simulation_filter_;
  };
  
  //===========================================================================
  class ConditionallyIndependentMultivariateStateSpaceModelBase
      : public MultivariateStateSpaceModelBase {
   public:
    // Variance of the observation error at time t.  Durbin and Koopman's H[t].
    virtual DiagonalMatrix observation_variance(int t) const = 0;
    
    //---------------- Prediction, filtering, smoothing ---------------
    // Run the full Kalman filter over the observed data, saving the information
    // in the filter_ object.  The log likelihood is computed as a by-product.
    void kalman_filter() override;
  };

  //===========================================================================
  class GeneralMultivariateStateSpaceModelBase
      : public MultivariateStateSpaceModelBase {
   public:
    virtual SpdMatrix observation_variance(int t) const = 0;

    //---------------- Prediction, filtering, smoothing ---------------
    // Run the full Kalman filter over the observed data, saving the information
    // in the filter_ object.  The log likelihood is computed as a by-product.
    void kalman_filter() override;
  };

} // namespace BOOM



#endif //  BOOM_MULTIVARIATE_STATE_SPACE_MODEL_BASE_HPP_

