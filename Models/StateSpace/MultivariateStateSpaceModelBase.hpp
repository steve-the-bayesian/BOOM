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
#include "LinAlg/Array.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include "Models/StateSpace/Filters/SparseVector.hpp"
#include "Models/StateSpace/Filters/ConditionalIidKalmanFilter.hpp"
#include "Models/StateSpace/Filters/ConditionallyIndependentKalmanFilter.hpp"
#include "Models/StateSpace/PosteriorSamplers/SufstatManager.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/StateSpace/StateModelVector.hpp"

#include "cpputil/ThreadTools.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {
  //===========================================================================
  // The general state space model is 
  //        y[t] = Z[t] * alpha[t] + epsilon[t] 
  //  alpha[t+1] = T[t] * alpha[t] + eta[t]
  //
  // The distinguishing feature of this class (and its children) is that y[t] is
  // a vector.  The general state space structure applies to this model, but we
  // can also structure things so that some components of state are
  // variable-specific.  This allows for MCMC strategies like
  //
  // (1) Sample shared state given variable specific state.
  // (2) Sample variable specific state given shared state.
  // (3) Sample parameters.
  //
  // This algorithm would have slightly worse mixing behavior than the MCMC that
  // just drew all state simultaneously, but the variable-specific portion can
  // be multi-threaded, and each Kalman filter based simulation draws a much
  // smaller state, so it has the potential to be fast.
  class MultivariateStateSpaceModelBase : virtual public Model {
   public:
    MultivariateStateSpaceModelBase()
        : state_is_fixed_(false)
    {}
    
    MultivariateStateSpaceModelBase *clone() const override = 0;
    MultivariateStateSpaceModelBase & operator=(
        const MultivariateStateSpaceModelBase &rhs);

    virtual int time_dimension() const = 0;

    // The 'state' component of this model refers to the components of state
    // shared across all time series.
    virtual int state_dimension() const = 0;
    virtual int number_of_state_models() const = 0;

    // ----- virtual functions required by the base class: ----------
    // These must be implemented by the concrete class.
    //---------------------------------------------------------------------------
    // virtual bool is_missing_observation(int t) const = 0;
    virtual Model *observation_model() = 0;
    virtual const Model *observation_model() const = 0;
    
    virtual void kalman_filter() = 0;
    virtual void observe_state(int t) = 0;
    virtual void observe_data_given_state(int t) = 0;

    // Sets the behavior of all client state models to 'behavior.'  State models
    // that can be represented as mixtures of normals should be set to MIXTURE
    // during data augmentation, and MARGINAL during forecasting.
    void set_state_model_behavior(StateModel::Behavior behavior);

    virtual void impute_state(RNG &rng);
    
    //---------------- Parameters for structural equations. -------------------
    // Durbin and Koopman's Z[t].  Defined as Y[t] = Z[t] * state[t] + error.
    // Note the lack of transpose on Z[t], so in the case of a single time
    // series Z[t] is a row vector.
    //
    // Args:
    //   t: The time index for which observation coefficients are desired.
    //   observed: Indicates which components of the observation at time t are
    //     observed (as opposed to missing).
    //
    // Returns:
    //   A subset of the observation coefficients at time t.  Row j of Z[t] is
    //   included if and only if observed[j] == true.
    virtual const SparseKalmanMatrix *observation_coefficients(
        int t, const Selector &observed) const = 0;

    // Return the KalmanFilter object responsible for filtering the data.
    virtual MultivariateKalmanFilterBase & get_filter() = 0;
    virtual const MultivariateKalmanFilterBase & get_filter() const = 0;
    virtual MultivariateKalmanFilterBase & get_simulation_filter() = 0;
    virtual const MultivariateKalmanFilterBase & get_simulation_filter() const = 0;

    // Durbin and Koopman's T[t] built from state models.
    virtual const SparseKalmanMatrix *state_transition_matrix(int t) const {
      return state_model_vector().state_transition_matrix(t);
    }

    // Durbin and Koopman's RQR^T.  Built from state models, often less than
    // full rank.
    virtual const SparseKalmanMatrix *state_variance_matrix(int t) const {
      return state_model_vector().state_variance_matrix(t);
    }

    // Durbin and Koopman's R matrix from the transition equation:
    //    state[t+1] = (T[t] * state[t]) + (R[t] * state_error[t]).
    //
    // This is the matrix that takes the low dimensional state_errors and turns
    // them into error terms for states.
    virtual const SparseKalmanMatrix *state_error_expander(int t) const {
      return state_model_vector().state_error_expander(t);
    }

    // The full rank variance matrix for the errors in the transition equation.
    // This is Durbin and Koopman's Q[t].  The errors with this variance are
    // multiplied by state_error_expander(t) to produce the errors described by
    // state_variance_matrix(t).
    virtual const SparseKalmanMatrix *state_error_variance(int t) const {
      return state_model_vector().state_error_variance(t);
    }

    //----------------- Access to data -----------------
    // Returns the value of y observed at time t.
    virtual ConstVectorView observation(int t) const = 0;

    // Some models contain components other than the shared state component.
    // Learning for such models involves subtracting off contributions from
    // other components, leaving just the contributions from the shared state
    // multivariate model.
    //
    // Returns the residual observation obtained after subtracting off
    // components other than shared state components.
    virtual ConstVectorView adjusted_observation(int time) const = 0;

    // Elements of the returned value indicate which elements of observation(t)
    // are actually observed.  In the typical case all elements will be true.
    virtual const Selector &observed_status(int t) const = 0;

    // The contributions of each state model to the mean of the response at each
    // time point.
    //
    // Returns:
    //   Matrix element (t, d) gives the contribution of state model
    //   which_state_model to dimension d of the response variable at time t.
    virtual Matrix state_contributions(int which_state_model) const = 0;

    //    void signal_complete_data_reset();

    //---------------- Access to state ---------------------------------------
    // A cast will be necessary in the child classes.
    
    virtual StateModelBase *state_model(int s) = 0;
    virtual const StateModelBase *state_model(int s) const = 0;

    ConstVectorView final_state() const {
      if (time_dimension() <= 0) {
        report_error("State size is zero.");
      }
      return shared_state(time_dimension() - 1);
    }
    
    VectorView state_component(Vector &full_state, int s) const {
      return state_model_vector().state_component(full_state, s);
    }
    VectorView state_component(VectorView &full_state, int s) const {
      return state_model_vector().state_component(full_state, s);
    }
    ConstVectorView state_component(
        const ConstVectorView &full_state, int s) const {
      return state_model_vector().state_component(full_state, s);
    }

    const Matrix &shared_state() const { return shared_state_; }

    ConstVectorView shared_state(int t) const {return shared_state().col(t);}

    ConstSubMatrix full_state_subcomponent(int state_model_index) const {
      return state_model_vector().full_state_subcomponent(
          shared_state_, state_model_index);
    }
    
    SubMatrix mutable_full_state_subcomponent(int state_model_index) {
      return state_model_vector().mutable_full_state_subcomponent(
          shared_state_, state_model_index);
    }
    
    Vector initial_state_mean() const;
    SpdMatrix initial_state_variance() const;

    // Set the shared state to the specified value, and mark the state as
    // 'fixed' so that it will no longer be updated by calls to 'impute_state'.
    // This function is intended for debugging purposes only.
    //
    // Args:
    //   state:  The state matrix.  Columns are time. Rows are state elements.
    void permanently_set_state(const Matrix &state);

    // The number of time series being modeled.  Not all model types know this.
    // For example the number of series in a dynamic intercept model changes
    // from time point to time point.  For this reason the default
    // implementation is to return -1.
    virtual int nseries() const { return -1; }
    
   protected:
    // Access to the state model vector owned by descendents.
    using StateModelVectorBase = StateSpaceUtils::StateModelVectorBase;
    virtual StateModelVectorBase & state_model_vector() = 0;
    virtual const StateModelVectorBase & state_model_vector() const = 0;

    // Advance the state vector to a future time stamp.  This method is used to
    // implement simulations from the posterior predictive distribution.
    // Args:
    //   rng:  The random number generator.
    //   time: The current time stamp of the state vector, as a number of time
    //     steps from the end of the training data.
    //   state:  The current value of the state vector.
    //   timestamp: The timestamp to advance to.  This must be no smaller than
    //     'time'.  On exit, 'time' will equal 'timestamp'.
    //   observation_index: This is only used to print an error message, if
    //     needed.  It is the observation number for the data point being
    //     predicted.
    //
    // Side effects:
    //   On exit, 'time' is advanced to 'timestamp', and 'state' is a draw from
    //   the state vector at time time_dimension() + timestamp.
    void advance_to_timestamp(RNG &rng, int &time, Vector &state,
                              int timestamp, int observation_index) const;

    Vector simulate_next_state(RNG &rng, const ConstVectorView &last,
                               int t) const;

    virtual void clear_client_data();
    void observe_fixed_state();

   private:
    // Implementation for impute_state.  
    void simulate_initial_state(RNG &rng, VectorView initial_state) const;
    Vector simulate_state_error(RNG &rng, int t) const;
    
    void simulate_forward(RNG &rng);
    void propagate_disturbances(RNG &rng);

    // If observation t is not fully observed, impute its missing values.  This
    // is a full imputation, including regression and series-level state model
    // effects.
    virtual void impute_missing_observations(int t, RNG &rng) = 0;
    
    void resize_state();
    
    // Simulate a fake observation to use as part of the Durbin-Koopman state
    // simulation algorithm.  If observed_status(t) is less than fully observed,
    // only the observed parts should be simulated.
    virtual Vector simulate_fake_observation(RNG &rng, int t) = 0;

    Matrix shared_state_;
    bool state_is_fixed_;
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

  //===========================================================================
  class ConditionallyIndependentMultivariateStateSpaceModelBase
      : public MultivariateStateSpaceModelBase {
   public:
    ConditionallyIndependentMultivariateStateSpaceModelBase()
        : filter_(this),
          simulation_filter_(this)
    {}

    // Variance of the observation error at time t.  Durbin and Koopman's H[t].
    virtual DiagonalMatrix observation_variance(int t) const = 0;

    virtual double single_observation_variance(int t, int dim) const = 0;
    
    //---------------- Prediction, filtering, smoothing ---------------
    // Run the full Kalman filter over the observed data, saving the information
    // in the filter_ object.  The log likelihood is computed as a by-product.
    void kalman_filter() override { filter_.update(); }

    using Filter = ConditionallyIndependentKalmanFilter;
    Filter &get_filter() override {return filter_;}
    const Filter &get_filter() const override { return filter_; }
    Filter &get_simulation_filter() override { return simulation_filter_; }
    const Filter & get_simulation_filter() const override {
      return simulation_filter_;
    }
    
   private:
    // This function is 
    Vector simulate_fake_observation(RNG &rng, int t) override;

    ConditionallyIndependentKalmanFilter filter_;
    ConditionallyIndependentKalmanFilter simulation_filter_;
  };

  //===========================================================================
  class ConditionalIidMultivariateStateSpaceModelBase
      : public MultivariateStateSpaceModelBase {
   public:
    ConditionalIidMultivariateStateSpaceModelBase();

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

   private:
    // Simulate a fake observation at time t to use as part of the Durbin and
    // Koopman data augmentation algorithm.
    //
    // This override handles the case where the dimension of the response is
    // known.  If the dimension is time varying then this function will generate
    // an error, so child classes involving time varying dimension must handle
    // the time varying case with their own overrides.
    Vector simulate_fake_observation(RNG &rng, int t) override;

    ConditionalIidKalmanFilter filter_;
    ConditionalIidKalmanFilter simulation_filter_;
  };
  
} // namespace BOOM



#endif //  BOOM_MULTIVARIATE_STATE_SPACE_MODEL_BASE_HPP_

