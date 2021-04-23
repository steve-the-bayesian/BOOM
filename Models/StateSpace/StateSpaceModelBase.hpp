#ifndef BOOM_STATE_SPACE_MODEL_BASE_HPP_
#define BOOM_STATE_SPACE_MODEL_BASE_HPP_
// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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
#include "cpputil/ThreadTools.hpp"
#include "cpputil/math_utils.hpp"

// There is an include order issue here.  ThreadTools.hpp must come before the
// headers listed below.

#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/StateSpace/Filters/KalmanFilterBase.hpp"
#include "Models/StateSpace/Filters/ScalarKalmanFilter.hpp"
#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include "Models/StateSpace/Filters/SparseVector.hpp"
#include "Models/StateSpace/MultiplexedData.hpp"
#include "Models/StateSpace/PosteriorSamplers/SufstatManager.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/StateSpace/StateModelVector.hpp"


namespace BOOM {
  //===========================================================================
  // A base class for logic common to ScalarStateSpaceModelBase and
  // MultivariateStateSpaceModelBase.
  class StateSpaceModelBase : virtual public Model {
   public:
    StateSpaceModelBase();
    StateSpaceModelBase(const StateSpaceModelBase &rhs);
    StateSpaceModelBase(StateSpaceModelBase &&rhs) = default;
    StateSpaceModelBase *clone() const override = 0;
    virtual StateSpaceModelBase *deepclone() const = 0;
    StateSpaceModelBase &operator=(const StateSpaceModelBase &rhs);
    StateSpaceModelBase &operator=(StateSpaceModelBase &&rhs) = default;

    //----- Sizes of things ------------
    // The number of time points in the training data.
    virtual int time_dimension() const = 0;

    // Number of elements in the state vector at a single time point.
    virtual int state_dimension() const {
      return state_models_.state_dimension();
    }

    // The number of state models.  Presently, a fixed regression model does not
    // count as a state model, nor does a Harvey Cumulator.  This may change in
    // the future.
    virtual int number_of_state_models() const {
      return state_models_.size();
    }

    // Returns true if observation t is missing, and false otherwise.  If the
    // observation at time t is multivariate, then is_missing_observation(t) ==
    // true indicates that the entire observation is missing.
    virtual bool is_missing_observation(int t) const = 0;

    //--------- Access to client models and model parameters ---------------
    // Returns a pointer to the model responsible for the observation variance.
    //
    // Some multivariate models use a proxy state space model for which there is
    // no observation model, in which case the return value can be nullptr.
    virtual PosteriorModeModel *observation_model() = 0;
    virtual const PosteriorModeModel *observation_model() const = 0;

    // Returns a pointer to the specified state model.
    virtual Ptr<StateModel> state_model(int s) { return state_models_[s]; }
    virtual const Ptr<StateModel> state_model(int s) const {
      return state_models_[s];
    }

    // Overrides that would normally be handled by a parameter policy.  These
    // are needed to ensure that parameters are vectorized in the correct order.
    std::vector<Ptr<Params>> parameter_vector() override;
    const std::vector<Ptr<Params>> parameter_vector() const override;

    // Return the subset of the vectorized set of model parameters pertaining to
    // the observation model, or to a specific state model.  Can also be used to
    // take subsets of gradients of functions of model parameters.
    //
    // Args:
    //   model_parameters: A vector of model parameters, ordered in the same way
    //     as model->vectorize_params(true).
    //   s: The index of the state model for which a parameter subset is
    //     desired.
    //
    // Returns:
    //   The subset of the parameter vector corresponding to the specified
    //   model.
    VectorView state_parameter_component(Vector &model_parameters, int s) const;
    ConstVectorView state_parameter_component(const Vector &model_parameters,
                                              int s) const;
    VectorView observation_parameter_component(Vector &model_parameters) const;
    ConstVectorView observation_parameter_component(
        const Vector &model_parameters) const;

    // --------------------------- Access to state ---------------------------
    // Add structure to the state portion of the model.  This is for local
    // linear trend and different seasonal effects.  It is not for regression,
    // which this class will handle separately.  The state model should be
    // initialized (including the model for the initial state), and have its
    // learning method (e.g. posterior sampler) set prior to being added using
    // add_state.
    void add_state(const Ptr<StateModel> &);

    // Returns a draw of the state vector (produced by impute_state()) for the
    // last time point in the training data.
    ConstVectorView final_state() const { return state_.last_col(); }

    // Returns the draw of the state vector (produced by impute_state()) at time
    // t.
    ConstVectorView state(int t) const { return state_.col(t); }

    // Returns the draw of the full state vector.  Each row is a state element.
    // Each column corresponds to a time point in the training data.
    const Matrix &state() const { return state_; }

    // Takes the full state vector as input, and returns the component of the
    // state vector belonging to state model s.
    //
    // Args:
    //   state:  The full state vector.
    //   s:  The index of the state model whose state component is desired.
    //
    // Returns:
    //   The subset of the 'state' argument corresponding to state model 's'.
    VectorView state_component(Vector &state, int s) const {
      return state_models_.state_component(state, s);
    }
    VectorView state_component(VectorView &state, int s) const {
      return state_models_.state_component(state, s);
    }
    ConstVectorView state_component(const ConstVectorView &state, int s) const {
      return state_models_.state_component(state, s);
    }

    // Return the component of the full state error vector corresponding to a
    // given state model.
    //
    // Args:
    //   full_state_error: The error for the full state vector (i.e. all state
    //     models).
    //   state_model_number:  The index of the desired state model.
    //
    // Returns:
    //   The error vector for just the specified state model.
    ConstVectorView const_state_error_component(const Vector &full_state_error,
                                                int state_model_number) const {
      return state_models_.const_state_error_component(
          full_state_error, state_model_number);
    }
    VectorView state_error_component(Vector &full_state_error,
                                     int state_model_number) const {
      return state_models_.state_error_component(
          full_state_error, state_model_number);
    }

    // Returns the subcomponent of the (block diagonal) error variance matrix
    // corresponding to a specific state model.
    //
    // Args:
    //   full_error_variance:  The full state error variance matrix.
    //   state: The index of the state model defining the desired sub-component.
    ConstSubMatrix state_error_variance_component(
        const SpdMatrix &full_error_variance, int state) const {
      return state_models_.state_error_variance_component(
          full_error_variance, state);
    }

    // Returns the complete state vector (across time, so the return value is a
    // matrix) for a specified state component.
    //
    // Args:
    //   state_model_index:  The index of the desired state model.
    //
    // Returns:
    //   A matrix giving the imputed value of the state vector for the specified
    //   state model.  The matrix has S rows and T columns, where S is the
    //   dimension of the state vector for the specified state model, and T is
    //   the number of time points.
    ConstSubMatrix full_state_subcomponent(int state_model_index) const {
      return state_models_.full_state_subcomponent(state_, state_model_index);
    }
    SubMatrix mutable_full_state_subcomponent(int state_model_index) {
      return state_models_.mutable_full_state_subcomponent(
          state_, state_model_index);
    }

    // The next two functions are mainly used for debugging a simulation.  You
    // can 'permanently_set_state' to the 'true' state value, then see if the
    // model recovers the parameters.  These functions are unlikely to be useful
    // in an actual data analysis.
    void permanently_set_state(const Matrix &state);
    void observe_fixed_state();

    // Parameters of initial state distribution, specified in the state models
    // given to add_state.  The initial state refers to the state at time 0
    // (other implementations sometimes assume the initial state is at time -1).
    virtual Vector initial_state_mean() const;
    virtual SpdMatrix initial_state_variance() const;

    //------------- Model matrices for structural equations. --------------
    // Durbin and Koopman's T[t] built from state models.
    virtual const SparseKalmanMatrix *state_transition_matrix(int t) const {
      return state_models_.state_transition_matrix(t);
    }

    // Durbin and Koopman's RQR^T.  Built from state models, often less than
    // full rank.
    virtual const SparseKalmanMatrix *state_variance_matrix(int t) const {
      return state_models_.state_variance_matrix(t);
    }

    // Durbin and Koopman's R matrix from the transition equation:
    //    state[t+1] = (T[t] * state[t]) + (R[t] * state_error[t]).
    //
    // This is the matrix that takes the low dimensional state_errors and turns
    // them into error terms for states.
    virtual const ErrorExpanderMatrix *state_error_expander(int t) const {
      return state_models_.state_error_expander(t);
    }

    // The full rank variance matrix for the errors in the transition equation.
    // This is Durbin and Koopman's Q[t].  The errors with this variance are
    // multiplied by state_error_expander(t) to produce the errors described by
    // state_variance_matrix(t).
    virtual const SparseKalmanMatrix *state_error_variance(int t) const {
      return state_models_.state_error_variance(t);
    }

    //----------------- Access to data -----------------
    // Clears sufficient statistics for state models and for the client model
    // describing observed data given state.
    virtual void clear_client_data();

    // This function is designed to be called in the constructor of a
    // PosteriorSampler managing this object.
    //
    // Args:
    //   observer: A functor, typically containing a pointer to a
    //     PosteriorSampler managing both this object and a set of complete data
    //     sufficient statistics.  Calling the functor with a non-negative
    //     integer t indicates that observation t has changed, so the complete
    //     data sufficient statistics should be updated.  Calling the functor
    //     with a negative integer is a signal to reset the complete data
    //     sufficient statistics.
    void register_data_observer(StateSpace::SufstatManagerBase *observer);

    // Send a signal to all data observers (typically just 1) that the complete
    // data for observation t has changed.
    //
    // This function is designed to be called as part of the implementation for
    // observe_data_given_state.
    void signal_complete_data_change(int t);

    //--------- Utilities for implementing data augmentation ----------
    // Sets the behavior of all client state models to 'behavior.'  State models
    // that can be represented as mixtures of normals should be set to MIXTURE
    // during data augmentation, and MARGINAL during forecasting.
    void set_state_model_behavior(StateModel::Behavior behavior);

    // Use the Durbin and Koopman method of forward filtering-backward
    // sampling.
    //  1. Sample the state vector and an auxiliary y vector from the model.
    //  2. Subtract the expected value of the state given the simulated y.
    //  3. Add the expected value of the state given the observed y.
    //
    // Args:
    //   rng:  The random number generator to use for simulation.
    virtual void impute_state(RNG &rng);

    //---------------- Prediction, filtering, smoothing ---------------
    // Run the full Kalman filter over the observed data, saving the information
    // produced in the process in full_kalman_storage_.  The log likelihood is
    // computed as a by-product.
    virtual void kalman_filter() = 0;

    // Return the KalmanFilter object responsible for filtering the data.
    virtual KalmanFilterBase & get_filter() = 0;
    virtual const KalmanFilterBase & get_filter() const = 0;
    virtual KalmanFilterBase & get_simulation_filter() = 0;
    virtual const KalmanFilterBase & get_simulation_filter() const = 0;

    //------------- Parameter estimation by MLE and MAP --------------------
    // Set model parameters to their maximum-likelihood estimates, and return
    // the likelihood at the MLE.  Note that some state models cannot be used
    // with this method.  In particular, regression models with spike-and-slab
    // priors can't be MLE'd.  If the model contains such a state model then an
    // exception will be thrown.
    //
    // Args:
    //   epsilon: Convergence for optimization algorithm will be declared when
    //     consecutive values of log-likelihood are observed with a difference
    //     of less than epsilon.
    //
    // Returns:
    //   The value of the log-likelihood at the MLE.
    double mle(double epsilon = 1e-5);

    // The E-step of the EM algorithm.  Computes complete data sufficient
    // statistics for state models and the observation variance parameter.
    //
    // Args:
    //   save_state_distributions: If true then the state distributions (the
    //     mean vector a and the variance P) will be saved in
    //     full_kalman_storage_.  If not then these quantities will be left as
    //     computed by the full_kalman_filter.
    //
    // Returns:
    //   The log likelihood of the data computed at the model's current
    //   parameter values.
    double Estep(bool save_state_distributions);

    // To be called after calling Estep().  Given the current values of the
    // complete data sufficient statistics, set model parameters to their
    // complete data maximum likelihood estimates.
    //
    // Args:
    //   epsilon: Additive convergence criteria for models that require
    //     numerical optimization.
    void Mstep(double epsilon);

    // Returns true if all the state models have been assigned priors that
    // implement find_posterior_mode.
    bool check_that_em_is_legal() const;

    // Returns a matrix containing the posterior mean of the state at each time
    // period.  These are stored in kalman_storage_, and computed by the
    // combination of full_kalman_filter() and either a call to Estep(true) or
    // kalman_smoother().
    //
    // Rows of the returned matrix correspond to state components.  Columns
    // correspond to time points.
    Matrix state_posterior_means() const;

    // If called after kalman_filter() and before any smoothing operations,
    // then state_filtering_means returns a matrix where column t contains the
    // expected value of the state at time t given data to time t-1.
    //
    // It is an error to call this function if kalman_filter() has not been
    // called, if smoothing steps have been taken after calling the filter, or
    // if model parameters have been changed.  In such cases the returned matrix
    // will not contain the expected values.
    Matrix state_filtering_means() const;

    // Returns the posterior variance (given model parameters and observed data)
    // of the state at time t.  This is stored in kalman_storage_, and computed
    // by the combination of kalman_filter() and either a call to
    // Estep(true) or kalman_smoother().
    const SpdMatrix &state_posterior_variance(int t) const;

    //---------- Likelihood calculations ---------------------------
    // Returns the log likelihood under the current set of model parameters.  If
    // the Kalman filter is current (i.e. no parameters or data have changed
    // since the last time it was run) then this function does no actual work.
    // Otherwise it sparks a fresh Kalman filter run.
    double log_likelihood();

    // Evaluate the model log likelihood as a function of the model parameters.
    //
    // Args:
    //   parameters: The vector of model parameters in the same order as
    //     produced by vectorize_params(true).
    double log_likelihood(const Vector &parameters);

    // Evaluate the log likelihood function and its derivatives as a function of
    // model parameters.
    // Args:
    //   parameters: The vector of model parameters in the same order as
    //     produced by vectorize_params(true).
    //   gradient: Will be filled with the derivatives of log likelihood with
    //     respect to the vector of model parameters.  The gradient vector will
    //     be resized if needed, and intialized to zero.
    // Returns:
    //   The value of log likelihood at the specified parameters.
    double log_likelihood_derivatives(const Vector &parameters,
                                      Vector &gradient);

    // Evaluate the log likelihood function and its derivatives at the current
    // model parameters.
    //
    // Args:
    //   gradient: Will be filled with the deriviatives of log likelihood with
    //     respect to the current vector of model parameters.  The gradient
    //     vector will be resized if needed, and intialized to zero.
    //
    // Returns:
    //   The value of log likelihood at the current set of parameter values.
    //
    // NOTE: This function is used to implement the version that also
    // takes a vector of arbitrary parameters.
    double log_likelihood_derivatives(VectorView gradient);

    //----------------- Simulating from the model -------------------
    // Simulate the initial model state from the initial state distribution.
    // The simulated value is returned in the vector view function argument.
    // The initial state refers to the state at time 0 (other implementations
    // sometimes assume the initial state is at time -1).
    virtual void simulate_initial_state(RNG &rng, VectorView state0) const;

    // Simulates the value of the state vector for the current time period, t,
    // given the value of state at the previous time period, t-1.
    // Args:
    //   last:  Value of state at time t-1.
    //   next:  VectorView to be filled with state at time t.
    //   t:  The time index of 'next'.
    void simulate_next_state(RNG &rng, const ConstVectorView &last,
                             VectorView next, int t) const;
    Vector simulate_next_state(RNG &rng,
                               const Vector &current_state,
                               int t) const;

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
    void advance_to_timestamp(RNG &rng, int &time, Vector &state, int timestamp,
                              int observation_index) const;

    // Returns a draw of the predictive distribution of 'state' over the next
    // 'horizon' time periods.  If any state models depend on external data,
    // they must have access to that data over the forecast horizon.
    //
    // Args:
    //   rng:  The random number generator to use for the simulation.
    //   horizon:  The number of time periods into the future to forecast.
    // Returns:
    //   A matrix with 'horizon + 1' columns, where column t contains the
    //   simulated state t periods after the final state at time
    //   time_dimension().  That means column zero contains final_state().  The
    //   matrix values are simulations from the predictive distribution of the
    //   state given data to time_dimension().
    Matrix simulate_state_forecast(RNG &rng, int horizon) const;

    // Simulates the error for the state at time t+1.  (Using the notation of
    // Durbin and Koopman, this uses the model matrices indexed as t.)
    //
    // Returns a vector of size state_dimension().  If the model matrices are
    // not full rank then some elements of this vector will be deterministic
    // functions of other elements.
    virtual Vector simulate_state_error(RNG &rng, int t) const;

    // Reset the size of the state_ matrix so that it has state_dimension() rows
    // and time_dimension() columns.
    void resize_state();

   protected:
    // Remove any posterior sampling methods from this model and all client
    // models.  Copy posterior samplers from rhs to *this.
    void copy_samplers(const StateSpaceModelBase &rhs);

    // Update the complete data sufficient statistics for the state models,
    // given the posterior distribution of the state error at time t (for the
    // transition between times t and t+1), given model parameters and all
    // observed data.
    //
    // Args:
    //   state_error_mean: The mean of the state error at time t given observed
    //     data and model parameters.
    //   state_error_variance: The variance of the state error at time t given
    //     observed data and model parameters.
    void update_state_level_complete_data_sufficient_statistics(
        int t, const Vector &state_error_mean,
        const SpdMatrix &state_error_variance);

    Matrix &mutable_state() { return state_; }

    //-----Implementation details for the Kalman filter and smoother -----
    // Compute the contribution to the complete data sufficient statistics, for
    // the observation model and all the state models, once the state at time
    // 't' has been imputed.
    void observe_state(int t);

    // The initial state can be treated specially, though the default for this
    // function is a no-op.  The initial state refers to the state at time 0
    // (other implementations sometimes assume the initial state is at time -1).
    virtual void observe_initial_state();

    // This is a hook that tells the observation model to update its sufficient
    // statisitcs now that the state for time t has been observed.
    virtual void observe_data_given_state(int t) = 0;

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
    virtual void update_observation_model(Vector &r, SpdMatrix &N, int t,
                                          bool save_state_distributions,
                                          bool update_sufficient_statistics,
                                          Vector *gradient) = 0;

    // A helper function used to implement average_over_latent_data().
    // Increments the gradient of log likelihood contribution of the state
    // models at time t (for the transition to time t+1).
    //
    // Args:
    //   gradient:  The gradient to be updated.
    //   t: The time index for the update.  Homogeneous models will
    //     ignore this, but models where the Kalman matrices depend on
    //     t need to know it.
    //   state_error_mean: The posterior mean of the state errors at
    //     time t (for the transition to time t+1).
    //   state_error_mean: The posterior variance of the state errors
    //     at time t (for the transition to time t+1).
    void update_state_model_gradient(Vector *gradient, int t,
                                     const Vector &state_error_mean,
                                     const SpdMatrix &state_error_variance);

    // Utility function used to implement E-step and log_likelihood_derivatives.
    //
    // Args:
    //   update_sufficient_statistics: If true then the complete data sufficient
    //     statistics for the observation model and the state models will be
    //     cleared and updated.  If false they will not be modified.
    //   save_state_distributions: If true then the state distributions (the
    //     mean vector a and the variance P) will be saved in kalman_storage_.
    //     If not then these quantities will be left as computed by the
    //     full_kalman_filter.
    //   gradient: If a nullptr is passed then no gradient information will be
    //     computed.  Otherwise the gradient vector is resized, cleared, and
    //     filled with the gradient of log likelihood.
    //
    // Returns:
    //   The log likeilhood value computed by the Kalman filter.
    double average_over_latent_data(bool update_sufficient_statistics,
                                    bool save_state_distributions,
                                    Vector *gradient);

    // Simulate fake data from the model, given current model parameters, as
    // part of Durbin and Koopman's state-simulation algorithm.
    virtual void simulate_forward(RNG &rng) = 0;

    // After the simulated and observed disturbances have been smoothed,
    // propagate them forward to achieve a draw of latent state given observed
    // data and parameters.
    //
    // As the state is finalized, the state models and the observation model are
    // updated to reflect the new state values.
    virtual void propagate_disturbances();

    // Send a signal to all data observers (typically just 1) that the
    // complete data sufficient statistics should be reset.
    void signal_complete_data_reset();

    // Some child classes maintain parallel vectors of state models.  When
    // implementing the copy constructor, the state models in the base class
    // need to be cleared out so that the overloaded add_state() in the child
    // classes can maintain parallelism correctly.
    void clear_state_models() { state_models_.clear(); }

   private:
    //----------------------------------------------------------------------
    // data starts here
    StateSpaceUtils::StateModelVector<StateModel> state_models_;

    // Position [s] is the index in the vector of parameters where the parameter
    // for state model s begins.  Note that the parameter vector for the
    // observation model begins in element 0.
    std::vector<int> parameter_positions_;

    // The most recent draw of the state from its posterior distribution.  This
    // matrix has state_dimension() rows and time_dimension() columns, so that
    // column t is the state at time t.
    Matrix state_;

    // A flag used for debugging.  If state_is_fixed_ is set then the state will
    // be held constant in the data imputation.
    bool state_is_fixed_;

    // Data observers exist so that changes to the (latent) data made by the
    // model can be incorporated by PosteriorSampler classes keeping track of
    // complete data sufficient statistics.  Gaussian models do not need
    // observers, but mixtures of Gaussians typically do.
    std::vector<StateSpace::SufstatManager> data_observers_;
  };

  //===========================================================================
  // Base class for models that assume a single scalar observation per time
  // period.
  class ScalarStateSpaceModelBase : public StateSpaceModelBase {
   public:
    ScalarStateSpaceModelBase();
    ScalarStateSpaceModelBase(const ScalarStateSpaceModelBase &rhs);
    ScalarStateSpaceModelBase *clone() const override = 0;
    ScalarStateSpaceModelBase *deepclone() const override = 0;

    //------------- Parameters for structural equations. --------------
    // Variance of observed data y[t], given state alpha[t].  Durbin and
    // Koopman's H.
    virtual double observation_variance(int t) const = 0;

    // Durbin and Koopman's Z[t].transpose() built from state models.
    virtual SparseVector observation_matrix(int t) const;

    //----------------- Access to data -----------------
    // Returns y[t], after adjusting for regression effects that are not
    // included in the state vector.  This is the value that the time series
    // portion of the model is supposed to describe.  If there are no regression
    // effects, or if the state contains a RegressionStateModel this is
    // literally y[t].  If there are regression effects it is y[t] - beta *
    // x[t].  If y[t] is missing then infinity() is returned.
    virtual double adjusted_observation(int t) const = 0;

    //---------------- Prediction, filtering, smoothing ---------------
    // Run the full Kalman filter over the observed data, saving the information
    // produced in the process in full_kalman_storage_.  The log likelihood is
    // computed as a by-product.
    void kalman_filter() override;

    // Returns the vector of one step ahead prediction errors for the training
    // data.
    Vector one_step_prediction_errors(bool standardize = false);

    virtual Matrix simulate_holdout_prediction_errors(
        int niter, int cutpoint_number, bool standardize) = 0;

    //------- Accessors for getting at state components -----------
    // Returns the contributions of each state model to the overall mean of the
    // series.  The outer vector is indexed by state model.  The inner Vector is
    // a time series.
    std::vector<Vector> state_contributions() const;

    // Returns a time series giving the contribution of state model
    // 'which_model' to the overall mean of the series being modeled.
    Vector state_contribution(int which_model) const;

    // Return true iff the model contains a regression component.
    virtual bool has_regression() const { return false; }

    // If the model contains a regression component, then return the
    // contribution of the regression model to the overall mean of y at each
    // time point.  If there is no regression component then an empty vector is
    // returned.
    virtual Vector regression_contribution() const;

    // The mean and variance of the errors from the observation equation (one
    // per time point).  These are used in MAP estimation.
    Vector observation_error_means() const;
    Vector observation_error_variances() const;

    ScalarKalmanFilter &get_filter() override;
    const ScalarKalmanFilter &get_filter() const override;
    ScalarKalmanFilter &get_simulation_filter() override;
    const ScalarKalmanFilter &get_simulation_filter() const override;

   protected:
    void simulate_forward(RNG &rng) override;

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
        int t, double observation_error_mean,
        double observation_error_variance);

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
        VectorView gradient, int t, double observation_error_mean,
        double observation_error_variance);

   private:
    // Simulate an observed Y value (minus any regression effects from a static
    // regression), conditional on the state at time t.
    double simulate_adjusted_observation(RNG &rng, int t);

    //-----------------------------------------------------------------------
    // Data begins here.
    ScalarKalmanFilter filter_;
    ScalarKalmanFilter simulation_filter_;
  };

  namespace StateSpaceUtils {

    // A helper class to manage the logical const-ness of evaluating a state
    // space model's log likelihood function.
    //
    // When called upon to evaluate log likelihood for a state space model, the
    // model's parameters are copied to a safe storage location, the new
    // parameters are injected into the model, and log likelihood is evaluated.
    // The old parameters are replaced upon completion (or if an exception is
    // thrown).
    class LogLikelihoodEvaluator {
     public:
      explicit LogLikelihoodEvaluator(const StateSpaceModelBase *model)
          : model_(const_cast<StateSpaceModelBase *>(model)) {}

      double evaluate_log_likelihood(const Vector &parameters) {
        ParameterHolder storage(model_, parameters);
        return model_->log_likelihood();
      }

      double evaluate_log_posterior(const Vector &parameters) {
        ParameterHolder storage(model_, parameters);
        double ans = model_->observation_model()->logpri();
        if (ans <= negative_infinity()) {
          return ans;
        }
        for (int s = 0; s < model_->number_of_state_models(); ++s) {
          ans += model_->state_model(s)->logpri();
          if (ans <= negative_infinity()) {
            return ans;
          }
        }
        ans += model_->log_likelihood();
        return ans;
      }

      double evaluate_log_likelihood_derivatives(
          const ConstVectorView &parameters, VectorView gradient) {
        ParameterHolder storage(model_, parameters);
        return model_->log_likelihood_derivatives(gradient);
      }

     private:
      mutable StateSpaceModelBase *model_;
    };

    // Compute one-step prediction errors on one or more holdout sets.
    //
    // Args:
    //   model:  The model to be assessed.
    //   niter:  The number of MCMC iterations.
    //   cutpoints: A set of integers giving the final time index to use in the
    //     training set.
    //   standardize: If true, then prediction errors are to be scaled by
    //     dividing by the one-step prediction standard error from the Kalman
    //     filter.  If false, then raw prediction errors are computed.
    //
    // Returns:
    //   A set of Matrices, each representing the posterior predictive
    //   distribution of the one-step prediction errors.  There is one matrix
    //   for each entry in 'cutpoints'.  Rows represent MCMC draws.  Columns
    //   represent time points.  Columns prior to the cutpoint are "in-sample"
    //   errors in the sense that the parameter estimates for the Kalman filter
    //   were learned based on that data, but they are still "out-of-sample"
    //   from the perspective of the Kalman filter (i.e. they are "filtering
    //   errors" rather than "smoothing errors").  Values after the cutpoint are
    //   "out of sample" in all senses.
    //
    // Example:
    // Suppose 'model' was fit on 200 data (time) points.
    // auto errors = compute_prediction_errors(model, 1000, {150, 175, 190}, false);
    // errors[1] is the posterior distribution of the one-step prediction errors
    // based on y[0..174].
    std::vector<Matrix> compute_prediction_errors(
        const ScalarStateSpaceModelBase &model,
        int niter,
        const std::vector<int> &cutpoints,
        bool standardize);


  }  // namespace StateSpaceUtils

}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_MODEL_BASE_HPP_
