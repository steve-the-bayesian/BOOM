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
  //
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // NOTE: This class is VERY, VERY similar to StateSpaceModelBase, but there
  // are some important differences.  The structuring of state() into
  // shared_state and series specific state is one example.  At some point in
  // the future, we should consider merging MultivariateStateSpaceModelBase and
  // StateSpaceModelBase, but that will require substantial effort, and should
  // not be undertaken lightly.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  class MultivariateStateSpaceModelBase
      : virtual public Model {
   public:
    MultivariateStateSpaceModelBase()
        : state_is_fixed_(false),
          show_warnings_(true),
          observation_model_parameter_size_(-1)
    {}

    MultivariateStateSpaceModelBase *clone() const override = 0;
    virtual MultivariateStateSpaceModelBase *deepclone() const = 0;
    MultivariateStateSpaceModelBase & operator=(
        const MultivariateStateSpaceModelBase &rhs);

    // The model does not use a parameter policy, so we need to override the
    // parameter_vector() members here.
    //
    // The order of the parameter vectors is (a) observation model, (b) shared
    // state_models.
    //
    // Child classes need to overload this method if they include series
    // specific state models.
    std::vector<Ptr<Params>> parameter_vector() override;
    const std::vector<Ptr<Params>> parameter_vector() const override;

    virtual int time_dimension() const = 0;

    // The number of time series being modeled.  Not all model types know this.
    // For example the number of series in a dynamic intercept model changes
    // from time point to time point.  For this reason the default
    // implementation is to return -1.
    virtual int nseries() const { return -1; }

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

    void kalman_filter() {get_filter().update();}
    void kalman_smoother() {get_filter().smooth();}

    // Return the state mean from the Kalman filter/smoother.  This function
    // does no filtering or smoothing itself.  It just returns the state_mean()
    // value for each node in the kalman filter.
    Matrix state_mean() const;

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
    //     observed (i.e. not missing).
    //
    // Returns:
    //   A subset of the observation coefficients at time t.  Row j of Z[t] is
    //   included if and only if observed[j] == true.
    virtual Ptr<SparseKalmanMatrix> observation_coefficients(
        int t, const Selector &observed) const = 0;

    // For testing.
    virtual SpdMatrix dense_observation_variance(
        int t, const Selector &observed) const = 0;

    // Return the KalmanFilter object responsible for filtering the data.
    virtual MultivariateKalmanFilterBase & get_filter() = 0;
    virtual const MultivariateKalmanFilterBase & get_filter() const = 0;
    virtual MultivariateKalmanFilterBase & get_simulation_filter() = 0;
    virtual const MultivariateKalmanFilterBase & get_simulation_filter() const = 0;

    // Returns the log likelihood under the current set of model parameters.  If
    // the Kalman filter is current (i.e. no parameters or data have changed
    // since the last time it was run) then this function does no actual work.
    // Otherwise it sparks a fresh Kalman filter run.
    double log_likelihood() {
      return get_filter().compute_log_likelihood();
    }

    // Set model parameters to their maximum likelihood estimates.
    //
    // Args:
    //   epsilon: A small positive number.  Absolute changes to log likelihood
    //     less than this value indicate that the algorithm has converged.
    //   max_tries:  Stop trying to optimzize after this many iterations.
    //
    // Returns:
    //   The log likelihood value at the maximum.
    //
    // Effects:
    //   Model parameters are set to the maximum likelihood estimates.
    //
    // This function is virtual so that child classes are free to handle edge
    // cases in the optimization.  One such case is the potential presence of
    // series-specific effects.
    virtual double mle(double epsilon, int max_tries=500);

    //------------- Model matrices for structural equations. --------------
    // Durbin and Koopman's T[t] built from state models.
    virtual SparseKalmanMatrix *state_transition_matrix(int t) const {
      return state_models().state_transition_matrix(t);
    }

    // Durbin and Koopman's RQR^T.  Built from state models, often less than
    // full rank.
    virtual SparseKalmanMatrix *state_variance_matrix(int t) const {
      return state_models().state_variance_matrix(t);
    }

    // Durbin and Koopman's R matrix from the transition equation:
    //    state[t+1] = (T[t] * state[t]) + (R[t] * state_error[t]).
    //
    // This is the matrix that takes the low dimensional state_errors and turns
    // them into error terms for states.
    virtual ErrorExpanderMatrix *state_error_expander(int t) const {
      return state_models().state_error_expander(t);
    }

    // The full rank variance matrix for the errors in the transition equation.
    // This is Durbin and Koopman's Q[t].  The errors with this variance are
    // multiplied by state_error_expander(t) to produce the errors described by
    // state_variance_matrix(t).
    virtual SparseKalmanMatrix *state_error_variance(int t) const {
      return state_models().state_error_variance(t);
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
    //
    // Only the elements with observed_status == true are returned.  The
    // dimension of the return value is thus observed_status(time).nvars().
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

    const Matrix &shared_state() const { return shared_state_; }
    void set_shared_state(const Matrix &shared_state) {
      shared_state_ = shared_state;
    }

    ConstVectorView shared_state(int t) const {return shared_state().col(t);}

    // Control the type of state the 'adjusted_observation' refers to.  Not all
    // models support this distinction, so default implementations are no-ops.
    virtual void isolate_shared_state() {};
    virtual void isolate_series_specific_state() {};

    Vector initial_state_mean() const;
    SpdMatrix initial_state_variance() const;

    // Set the shared state to the specified value, and mark the state as
    // 'fixed' so that it will no longer be updated by calls to 'impute_state'.
    // This function is intended for debugging purposes only.
    //
    // Args:
    //   state:  The state matrix.  Columns are time. Rows are state elements.
    void permanently_set_state(const Matrix &state);

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
      return state_models().state_component(state, s);
    }
    VectorView state_component(VectorView &state, int s) const {
      return state_models().state_component(state, s);
    }
    ConstVectorView state_component(const ConstVectorView &state, int s) const {
      return state_models().state_component(state, s);
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
    ConstVectorView const_state_error_component(
        const Vector &full_state_error, int state_model_number) const {
      return state_models().const_state_error_component(
          full_state_error, state_model_number);
    }
    VectorView state_error_component(
        Vector &full_state_error, int state_model_number) const {
      return state_models().state_error_component(
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
      return state_models().state_error_variance_component(
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
      return state_models().full_state_subcomponent(shared_state(), state_model_index);
    }
    SubMatrix mutable_full_state_subcomponent(int state_model_index) {
      return state_models().mutable_full_state_subcomponent(
          shared_state_, state_model_index);
    }

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

    bool show_warnings() const {return show_warnings_;}
    void show_warnings(bool should_warnings_be_shown) {
      show_warnings_ = should_warnings_be_shown;
    }

   protected:
    // Access to the state model vector owned by descendents.
    using StateModelVectorBase = StateSpaceUtils::StateModelVectorBase;

    virtual StateModelVectorBase & state_models() = 0;
    virtual const StateModelVectorBase & state_models() const = 0;

    // Remove any posterior sampling methods from this model and all client
    // models.  Copy posterior samplers from rhs to *this.
    void copy_samplers(const MultivariateStateSpaceModelBase &rhs);

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

    void clear_client_data();
    void observe_fixed_state();

    // Ensure that shared_state_ has state_dimension() rows and time_dimension()
    // columns.
    void resize_state();

   private:
    // Implementation for impute_state.
    void simulate_initial_state(RNG &rng, VectorView initial_state) const;
    Vector simulate_state_error(RNG &rng, int t) const;

    void simulate_forward(RNG &rng);
    void propagate_disturbances(RNG &rng);


    // Simulate a fake observation to use as part of the Durbin-Koopman state
    // simulation algorithm.  If observed_status(t) is less than fully observed,
    // only the observed parts should be simulated.
    virtual Vector simulate_fake_observation(RNG &rng, int t) = 0;

    Matrix shared_state_;
    bool state_is_fixed_;

    bool show_warnings_;

    mutable int observation_model_parameter_size_;
  };

  //===========================================================================
  // A GeneralMultivariateStateSpaceModelBase is a
  // MultivariateStateSpaceModelBase with an observation error variance that is
  // a SpdMatrix.  This setting provides a general form for fitting models, but
  // does not scale to large numbers of series.
  class GeneralMultivariateStateSpaceModelBase
      : public MultivariateStateSpaceModelBase {
   public:
    virtual SpdMatrix observation_variance(int t) const = 0;
    virtual SpdMatrix observation_variance(
        int t, const Selector &observed) const = 0;
    SpdMatrix dense_observation_variance(
        int t, const Selector &observed) const override {
      return observed.select(observation_variance(t));
    }

    //---------------- Prediction, filtering, smoothing ---------------
    // Run the full Kalman filter over the observed data, saving the information
    // in the filter_ object.  The log likelihood is computed as a by-product.
    // void kalman_filter() override;

    void update_observation_model(Vector &r, SpdMatrix &N, int t,
                                  bool save_state_distributions,
                                  bool update_sufficient_statistics,
                                  Vector *gradient);
  };

  //===========================================================================
  // A ConditionallyIndependentMultivariateStateSpaceModelBase is a
  // MultivariateStateSpaceModelBase where, conditional on latent factors, the
  // various series in the model are independent of one another.  The assumption
  // is that the factor structure captures any structural dependence, so the
  // observation errors are independent.
  class ConditionallyIndependentMultivariateStateSpaceModelBase
      : public MultivariateStateSpaceModelBase {
   public:
    ConditionallyIndependentMultivariateStateSpaceModelBase()
        : filter_(this),
          simulation_filter_(this)
    {}

    // Variance of the observation error at time t.  Durbin and Koopman's H[t].
    // This matrix includes elements for both missing and observed data.  If you
    // just want the matrix for observed data, use the interface with the
    // Selector as a second argument.
    virtual DiagonalMatrix observation_variance(int t) const = 0;

    // Args:
    //   t:   The time index.
    //   observed:  The subset of time series for which variances are desired.
    //
    // Returns:
    //   A DiagonalMatrix containing the variance of the observation error at
    //   time t for the observed subset of of the data.
    virtual DiagonalMatrix observation_variance(
        int t, const Selector &observed) const = 0;

    // The observation variance, for the requested subset of the data, at a
    // given time point, as an SpdMatrix.
    SpdMatrix dense_observation_variance(
        int t, const Selector &observed) const override {
      SpdMatrix ans(observed.nvars(), 1.0);
      ans.diag() = observation_variance(t, observed).diag();
      return ans;
    }

    // Args:
    //   t:  The index of a time point.
    //   dim:  The index of a specific time series.
    //
    // Returns:
    //   The residual variance of the specified time series at the requested
    //   time point.  If the concrete model is a mixture of Gaussians, then the
    //   returned variance is conditional on the latent mixing variables.
    //
    // This method is needed to implement proxy models that handle
    // series-specific state.
    virtual double single_observation_variance(int t, int dim) const = 0;

    //---------------- Prediction, filtering, smoothing ---------------
    // Run the full Kalman filter over the observed data, saving the information
    // in the filter_ object.  The log likelihood is computed as a by-product.
    // void kalman_filter() override { filter_.update(); }

    using Filter = ConditionallyIndependentKalmanFilter;
    Filter &get_filter() override {return filter_;}
    const Filter &get_filter() const override { return filter_; }
    Filter &get_simulation_filter() override { return simulation_filter_; }
    const Filter & get_simulation_filter() const override {
      return simulation_filter_;
    }

    void update_observation_model(Vector &r, SpdMatrix &N, int t,
                                  bool save_state_distributions,
                                  bool update_sufficient_statistics,
                                  Vector *gradient);

    // For models that have a "sigma_squared" parameter (like Gaussian and
    // Student T), the return value is a series-specific sigma squared's.
    //
    // For models that don't have natural scale parameters (e.g. Poisson, logit,
    // probit), the return value is a vector of 1's.
    virtual Vector observation_variance_parameter_values() const = 0;

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
        const Vector &observation_error_variances) = 0;

    virtual void update_observation_model_gradient(
        VectorView gradient,
        int t,
        const Vector &observation_error_mean,
        const Vector &observation_error_variances) = 0;

   private:
    // This function is
    Vector simulate_fake_observation(RNG &rng, int t) override;

    ConditionallyIndependentKalmanFilter filter_;
    ConditionallyIndependentKalmanFilter simulation_filter_;
  };

  //===========================================================================
  // ConditionalIidMultivariateStateSpaceModelBase is a
  // MultivariateStateSpaceModelBase where, conditional on latent factors the
  // observed series all have the same scalar variance.
  class ConditionalIidMultivariateStateSpaceModelBase
      : public MultivariateStateSpaceModelBase {
   public:
    ConditionalIidMultivariateStateSpaceModelBase();

    // All observations at time t have this variance.
    virtual double observation_variance(int t) const = 0;
    SpdMatrix dense_observation_variance(
        int t, const Selector &observed) const override {
      return SpdMatrix(observed.nvars(), observation_variance(t));
    }

    // Run the full Kalman filter over the observed data, saving the information
    // in the filter_ object.  The log likelihood is computed as a by-product.
    // void kalman_filter() override { filter_.update(); }

    ConditionalIidKalmanFilter &get_filter() override;
    const ConditionalIidKalmanFilter &get_filter() const override;
    ConditionalIidKalmanFilter &get_simulation_filter() override;
    const ConditionalIidKalmanFilter &get_simulation_filter() const override;

    void update_observation_model(Vector &r, SpdMatrix &N, int t,
                                  bool save_state_distributions,
                                  bool update_sufficient_statistics,
                                  Vector *gradient);

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

#endif  //  BOOM_MULTIVARIATE_STATE_SPACE_MODEL_BASE_HPP_
