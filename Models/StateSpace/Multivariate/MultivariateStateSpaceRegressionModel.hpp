#ifndef BOOM_MULTIVARIATE_STATE_SPACE_REGRESSION_HPP_
#define BOOM_MULTIVARIATE_STATE_SPACE_REGRESSION_HPP_
/*
  Copyright (C) 2019 Steven L. Scott

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

#include "Models/IndependentMvnModel.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/Glm/IndependentRegressionModels.hpp"
#include "Models/StateSpace/Multivariate/StateModels/SharedStateModel.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateModelVector.hpp"
#include "Models/StateSpace/Multivariate/MultivariateStateSpaceModelBase.hpp"

namespace BoomStateSpaceTesting{
  class MultivariateStateSpaceRegressionModelTest;
}

namespace BOOM {

  //===========================================================================
  // A scalar response value, paired with a set of predictor variables, at a
  // given point in time.
  class MultivariateTimeSeriesRegressionData : public RegressionData {
   public:
    // Args:
    //   y: The response variable.
    //   x: A vector of predictors.
    //   series: The identifier of the time series (0.. number of series - 1) to
    //     which this observation belongs.
    //   timestamp: The time-index of the time series (0.. sample_size - 1)
    //     containing this observation.
    MultivariateTimeSeriesRegressionData(double y,
                                         const Vector &x,
                                         int series,
                                         int timestamp);

    // As above, but y and x are Ptr's.  If Y and X are matrices, with the same
    // X's applying to each time series in Y, then this constructor is more
    // space efficient than the one above, because multiple Ptr's can point the
    // the same predictor vector.
    MultivariateTimeSeriesRegressionData(const Ptr<DoubleData> &y,
                                         const Ptr<VectorData> &x,
                                         int series,
                                         int timestamp);

    MultivariateTimeSeriesRegressionData *clone() const override {
      return new MultivariateTimeSeriesRegressionData(*this);
    }

    // The index of the time series to which this data point corresponds.  If
    // you think about a multivariate time series as a matrix, with rows
    // representing time, this is the column identifier.
    int series() const {return which_series_;}

    // The time-index of the time series to which this data point belongs.  If
    // you think about a multivariate time series as a matrix, with rows
    // representing time, this is the run number.
    int timestamp() const {return timestamp_index_;}

    std::ostream &display(std::ostream &out) const override {
      out << "series " << which_series_ << "\n"
          << "time   " << timestamp_index_ << "\n";
      return RegressionData::display(out);
    }

   private:
    int which_series_;
    int timestamp_index_;
  };

  //===========================================================================
  // That MultivariateStateSpaceRegressionModel maintains a set of
  // ScalarKalmanFilter objects to handle simulating from series-specific state.
  // Each of these series-specific filters needs a "state space model" to supply
  // the kalman matrices and data.  This class is a proxy state space model
  // filling that role.  The proxy model keeps a pointer to the host model from
  // which it draws data and parameters.  The proxy also assumes ownership of
  // any series-specific state.
  class MultivariateStateSpaceRegressionModel;
  class ProxyScalarStateSpaceModel : public StateSpaceModel {
   public:
    // Args:
    //   model:  The host model.
    //   which_series: The index of the time series that this object describes.
    ProxyScalarStateSpaceModel(MultivariateStateSpaceRegressionModel *model,
                               int which_series);

    // The number of distinct time points in the host model.
    int time_dimension() const override;

    // The value of the time series specific to this proxy.  The host should
    // have subtracted any regression effects or shared state before this
    // function is called.
    double adjusted_observation(int t) const override;

    bool is_missing_observation(int t) const override;

    ZeroMeanGaussianModel *observation_model() override { return nullptr; }
    const ZeroMeanGaussianModel *observation_model() const override {
      return nullptr;
    }

    double observation_variance(int t) const override;

    // Because the proxy model has no observation model,
    // observe_data_given_state is a no-op.
    void observe_data_given_state(int t) override {}

    // Simulate 'horizon' time periods beyond time_dimension().
    //
    // Args:
    //   rng: The [0, 1) random number generator to use for the simulation.
    //   horizon: The number of periods beyond 'time_dimension()' to simulate.
    //   final_state:  The value of the state at time time_dimension() - 1.
    //
    // Returns:
    //   A draw from the predictive distribution of the state contribution over
    //   the next 'horizon' time periods.
    Vector simulate_state_contribution_forecast(
        RNG &rng, int horizon, const Vector &final_state);

    // Ensure all state models are capable of handling times up to t-1.
    void observe_time_dimension(int t) {
      for (int s = 0; s < number_of_state_models(); ++s) {
        state_model(s)->observe_time_dimension(t);
      }
    }

   private:
    // The add_data method is disabled.
    void add_data(const Ptr<StateSpace::MultiplexedDoubleData>
                  &data_point) override;
    void add_data(const Ptr<Data> &data_point) override;

    MultivariateStateSpaceRegressionModel *host_;
    int which_series_;
  };

  //===========================================================================
  // A multivariate state space regression model describes a fixed dimensional
  // vector Y[t] as it moves throughout time.  The model is a state space model
  // of the form
  //
  //        Y[t] = Z[t] * alpha[t] + B * X[t] + epsilon[t]
  //  alpha[t+1] = T[t] * alpha[t] + R[t] * eta[t].
  //
  // The state alpha[t] consists of two types: shared and series-specific.  A
  // shared state component is a regular state component from a dynamic factor
  // model, with a matrix Z[t] mapping state to outcomes.  A series specific
  // model maintains a separate element of state for each dimension of Y[t].
  //
  // The learning algorithm can cycle between (draw shared state given data and
  // series-specific state), (draw series-specific state), and (draw parameters
  // given complete data).
  //
  // The model assumes that errors from each state component are independent of
  // other state components (given model parameters), and that the observation
  // errors epsilon[t] are conditionally independent of everything else given
  // state and model parameters.  Both eta[t] and epsilon[t] are Gaussian.  This
  // model makes the further simplifying assumption that Var(epsilon[t]) is
  // diagonal, so that any cross sectional correlations between elements of Y[t]
  // are captured by shared state.
  //
  // Thus epsilon[t] ~ N(0, diag(sigma^2)).  There is a different sigma^2 for
  // each series, but the off-diagonal elements are all zero.  Internally this
  // means the regression is handled by nseries() separate regression models.
  // Each can have its own prior, which can be linked by a hierarchy.  If there
  // is a model hierarchy, it is to be maintained by the PosteriorSampler.
  //
  //---------------------------------------------------------------------------
  // The basic usage idiom is
  // NEW(MultivariateStateSpaceRegressionModel, model)(xdim, ydim);
  // for() { model->add_data(data_point); }
  // model->add_state(shared_state_model_1);
  // model->add_state(shared_state_model_2);
  // ...
  // model->series_specific_model(0)->add_state(series_specific_state_model_11);
  // model->series_specific_model(0)->add_state(series_specific_state_model_12);
  // model->series_specific_model(1)->add_state(series_specific_state_model_21);
  // model->observation_model()->set_method(prior_for_regression_part);
  //
  // The posterior samplers for the individual state models must be set
  // separately.  Likewise for the samplers for the regression models.  If
  // (e.g.) a hierarchical regression is desired then that is a new posterior
  // sampler class for IndependentRegressionModels.
  class MultivariateStateSpaceRegressionModel
      : public ConditionallyIndependentMultivariateStateSpaceModelBase,
        public IID_DataPolicy<MultivariateTimeSeriesRegressionData>,
        public PriorPolicy
  {
    friend class BoomStateSpaceTesting::MultivariateStateSpaceRegressionModelTest;
   public:
    // Args:
    //   xdim:  The dimension of the static regression component.
    //   nseries:  The number of time series being modeled.
    explicit MultivariateStateSpaceRegressionModel(int xdim, int nseries);

    // This is a complex model with lots of subordinate parts.  Copying it
    // correctly would be really hard, so copying is disallowed.
    MultivariateStateSpaceRegressionModel(
        const MultivariateStateSpaceRegressionModel &rhs) = delete;
    MultivariateStateSpaceRegressionModel &operator=(
        const MultivariateStateSpaceRegressionModel &rhs) = delete;
    MultivariateStateSpaceRegressionModel(
        MultivariateStateSpaceRegressionModel &&rhs) = delete;
    MultivariateStateSpaceRegressionModel &operator=(
        MultivariateStateSpaceRegressionModel &&rhs) = delete;

    // An error will be reported if someone attempts to clone this model.
    MultivariateStateSpaceRegressionModel *clone() const override;
    MultivariateStateSpaceRegressionModel *deepclone() const override;

    // Simulate a multi-period forecast.
    // Args:
    //   rng:  The [0, 1) random number generator to use for the simulation.
    //   forecast_predictors: A matrix containing the predictor variables to use
    //     for the forecast.  The number of rows in the matrix is nseries() *
    //     forecast_horizon.  The rows are assumed to go (time0, series0),
    //     (time0, series1), ..., (time1, series0), (time1, series1), ...
    //   final_shared_state: The vector of shared state as of the final training
    //     data point at time_dimension() - 1.
    //   series_specific_final_state: This argument can be empty if there is no
    //     series-specific state in the model.  Otherwise, it must have length
    //     nseries(), where each element is the series specific state vector at
    //     time time_dimension() - 1 for the corresponding series.  Individual
    //     elements can be empty if there is no series-specific state for that
    //     series.
    //
    // Returns:
    //   A matrix containing draws of the next 'forecast_horizon' time periods.
    //   Each series corresponds to a row in the returned matrix, while columns
    //   correspond to time.  The simulation includes simulated residual error.
    Matrix simulate_forecast(
        RNG &rng,
        const Matrix &forecast_predictors,
        const Vector &final_shared_state,
        const std::vector<Vector> &series_specific_final_state);

    //------------------------------------------------------------------------
    // Access to state models.  Access to state comes from the
    // MultivariateStateSpaceModelBase "grandparent" base class.
    // ------------------------------------------------------------------------

    // Add state to the "shared-state" portion of the state space.
    void add_state(const Ptr<SharedStateModel> &state_model);

    // Add state to the state model for an individual time series.
    //
    // Args:
    //   state_model:  The state model defining the state to be added.
    //   series:  The index of the scalar time series described by the state.
    void add_series_specific_state(const Ptr<StateModel> &state_model,
                                   int series) {
      proxy_models_[series]->add_state(state_model);
    }

    // Indicates whether any of the proxy models have had state assigned.
    bool has_series_specific_state() const;

    // Dimension of shared state.
    int state_dimension() const override {
      return shared_state_models_.state_dimension();
    }

    // The dimension of the series-specific state associated with a particular
    // time series.
    int series_state_dimension(int which_series) const {
      if (proxy_models_.empty()) {
        return 0;
      } else {
        return proxy_models_[which_series]->state_dimension();
      }
    }

    int number_of_state_models() const override {
      return shared_state_models_.size();
    }

    SharedStateModel *state_model(int s) override {
      if (s < 0 || s >= shared_state_models_.size()) {
        return nullptr;
      } else {
        return shared_state_models_[s].get();
      }
    }

    const SharedStateModel *state_model(int s) const override {
      return shared_state_models_[s].get();
    }

    // Impute both the shared and series-specific state, each conditional on the
    // other.
    void impute_state(RNG &rng) override;

    //-----------------------------------------------------------------------
    // Data policy overrides, and access to raw data.
    //-----------------------------------------------------------------------

    // The number of time points that have been observed.
    int time_dimension() const override {return time_dimension_;}

    // The number of time series being modeled.
    int nseries() const override {return nseries_;}

    // The dimension of the predictors.
    int xdim() const {return observation_model_->xdim();}

    // Adding data to this model adjusts time_dimension_, data_indices_, and
    // observed_.
    void add_data(const Ptr<Data> &dp) override;
    void add_data(const Ptr<MultivariateTimeSeriesRegressionData> &dp) override;
    void add_data(MultivariateTimeSeriesRegressionData *dp) override;

    // Return the position in the data vector containing the Y value for the
    // given series at the given time.  If no data point exists for the
    // requested (series, time) pair then -1 is returned.
    int data_index(int series, int time) const;

    // An override is needed so model-specific meta-data can be cleared as well.
    void clear_data() override;

    // Scalar data access.
    double response(int series, int time) const {
      int index = data_index(series, time);
      if (index < 0) {
        return negative_infinity();
      } else {
        return dat()[index]->y();
      }
    }

    // A flag indicating whether a specific series was observed at time t.
    bool is_observed(int series, int time) const {
      return observed_[time][series];
    }

    // Vector data access.
    ConstVectorView observation(int t) const override {
      const Selector &observed(observed_[t]);
      response_workspace_.resize(observed.nvars());
      for (int i = 0; i < observed.nvars(); ++i) {
        int series = observed.sparse_index(i);
        response_workspace_[i] = response(series, t);
      }
      return ConstVectorView(response_workspace_);
    }

    const Selector &observed_status(int t) const override {
      return observed_[t];
    }

    // Set the observation status for the data at time t.
    void set_observed_status(int t, const Selector &status);

    // Returns the observed data point for the given series at the given time
    // point.  If that data point is missing, negative_infinity is returned.
    double observed_data(int series, int time) const {
      return response(series, time);
    }

    // The response value after contributions from "other models" have been
    // subtracted off.  It is the caller's responsibility to do the subtracting
    // (e.g. with isolate_shared_state() or isolate_series_specific_state()).
    double adjusted_observation(int series, int time) const;

    // The vector of adjusted observations across all time series at time t.
    ConstVectorView adjusted_observation(int time) const override;

    void isolate_shared_state() override;
    void isolate_series_specific_state() override;

    //--------------------------------------------------------------------------
    // Kalman filter parameters.
    //--------------------------------------------------------------------------

    // The observation coefficients from the shared state portion of the model.
    // This does not include the regression coefficients from the regression
    // model, nor does it include the series-specific state.
    Ptr<SparseKalmanMatrix> observation_coefficients(
        int t, const Selector &observed) const override;

    DiagonalMatrix observation_variance(int t) const override;
    DiagonalMatrix observation_variance(
        int t, const Selector &observed) const override;

    double single_observation_variance(int t, int dim) const override {
      return observation_model_->model(dim)->sigsq();
    }

    Ptr<ProxyScalarStateSpaceModel> series_specific_model(int index) {
      return proxy_models_[index];
    }

    const Ptr<ProxyScalarStateSpaceModel>
    series_specific_model(int index) const {
      return proxy_models_[index];
    }

    IndependentRegressionModels *observation_model() override {
      return observation_model_.get();
    }

    const IndependentRegressionModels *observation_model() const override {
      return observation_model_.get();
    }

    // The contribution of a particular state model to the mean of the response.
    //
    // Args:
    //   which_state_model:  The index of the desired state model.
    //
    // Returns:
    //   A matrix with rows corresponding to dimension of Y, and columns
    //   corresponding to time.
    Matrix state_contributions(int which_state_model) const override;

    StateSpaceUtils::StateModelVector<SharedStateModel>
    &state_models() override { return shared_state_models_; }

    const StateSpaceUtils::StateModelVector<SharedStateModel>
    &state_models() const override { return shared_state_models_; }

    // Ensure that all state and proxy models are aware of times up to time 't'.
    void observe_time_dimension(int t) {
      for (int s = 0; s < number_of_state_models(); ++s) {
        state_model(s)->observe_time_dimension(t);
      }
      for (int m = 0; m < proxy_models_.size(); ++m) {
        if (!!proxy_models_[m]) {
          proxy_models_[m]->observe_time_dimension(t);
        }
      }
    }

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
    double mle(double epsilon = 1e-5, int ntries = 500) override;

    // Returns true if all the state models have been assigned priors that
    // implement find_posterior_mode.
    bool check_that_em_is_legal() const;

    double Estep(bool save_state_distributions);
    void Mstep(double epsilon);

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
    // void update_observation_model(Vector &r, SpdMatrix &N, int t,
    //                               bool save_state_distributions,
    //                               bool update_sufficient_statistics,
    //                               Vector *gradient);

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
    //   The log likelihood value computed by the Kalman filter.
    double average_over_latent_data(bool update_sufficient_statistics,
                                    bool save_state_distributions,
                                    Vector *gradient);

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
    //   observation_error_variances: The posterior variance of the observation
    //     error at time t.
    void update_observation_model_gradient(
        VectorView gradient,
        int t,
        const Vector &observation_error_mean,
        const Vector &observation_error_variances) override;

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
    void update_observation_model_complete_data_sufficient_statistics(
        int t,
        const Vector &observation_error_mean,
        const Vector &observation_error_variances) override;

    using ConditionallyIndependentMultivariateStateSpaceModelBase::get_filter;

   private:
    // Populate the vector of proxy models with 'nseries_' empty models.
    void initialize_proxy_models();

    // Set observers on the variance parameters of the regression models, so
    // that the diagonal variance matrix can be updated when it gets out of
    // sync.
    void set_observation_variance_observers();

    // Ditto for adjusted_data_workspace_.
    void set_workspace_observers();

    // To be called during construction.  Set an observer on the parameters of
    // *model that will invalidate the current kalman filter likelihood
    // calculation when the parameters change.
    void set_parameter_observers(Model *model);

    // If the observation variance is out of step with the observation_variance_
    // data member, update the data member.  This function is logically const.
    void update_observation_variance() const;

    void resize_subordinate_state();
    void observe_state(int t) override;
    void observe_initial_state();
    void observe_data_given_state(int t) override;

    //
    void impute_missing_observations(int t, RNG &rng) override;

    void impute_shared_state_given_series_state(RNG &rng);
    void impute_series_state_given_shared_state(RNG &rng);

    // Sets adjusted_data_workspace_ to observed_data minus contributions from
    // series specific state.
    // void isolate_shared_state();
    void isolate_shared_state(int time) const;

    // Sets adjusted_data_workspace_ to observed_data minus contributions from
    // shared state.
    //    void isolate_series_specific_state();
    void isolate_series_specific_state(int time) const;

    // The contribution of the series_specific state to the given series at the
    // given time.
    double series_specific_state_contribution(int series, int time) const;

    //--------------------------------------------------------------------------
    // Data section.
    //--------------------------------------------------------------------------

    // The number of series being modeled.
    int nseries_;

    // The time dimension is the number of distinct time points.
    int time_dimension_;

    // The shared state models are stored in this container.  The series
    // specific state models are stored in proxy_models_.
    StateSpaceUtils::StateModelVector<SharedStateModel> shared_state_models_;

    // The proxy models hold components of state that are specific to individual
    // data series.
    std::vector<Ptr<ProxyScalarStateSpaceModel>> proxy_models_;

    // data_indices_[series][time] gives the index of the corresponding element
    // of dat().
    std::map<int, std::map<int, int>> data_indices_;

    // The observation model.
    Ptr<IndependentRegressionModels> observation_model_;

    // mutable Ptr<StackedMatrixBlock> observation_coefficients_;

    // The response matrix organizes all the scalar responses from each data
    // point.  Time flows horizontally, so each column is a single time point.
    mutable Vector response_workspace_;
    //     mutable SelectorMatrix observed_;

    // observed_[t] indicates which time series are observed at time t.
    std::vector<Selector> observed_;

    // A workspace where observed data can be modified by subtracting off
    // components on which we wish to condition.
    //
    // The point of having this workspace, as opposed to simply making the
    // adjustments on demand, is that the adjustments are best performed on a
    // time-by-time basis, but the model also supports a (series, time)
    // interface.  To make the latter more efficient we make adjustments on the
    // time basis, store the results, and then look up the (series, time)
    // answer.
    mutable Vector adjusted_data_workspace_;

    // Metadata about the adjusted_data_workspace_.
    //
    // A flag indicating that the workspace holds current values.  This is set
    // to false whenever new parameters are assigned or new state is drawn.
    mutable bool workspace_current_;

    // The time index that the workspace currently describes.
    mutable int workspace_time_index_;

    // The type of information currently stored in the workspace.
    enum WorkspaceStatus {
      UNSET,
      ISOLATE_SHARED_STATE,
      ISOLATE_SERIES_SPECIFIC_STATE,
      ISOLATE_REGRESSION_EFFECTS
    };
    mutable WorkspaceStatus workspace_status_;

    // A workspace to copy the residual variances stored in observation_model_
    // in the data structure expected by the model.
    mutable DiagonalMatrix observation_variance_;

    // A flag to keep track of whether the observation variance is current.
    mutable bool observation_variance_current_;

    // A Selector of size nseries() with all elements included.  Useful for
    // calling observation_coefficients when you want to assume all elements are
    // included.
    Selector dummy_selector_;
  };

}  // namespace BOOM

#endif  // BOOM_MULTIVARIATE_STATE_SPACE_REGRESSION_HPP_
