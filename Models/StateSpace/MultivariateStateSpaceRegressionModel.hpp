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
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateModelVector.hpp"
#include "Models/StateSpace/MultivariateStateSpaceModelBase.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"

namespace BOOM {

  //===========================================================================
  // The data type represents a scalar entry in the "matrix" of time series
  // data.  By organizing the data this way, we can allow for each "data point"
  // to have its own set of regressors, which would be difficult to do if the
  // response data were an actual matrix.
  class TimeSeriesRegressionData : public RegressionData {
   public:
    // Args:
    //   y: The response variable.
    //   x: A vector of predictors.
    //   series: The identifier of the time series (0.. number of series - 1) to
    //     which this observation belongs.
    //   timestamp: The time-index of the time series (0.. sample_size - 1)
    //     containing this observation.
    TimeSeriesRegressionData(double y,
                             const Vector &x,
                             int series,
                             int timestamp);
    
    // As above, but y and x are Ptr's.  If Y and X are matrices, with the same
    // X's applying to each time series in Y, then this constructor is more
    // space efficient than the one above, because multiple Ptr's can point the
    // the same predictor vector.
    TimeSeriesRegressionData(const Ptr<DoubleData> &y,
                             const Ptr<VectorData> &x,
                             int series,
                             int timestamp);
    
    TimeSeriesRegressionData *clone() const override {
      return new TimeSeriesRegressionData(*this);
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
  // An implementation detail for MultivariateStateSpaceRegressionModel, which
  // maintains a set of ScalarKalmanFilter objects to handle simulating from
  // series-specific state.  Each of these models needs a "state space model" to
  // supply the kalman matrices and data.  This class defines a proxy state
  // space model to fill that role.  The proxy model keeps a pointer to the host
  // model from which it draws data and parameters.
  //
  // However, if there is series-specific state it is owned by the proxy.
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
  // Each can have its own prior, which can be linked by a hierarchy.
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
        public CompositeParamPolicy,
        public IID_DataPolicy<TimeSeriesRegressionData>,
        public PriorPolicy
  {
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
    MultivariateStateSpaceRegressionModel *clone() const override {
      report_error("Model cannot be copied.");
      return nullptr;
    }

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
    // Access to state models.  Access to state comes from the "grandparent"
    // base class
    //------------------------------------------------------------------------
    
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
    // data_is_finalized_.
    void add_data(const Ptr<Data> &dp) override;
    void add_data(const Ptr<TimeSeriesRegressionData> &dp) override;
    void add_data(TimeSeriesRegressionData *dp) override;

    // An override is needed so model-specific meta-data can be cleared as well.
    void clear_data() override;

    // Scalar data access.
    double response_matrix(int series, int time) const {
      finalize_data();
      return response_matrix_(series, time);
    }

    // A flag indicating whether a specific series was observed at time t.
    bool is_observed(int series, int time) const {
      finalize_data();
      return observed_(series, time);
    }
    
    // Vector data access.
    ConstVectorView observation(int t) const override {
      finalize_data();
      return response_matrix_.col(t);
    }

    const Selector &observed_status(int t) const override {
      finalize_data();
      return observed_.col(t);
    }
    
    // Returns the observed data point for the given series at the given time
    // point.  If that data point is missing, negative_infinity is returned.
    double observed_data(int series, int time) const {
      return response_matrix(series, time);
    }

    // The response value after contributions from "other models" has been
    // subtracted off.  It is the caller's responsibility to do the subtracting
    // (e.g. with isolate_shared_state() or isolate_series_specific_state()).
    double adjusted_observation(int series, int time) const {
      return adjusted_data_workspace_(series, time);
    }

    // The vector of adjusted observations across all time series at time t.
    ConstVectorView adjusted_observation(int time) const override {
      return adjusted_data_workspace_.col(time);
    }
   
    //--------------------------------------------------------------------------
    // Kalman filter parameters.
    //--------------------------------------------------------------------------
    const SparseKalmanMatrix *observation_coefficients(
        int t, const Selector &observed) const override;
    
    DiagonalMatrix observation_variance(int t) const override;
    
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
    &state_model_vector() override { return shared_state_models_; }
    
    const StateSpaceUtils::StateModelVector<SharedStateModel>
    &state_model_vector() const override { return shared_state_models_; }

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

   private:
    // To be called after add_data has been called for the last time.
    // This method is logically const so that it can be called by accessors.
    void finalize_data() const;
    
    // Populate the vector of proxy models with 'nseries_' empty models.
    void initialize_proxy_models();

    // Set observers on the variance parameters of the regression models, so
    // that the diagonal variance matrix can be updated when it gets out of
    // sync.
    void set_observation_variance_observers();
    
    // If the observation variance is out of step with the observation_variance_
    // data member, update the data member.  This function is logically const.
    void update_observation_variance() const;

    void resize_subordinate_state();
    void observe_state(int t) override;
    void observe_initial_state();
    void observe_data_given_state(int t) override;
    
    using ConditionallyIndependentMultivariateStateSpaceModelBase::get_filter;

    void impute_missing_observations(int t, RNG &rng) override;
    void impute_shared_state_given_series_state(RNG &rng);
    void impute_series_state_given_shared_state(RNG &rng);
    
    // Sets adjusted_data_workspace_ to observed_data minus contributions from
    // series specific state.
    void isolate_shared_state();

    // Sets adjusted_data_workspace_ to observed_data minus contributions from
    // shared state.
    void isolate_series_specific_state();

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

    // The observation coefficients from the shared state portion of the model.
    // This does not include the regression coefficients from the regression
    // model, nor does it include the series-specific state.
    mutable Ptr<StackedMatrixBlock> observation_coefficients_;
    
    // The response matrix organizes all the scalar responses from each data
    // point.  Time flows horizontally, so each column is a single time point.
    mutable Matrix response_matrix_;
    mutable SelectorMatrix observed_;

    // A flag that gets set to false each time a new data point is added.  This
    // flag is checked each time observed_data_ is called.
    mutable bool data_is_finalized_;

    // A workspace where observed data can be modified by subtracting off
    // components on which we wish to condition.
    Matrix adjusted_data_workspace_;

    enum WorkspaceStatus {
      UNSET,
      SHOWS_SHARED_EFFECTS,
      SHOWS_SERIES_EFFECTS
    };
    WorkspaceStatus workspace_status_;
    
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

