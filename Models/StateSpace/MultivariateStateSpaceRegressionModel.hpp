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
  // Store y, x_series, x_shared, series_id, timestamp?
  class TimeSeriesRegressionData : public RegressionData {
   public:
    TimeSeriesRegressionData(double y, const Vector &x, int series,
                             int timestamp);
    TimeSeriesRegressionData(const Ptr<DoubleData> &y,
                             const Ptr<VectorData> &x,
                             int series,
                             int timestamp);

    TimeSeriesRegressionData(const TimeSeriesRegressionData &rhs);
    TimeSeriesRegressionData &operator=(
        const TimeSeriesRegressionData &rhs);
    
    TimeSeriesRegressionData(TimeSeriesRegressionData &&rhs) = default;
    TimeSeriesRegressionData &operator=(
        TimeSeriesRegressionData &&rhs) = default;
    TimeSeriesRegressionData *clone() const override = 0;

    int series() const {return which_series_;}
    int timestamp() const {return timestamp_index_;}
    
   private:
    int which_series_;
    int timestamp_index_;
  };

  //===========================================================================
  // This class is an implementation detail for
  // MultivariateStateSpaceRegressionModel, which maintains a set of
  // ScalarKalmanFilter objects to handle simulating from series-specific state.
  // Each of these models needs a "state space model" to supply the kalman
  // matrices and data.  This class defines a "fake" state space model that can
  // fill that role.
  class MultivariateStateSpaceRegressionModel;
  class ProxyScalarStateSpaceModel : public StateSpaceModel {
   public:
    // Args:
    //   model:  The host model.
    //   which_series: The series in the host model that this object describes.
    ProxyScalarStateSpaceModel(MultivariateStateSpaceRegressionModel *model,
                               int which_series);
      
    int time_dimension() const override;
    
    double adjusted_observation(int t) const override;
    bool is_missing_observation(int t) const override;
    
   private:
    // Disabling add_data.
    void add_data(const Ptr<StateSpace::MultiplexedDoubleData> &data_point) override;
    void add_data(const Ptr<Data> &data_point) override;
    
    MultivariateStateSpaceRegressionModel *model_;
    int which_series_;
  };

  //===========================================================================
  // A multivariate state space regression model describes a fixed dimensional
  // vector Y[t] as it moves throughout time.  The model is a state space model
  // of the form
  //
  //        Y[t] = Z[t] * alpha[t] + epsilon[t]
  //  alpha[t+1] = T[t] * alpha[t] + R[t] * eta[t].
  //
  // There is structure to alpha[t] that can be used for more efficient
  // learning.  The state consists of two types of state components: shared and
  // series-specific.  A shared state component is a regular state component
  // from a dynamic factor model, with a matrix Z[t] mapping state to outcomes.
  // A series specific model maintains a separate element of state for each
  // dimension of Y[t].
  //
  // The learning algorithm can cycle back and forth between (draw shared state
  // given data and series-specific state), (draw series-specific state), and
  // (draw parameters given complete data).
  //
  // The model assumes that errors from each state component are independent of
  // other state components (given model parameters), and that the observation
  // errors epsilon[t] are conditionally independent of everything else given
  // state and model parameters.  Both eta[t] and epsilon[t] are Gaussian.  This
  // model makes the further simplifying assumption that Var(epsilon[t]) is
  // diagonal, so that any cross sectional correlations between elements of Y[t]
  // are captured by shared state.
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

    MultivariateStateSpaceRegressionModel(
        const MultivariateStateSpaceRegressionModel &rhs) = delete;
    MultivariateStateSpaceRegressionModel &operator=(
        const MultivariateStateSpaceRegressionModel &rhs) = delete;
    MultivariateStateSpaceRegressionModel(
        MultivariateStateSpaceRegressionModel &&rhs) = delete;
    MultivariateStateSpaceRegressionModel &operator=(
        MultivariateStateSpaceRegressionModel &&rhs) = delete;
    
    MultivariateStateSpaceRegressionModel *clone() const override {
      report_error("Model cannot be copied.");
      return nullptr;
    }

    //-----------------------------------------------------------------
    // Dimension of shared state.
    int state_dimension() const override {
      return shared_state_models_.state_dimension();
    }
    
    int number_of_state_models() const override {
      return shared_state_models_.size();
    }

    SharedStateModel *state_model(int s) override {
      return shared_state_models_[s].get();
    }

    const SharedStateModel *state_model(int s) const override {
      return shared_state_models_[s].get();
    }
    
    //-----------------------------------------------------------------
    // Data policy overrides and augmentations.
    
    // The number of time points that have been observed.
    int time_dimension() const override {return time_dimension_;}

    // Adding data to this model also adjusts time_dimension_.
    void add_data(const Ptr<Data> &dp) override;
    void add_data(const Ptr<TimeSeriesRegressionData> &dp) override;
    void add_data(TimeSeriesRegressionData *dp) override;
    void clear_data() override;

    // Scalar data access.
    double response_matrix(int series, int time) const {
      return response_matrix_(series, time);
    }

    // Vector data access.
    ConstVectorView observation(int t) const override {
      return response_matrix_.col(t);
    }

    const Selector &observed_status(int t) const override {
      return observed_.col(t);
    }
    
    // To be called after add_data has been called for the last time.
    void finalize_data();

    
    // The dimension of the series-specific state associated with a particular
    // time series.
    int series_state_dimension(int which_series) const {
      if (proxy_models_.empty()) {
        return 0;
      } else {
        return proxy_models_[which_series]->state_dimension();
      }
    }
    
    // The number of time series being modeled.  This is an override of the
    // nseries() method in the data policy.
    int nseries() const {return nseries_;}

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
      has_series_specific_state_ = true;
    }

    // Indicates whether any of the proxy models have had state assigned.
    bool has_series_specific_state() const {
      return has_series_specific_state_;
    }
    
    // Returns the observed data point for the given series at the given time
    // point.  If that data point is missing, negative_infinity is returned.
    double observed_data(int series, int time) const;

    double adjusted_observation(int series, int time) const {
      return adjusted_data_workspace_(series, time);
    }

    ConstVectorView adjusted_observation(int time) const override {
      return adjusted_data_workspace_.col(time);
    }

    const SparseKalmanMatrix *observation_coefficients(
        int t, const Selector &observed) const override;
    
    DiagonalMatrix observation_variance(int t) const override;
    
    double single_observation_variance(int t, int dim) const override {
      return observation_model_->model(dim)->sigsq();
    }
    
    // Returns a flag indicating whether the requested series was observed at
    // the requested time.
    bool is_observed(int series, int time) const {
      return observed_(series, time);
    }
    
    void impute_state(RNG &rng) override;

    Ptr<ProxyScalarStateSpaceModel> series_specific_model(int index) {
      return proxy_models_[index];
    }

    IndependentRegressionModels *observation_model() override {
      return observation_model_.get();
    }

    const IndependentRegressionModels *observation_model() const override {
      return observation_model_.get();
    }

    //    void kalman_filter() override;

    Matrix state_contributions(int which_state_model) const override;
      
   private:
    // Populate the vector of proxy models with 'nseries_' empty models.
    void initialize_proxy_models();

    // Set observers on the variance parameters of the regression models, so
    // that the diagonal variance matrix can be updated when it gets out of
    // sync.
    void set_observation_variance_observers();
    
    // If the observation variance is out of step with the observation_variance_
    // data member, update the data member.  This function is logically const.
    void update_observation_variance() const;
    
    void observe_state(int t) override;
    void observe_initial_state();
    void observe_data_given_state(int t) override;
    
    using ConditionallyIndependentMultivariateStateSpaceModelBase::get_filter;

    StateModelVectorBase &state_model_vector() override {
      return shared_state_models_;
    }
    
    const StateModelVectorBase &state_model_vector() const override {
      return shared_state_models_;
    }
    
    // Returns a view into series_specific_state_component for a particular
    // state model, series, and point in time.
    ConstVectorView series_specific_state_component(
        int series, int time, int state_model_index) const;
    
    void impute_shared_state_given_series_state(RNG &rng);
    void impute_series_state_given_shared_state(RNG &rng);

    // Sets adjusted_data_workspace_ to observed_data minus contributions from
    // series specific state.
    void subtract_series_specific_state();

    // Sets adjusted_data_workspace_ to observed_data minus contributions from
    // shared state.
    void subtract_shared_state();

    //--------------------------------------------------------------------------
    // Data section.
    
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
    
    mutable Ptr<StackedMatrixBlock> observation_coefficients_;
    
    // Initially set to false.  Flips to true if any state is assigned to proxy
    // models.
    bool has_series_specific_state_;
    
    // The response matrix organizes all the scalar responses from each data
    // point.  Time flows horizontally, so each column is a single time point.
    mutable Matrix response_matrix_;
    mutable SelectorMatrix observed_;
    
    mutable bool data_is_finalized_;

    // Shared state components.  The columns of shared_state_ represent time
    // points.  The rows represent state variables.
    Matrix shared_state_;

    // A workspace where observed data can be modified by subtracting off
    // components on which we wish to condition.
    Matrix adjusted_data_workspace_;

    // A workspace to copy the residual variances stored in observation_model_
    // in the data structure expected by the model.
    mutable DiagonalMatrix observation_variance_;

    // A flag to keep track of whether the observation variance is current.
    mutable bool observation_variance_current_;
  };

}  // namespace BOOM

#endif  // BOOM_MULTIVARIATE_STATE_SPACE_REGRESSION_HPP_

