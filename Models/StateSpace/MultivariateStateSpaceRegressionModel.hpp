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
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateModelVector.hpp"
#include "Models/StateSpace/MultivariateStateSpaceModelBase.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"

namespace BOOM {

  // Store y, x_series, x_shared, series_id, timestamp?
  class TimeSeriesRegressionData : public Data {
   public:
   private:
    Ptr<VectorData> shared_predictors_;
    Ptr<VectorData> series_specific_predictors_;
  };


  // The MultivariateStateSpaceRegressionModel maintains a set of
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
  
  class TimeSeriesRegressionDataPolicy
      : public IID_DataPolicy<TimeSeriesRegressionData> {
   public:
   private:
  };

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
  // diagonal, so that any cross seciontional correlations between elements of
  // Y[t] are captured by shared state.

  // DEVELOPMENT NOTES: This class conceptually shares much with
  // StateSpaceModelBase, but there will be important differences.  We will
  // start by making it completely independent of StateSpaceModelBase, with the
  // aim of sharing code once the class is functional.
  class MultivariateStateSpaceRegressionModel
      : public CompositeParamPolicy,
        public IID_DataPolicy<TimeSeriesRegressionData>,
        public PriorPolicy
  {
   public:
    // Args:
    //   nseries:  The number of time series being modeled.
    explicit MultivariateStateSpaceRegressionModel(int nseries);
    
    int time_dimension() const;

    // Dimension of shared state.
    int state_dimension() const;

    // The dimension of the series-specific state associated with a particular
    // time series.
    int series_state_dimension(int which_series) const;

    // The number of time series being modeled.
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
      total_state_dimension_ += state_model->state_dimension();
      total_state_error_dimension_ += state_model->state_error_dimension();
      proxy_models_[series]->add_state(state_model);
    }

    // Returns the observed data point for the given series at the given time
    // point.  If that data point is missing, negative_infinity is returned.
    double observed_data(int series, int time) const;

    double adjusted_observation(int series, int time) const {
      return adjusted_data_workspace_(series, time);
    }
    
    // Returns a flag indicating whether the requested series was observed at
    // the requested time.
    bool is_observed(int series, int time) const;
    
    void impute_state(RNG &rng);
    
   private:
    
    // Returns a view into series_specific_state_component for a particular
    // state model, series, and point in time.
    ConstVectorView series_specific_state_component(
        int series, int time, int state_model_index) const;
    
    void impute_shared_state_given_series_state(RNG &rng);
    void impute_series_state_given_shared_state(RNG &rng);

    // Compute a matrix of data formed by subtracting the series specific state
    // contributions from the observed data.
    void subtract_series_specific_state();

    StateSpaceUtils::StateModelVector<SharedStateModel> shared_state_models_;

    // The proxy models hold components of state that are specific to individual
    // data series.
    std::vector<Ptr<ProxyScalarStateSpaceModel>> proxy_models_;

    int nseries_;         // set in constructor.
    int time_dimension_;  // set in add_data.
    
    // Shared state components.  The columns of shared_state_ represent time
    // points.  The rows represent state variables.
    Matrix shared_state_;

    // shared_state_positions_[s] is the index in the state vector where the
    // state for shared_state_models_[s] begins. There will be one more entry in
    // this vector than the number of state models.  The last entry can be
    // ignored.
    std::vector<int> shared_state_positions_;
    std::vector<int> shared_state_error_positions_;
    int state_dimension_;
    
    int total_state_dimension_;
    int total_state_error_dimension_;

    bool state_resize_needed_;

    // A workspace where observed data can be modified by subtracting off
    // components on which we wish to condition.
    Matrix adjusted_data_workspace_;
  };

}  // namespace BOOM

#endif  // BOOM_MULTIVARIATE_STATE_SPACE_REGRESSION_HPP_

