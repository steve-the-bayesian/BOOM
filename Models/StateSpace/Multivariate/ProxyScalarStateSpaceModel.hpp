#ifndef BOOM_STATE_SPACE_SCALAR_PROXY_MODEL_HPP_
#define BOOM_STATE_SPACE_SCALAR_PROXY_MODEL_HPP_
/*
  Copyright (C) 2005-2022 Steven L. Scott

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

#include "cpputil/Ptr.hpp"

namespace BOOM {
  //===========================================================================
  // A multivariate state space regression model maintains a set of
  // ScalarKalmanFilter objects to handle simulating from series-specific state.
  // Each of these series-specific filters needs a "state space model" to supply
  // the kalman matrices and data.  This class is a proxy state space model
  // filling that role.  The proxy model keeps a pointer to the host model from
  // which it draws data and parameters.  The proxy also assumes ownership of
  // any series-specific state.
  template <class HOST_TYPE>
  class ProxyScalarStateSpaceModel : public StateSpaceModel {
   public:
    // Args:
    //   model:  The host model.
    //   which_series: The index of the time series that this object describes.
    ProxyScalarStateSpaceModel(HOST_TYPE *host_model, int which_series)
        : host_(host_model),
          which_series_(which_series)
    {}

    // The number of distinct time points in the host model.
    int time_dimension() const override {return host_->time_dimension();}

    // The value of the time series specific to this proxy.  The host should
    // have subtracted any regression effects or shared state before this
    // function is called.
    double adjusted_observation(int t) const override {
      return host_->adjusted_observation(which_series_, t);
    }

    bool is_missing_observation(int t) const override {
      return !host_->is_observed(which_series_, t);
    }

    ZeroMeanGaussianModel *observation_model() override { return nullptr; }
    const ZeroMeanGaussianModel *observation_model() const override {
      return nullptr;
    }

    double observation_variance(int t) const override {
      return host_->single_observation_variance(t, which_series_);
    }

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
    //
    // Note, this is StateSpaceModel::simulate_forecast, but with a zero
    // residual variance.
    Vector simulate_state_contribution_forecast(
        RNG &rng, int horizon, const Vector &final_state) {
      Vector ans(horizon, 0.0);
      if (state_dimension() > 0) {
        Vector state = final_state;
        int t0 = time_dimension();
        for (int t = 0; t < horizon; ++t) {
          state = simulate_next_state(rng, state, t + t0);
          ans[t] = observation_matrix(t + t0).dot(state);
        }
      }
      return ans;
    }

    // Ensure all state models are capable of handling times up to t-1.
    void observe_time_dimension(int t) {
      for (int s = 0; s < number_of_state_models(); ++s) {
        state_model(s)->observe_time_dimension(t);
      }
    }

   private:
    // The add_data method is disabled.
    void add_data(const Ptr<StateSpace::MultiplexedDoubleData>
                  &data_point) override {}
    void add_data(const Ptr<Data> &data_point) override {}

    HOST_TYPE *host_;
    int which_series_;
  };



}  // namespace BOOM


#endif  // BOOM_STATE_SPACE_SCALAR_PROXY_MODEL_HPP_
