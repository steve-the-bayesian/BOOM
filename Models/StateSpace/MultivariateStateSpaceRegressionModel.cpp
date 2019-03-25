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

#include "Models/StateSpace/MultivariateStateSpaceRegressionModel.hpp"

namespace BOOM {

  namespace {
    using MSSRM = MultivariateStateSpaceRegressionModel;
    using PSSSM = ProxyScalarStateSpaceModel;
  }  // namespace

  PSSSM::ProxyScalarStateSpaceModel(
      MultivariateStateSpaceRegressionModel *model,
      int which_series)
      : model_(model),
        which_series_(which_series)
  {}

  int PSSSM::time_dimension() const {return model_->time_dimension();}
  double PSSSM::adjusted_observation(int t) const {
    return model_->adjusted_observation(which_series_, t);
  }
  bool PSSSM::is_missing_observation(int t) const {
    return model_->is_observed(which_series_, t);
  }

  void PSSSM::add_data(
      const Ptr<StateSpace::MultiplexedDoubleData> &data_point) {
    report_error("add_data is disabled.");
  }
  
  void PSSSM::add_data(const Ptr<Data> &data_point) {
    report_error("add_data is disabled.");
  }

  //===========================================================================
  MSSRM::MultivariateStateSpaceRegressionModel(int nseries)
      : nseries_(nseries),
        time_dimension_(0),
        shared_state_positions_(1, 0),
        shared_state_error_positions_(1, 0),
        state_dimension_(0),
        total_state_dimension_(0),
        total_state_error_dimension_(0),
        state_resize_needed_(true)
  {
    proxy_models_.reserve(nseries_);
    for (int i = 0; i < nseries_; ++i) {
      proxy_models_.push_back(new ProxyScalarStateSpaceModel(this, i));
    }
  }
  
  void MSSRM::add_state(const Ptr<SharedStateModel> &state_model) {
    shared_state_models_.add_state(state_model);
    total_state_dimension_ += state_model->state_dimension();
    total_state_error_dimension_ += state_model->state_error_dimension();
  }
  
  void MSSRM::impute_state(RNG &rng) {
    impute_shared_state_given_series_state(rng);
    impute_series_state_given_shared_state(rng);
  }

  void MSSRM::impute_shared_state_given_series_state(RNG &rng) {
    subtract_series_specific_state();

    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////

  }

  void MSSRM::subtract_series_specific_state() {
    adjusted_data_workspace_.resize(nseries(), time_dimension());
    for (int time = 0; time < time_dimension(); ++time) {
      for (int series = 0; series < nseries(); ++series) {
        adjusted_data_workspace_(series, time) = observed_data(series, time);
        if (is_observed(series, time)) {
          auto proxy_model = proxy_models_[series];
          int nstate = proxy_model->number_of_state_models();
          for (int s = 0; s < nstate; ++s) {
            adjusted_data_workspace_(series, time) -=
                proxy_model->observation_matrix(time).dot(
                    series_specific_state_component(series, time, s));
          }
        }
      }
    }
  }
  
  
  
}  // namespace BOOM
