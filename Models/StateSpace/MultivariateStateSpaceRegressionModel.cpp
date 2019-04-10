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
    using TSRD = TimeSeriesRegressionData;
  }  // namespace

  TSRD::TimeSeriesRegressionData(double y, const Vector &x, int series, int timestamp)
      : RegressionData(y, x),
        which_series_(series),
        timestamp_index_(timestamp)
  {}

  TSRD::TimeSeriesRegressionData(const Ptr<DoubleData> &y,
                                 const Ptr<VectorData> &x,
                                 int series,
                                 int timestamp)
      : RegressionData(y, x),
        which_series_(series),
        timestamp_index_(timestamp)
  {}
  
  //===========================================================================
  void MSSRM::add_data(const Ptr<TimeSeriesRegressionData> &dp) {
    if (dp->series() >= nseries()) {
      report_error("Series ID too large.");
    }
    data_is_finalized_ = false;
    time_dimension_ = std::max<int>(time_dimension_, 1 + dp->timestamp());
    data_indices_[dp->series()][dp->timestamp()] = dat().size();
    IID_DataPolicy<TimeSeriesRegressionData>::add_data(dp);
  }

  void MSSRM::add_data(const Ptr<Data> &dp) {
    this->add_data(dp.dcast<TimeSeriesRegressionData>());
  }

  void MSSRM::add_data(TimeSeriesRegressionData *dp) {
    this->add_data(Ptr<TimeSeriesRegressionData>(dp));
  }

  void MSSRM::clear_data() {
    time_dimension_ = 0;
    response_matrix_.resize(0, 0);
    observed_ = SelectorMatrix(0, 0);
    data_is_finalized_ = false;
    data_indices_.clear();
    IID_DataPolicy<TimeSeriesRegressionData>::clear_data();
  }
  
  void MSSRM::finalize_data() const {
    if (!data_is_finalized_) {
      if (time_dimension_ <= 0) {
        return;
      }
      response_matrix_.resize(nseries(), time_dimension_);
      response_matrix_ = negative_infinity();
      observed_ = SelectorMatrix(nseries(), time_dimension_, false);
      for (int i = 0; i < dat().size(); ++i) {
        const Ptr<TimeSeriesRegressionData> &dp(dat()[i]);
        int time = dp->timestamp();
        int series = dp->series();
        if (dp->missing() == Data::observed) {
          response_matrix_(series, time) = dp->y();
          observed_.add(series, time);
        }
      }
      data_is_finalized_ = true;
    }
  }
  
  //===========================================================================

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
  MSSRM::MultivariateStateSpaceRegressionModel(int xdim, int nseries)
      : nseries_(nseries),
        time_dimension_(0),
        observation_model_(new IndependentRegressionModels(xdim, nseries)),
        has_series_specific_state_(false),
        response_matrix_(0, 0),
        observed_(0, 0, false),
        data_is_finalized_(false),
        observation_variance_(nseries),
        observation_variance_current_(false)
  {
    initialize_proxy_models();
    set_observation_variance_observers();
  }

  void MSSRM::add_state(const Ptr<SharedStateModel> &state_model) {
    shared_state_models_.add_state(state_model);
  }

  double MSSRM::observed_data(int series, int time) const {
    finalize_data();
    return response_matrix(series, time);
  }
  
  const SparseKalmanMatrix *MSSRM::observation_coefficients(
      int t, const Selector &observed) const {
    observation_coefficients_->clear();
    for (int s = 0; s < number_of_state_models(); ++s) {
      observation_coefficients_->add_block(
          shared_state_models_[s]->observation_coefficients(t, observed));
    }
    return observation_coefficients_.get();
  }
  
  DiagonalMatrix MSSRM::observation_variance(int t) const {
    update_observation_variance();
    return observation_variance_;
  }
  
  void MSSRM::impute_state(RNG &rng) {
    impute_shared_state_given_series_state(rng);
    impute_series_state_given_shared_state(rng);
  }

  void MSSRM::impute_shared_state_given_series_state(RNG &rng) {
    isolate_shared_state();
    MultivariateStateSpaceModelBase::impute_state(rng);
  }

  void MSSRM::impute_series_state_given_shared_state(RNG &rng) {
    if (has_series_specific_state()) {
      isolate_series_specific_state();
      for (int s = 0; s < nseries(); ++s) {
        proxy_models_[s]->impute_state(rng);
      }
    }
  }

  void MSSRM::observe_state(int t) {
    if (t == 0) {
      observe_initial_state();
      return;
    }
    const ConstVectorView now(shared_state_.col(t));
    const ConstVectorView then(shared_state_.col(t - 1));
    for (int s = 0; s < number_of_state_models(); ++s) {
      state_model(s)->observe_state(state_component(then, s),
                                    state_component(now, s), t);
    }
  }

  // If an observation is observed, subtract off the time series contribution
  // and add the residual bit to the regression model.
  void MSSRM::observe_data_given_state(int time) {
    for (int series = 0; series < nseries(); ++series) {
      Vector time_series_contribution(nseries(), 0.0);
      // TODO(steve):  Fill this.
      
      if (is_observed(series, time)) {
        int index = data_indices_[series][time];
        Ptr<TimeSeriesRegressionData> dp = dat()[index];
        double regression_contribution =
            dp->y() - time_series_contribution[series];
        observation_model_->model(series)->suf()->add_mixture_data(
            regression_contribution, dp->x(), 1.0);
      }
    }
  }
  
  //----------------------------------------------------------------------
  void MSSRM::observe_initial_state() {
    for (int s = 0; s < number_of_state_models(); ++s) {
      ConstVectorView state(state_component(shared_state_.col(0), s));
      state_model(s)->observe_initial_state(state);
    }
  }

  //----------------------------------------------------------------------
  Matrix MSSRM::state_contributions(int which_state_model) const {
    Selector observed(nseries(), true);
    const SharedStateModel* model = state_model(which_state_model);
    Matrix ans(model->state_dimension(), time_dimension());
    for (int t = 0; t < time_dimension(); ++t) {
      ConstVectorView state(state_component(
          shared_state_.col(t), which_state_model));
      ans.col(t) = *model->observation_coefficients(t, observed) * state;
    }
    return ans;
  }

  //---------------------------------------------------------------------------
  // Implementation of private member functions.
  //---------------------------------------------------------------------------
  void MSSRM::update_observation_variance() const {
    if (observation_variance_current_) return;
    VectorView elements(observation_variance_.diag());
    for (int i = 0; i < nseries(); ++i) {
      elements[i] = observation_model_->model(i)->sigsq();
    }
    observation_variance_current_ = true;
  }

  void MSSRM::initialize_proxy_models() {
    proxy_models_.clear();
    has_series_specific_state_ = false;
    proxy_models_.reserve(nseries_);
    for (int i = 0; i < nseries_; ++i) {
      proxy_models_.push_back(new ProxyScalarStateSpaceModel(this, i));
    }
  }

  void MSSRM::set_observation_variance_observers() {
    for (int i = 0; i < observation_model_->ydim(); ++i) {
      observation_model_->model(i)->Sigsq_prm()->add_observer(
          [this]() {this->observation_variance_current_ = false;});
    }
  }

  // TODO(steve):  Subtract regression component as well.
  void MSSRM::isolate_shared_state() {
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

  // TODO(steve):  Subtract regression component as well.
  void MSSRM::isolate_series_specific_state() {
    adjusted_data_workspace_.resize(nseries(), time_dimension());
    Selector observed(nseries(), true);
    for (int time = 0; time < time_dimension(); ++time) {
      ConstVectorView state(shared_state_.col(time));
      Vector shared_state_contribution =
          *observation_coefficients(time, observed) * state;
      for (int series = 0; series < nseries(); ++series) {
        adjusted_data_workspace_(series, time)
            = observed_data(series, time) - shared_state_contribution[series];
      }
    }
  }
  
  
}  // namespace BOOM
