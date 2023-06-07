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

#include "Models/StateSpace/Multivariate/StudentMvssRegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/TDataImputer.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    using StudentData = StudentMultivariateTimeSeriesRegressionData;
  }

  StudentData::StudentMultivariateTimeSeriesRegressionData(
      double y, const Vector &x, int series, int timestamp)
      : MultivariateTimeSeriesRegressionData(y, x, series, timestamp),
        weight_(1.0)
  {}

  StudentData::StudentMultivariateTimeSeriesRegressionData(
      const Ptr<DoubleData> & y,
      const Ptr<VectorData> &x,
      int series,
      int timestamp)
      : MultivariateTimeSeriesRegressionData(y, x, series, timestamp),
        weight_(1.0)
  {}

  StudentMvssRegressionModel::StudentMvssRegressionModel(int xdim, int nseries)
      : data_policy_(nseries),
        observation_model_(new ObservationModel(xdim, nseries)),
        observation_variance_(nseries),
        observation_variance_current_(false),
        dummy_selector_(nseries, true)
  {
    state_manager_.initialize_proxy_models(this);
    set_observation_variance_observers();
    set_workspace_observers();
    set_parameter_observers(observation_model_.get());
  }

  StudentMvssRegressionModel * StudentMvssRegressionModel::clone() const {
    report_error("Model is not clonable");
    return const_cast<StudentMvssRegressionModel *>(this);
  }

  StudentMvssRegressionModel * StudentMvssRegressionModel::deepclone() const {
    report_error("Model is not deepclonable");
    return const_cast<StudentMvssRegressionModel *>(this);
  }

  Matrix StudentMvssRegressionModel::simulate_forecast(
      RNG &rng,
      const Matrix &forecast_predictors,
      const Vector &final_shared_state,
      const std::vector<Vector> &series_specific_final_state) {
    int horizon = forecast_predictors.nrow() / nseries();
    if (horizon * nseries() != forecast_predictors.nrow()) {
      report_error("The number of rows in forecast_predictors must be an integer "
                   "multiple of the number of series.");
    }

    Matrix forecast(nseries(), horizon, 0.0);
    if (has_series_specific_state()) {
      forecast += state_manager_.series_specific_forecast(
          rng, horizon, series_specific_final_state);
    }

    // Add shared state component.
    int time = 0;
    Vector state = final_shared_state;
    Selector fully_observed(nseries(), true);
    int t0 = time_dimension();
    for (int t = 0; t < horizon; ++t) {
      advance_to_timestamp(rng, time, state, t, t);
      forecast.col(t) += *observation_coefficients(time + t0, fully_observed) * state;
    }

    // Add regression component and residual error.
    int index = 0;
    for (int t = 0; t < horizon; ++t) {
      for (int series = 0; series < nseries(); ++series) {
        const TRegressionModel *obs_model = observation_model()->model(series);
        forecast(series, t) += obs_model->predict(forecast_predictors.row(index))
            + rt_mt(rng, obs_model->nu()) * obs_model->sigma();
      }
    }
    return forecast;

  }

  //---------------------------------------------------------------------------
  DiagonalMatrix StudentMvssRegressionModel::observation_variance(int t) const {
    return observation_variance(t, dummy_selector_);
  }

  //---------------------------------------------------------------------------
  DiagonalMatrix StudentMvssRegressionModel::observation_variance(
      int t, const Selector &observed) const {
    Vector diagonal_elements(observed.nvars());
    for (int i = 0; i < observed.nvars(); ++i) {
      int I = observed.expanded_index(i);
      diagonal_elements[i] = observation_model()->model(I)->sigsq()
          / data_policy_.data_point(I, t)->weight();
    }
    return DiagonalMatrix(diagonal_elements);
  }

  Vector
  StudentMvssRegressionModel::observation_variance_parameter_values() const {
    Vector ans(nseries());
    for (int i = 0; i < nseries(); ++i) {
      ans[i] = observation_model()->model(i)->sigsq();
    }
    return ans;
  }

  //---------------------------------------------------------------------------
  double StudentMvssRegressionModel::single_observation_variance(
      int time, int series) const {
    const Selector &observed(observed_status(time));
    int I = observed.expanded_index(series);
    double ans;
    ///// should time be > 0 or >= 0????
    if (time >= 0 && time < time_dimension()) {
      ans = observation_model()->model(I)->sigsq() /
          data_policy_.data_point(I, time)->weight();
    } else {
      ans = observation_model()->model(I)->residual_variance();
    }
    return ans;
  }

  void StudentMvssRegressionModel::impute_student_weights(RNG &rng) {
    TDataImputer imputer;
    for (size_t time = 0; time < time_dimension(); ++time) {
      const Selector &observed(observed_status(time));

      // state_contribution contains the contribution from the shared state to
      // each observed data value.  Its index runs from 0 to the number of
      // observed time series (i.e. observed.nvars()).
      Vector shared_state_contribution =
          *observation_coefficients(time, observed) * shared_state(time);

      for (size_t s = 0; s < observed.nvars(); ++s) {
        int series = observed.sparse_index(s);
        StudentData *data_point = data_policy_.data_point(series, time).get();
        double time_series_residual =
            data_point->y() - shared_state_contribution[s];
        if (has_series_specific_state()) {
          time_series_residual -= series_specific_state_contribution(series, time);
        }
        const TRegressionModel *obs_model = observation_model()->model(series);

        double residual = time_series_residual
            - obs_model->predict(data_point->x());
        double weight = imputer.impute(
            rng, residual, obs_model->sigma(), obs_model->nu());
        data_point->set_weight(weight);
      }
    }
  }

  ConstVectorView StudentMvssRegressionModel::adjusted_observation(
      int time) const {
    const Selector &observed(observed_status(time));
    Ptr<SparseKalmanMatrix> coefficients(
        observation_coefficients(time, observed));
    return data_policy_.adjusted_observation(
        time, state_manager_, observation_model_.get(), *coefficients,
        shared_state());
  }


  void StudentMvssRegressionModel::observe_state(int t) {
    if (t == 0) {
      observe_initial_state();
    } else {
      const ConstVectorView now(shared_state(t));
      const ConstVectorView then(shared_state(t - 1));
      for (int s = 0; s < number_of_state_models(); ++s) {
        state_model(s)->observe_state(state_component(then, s),
                                      state_component(now, s), t);
      }
    }
  }

  void StudentMvssRegressionModel::observe_initial_state() {
    for (int s = 0; s < number_of_state_models(); ++s) {
      ConstVectorView state(state_component(shared_state(0), s));
      state_model(s)->observe_initial_state(state);
    }
  }

  void StudentMvssRegressionModel::set_observation_variance_observers() {
    for (int i = 0; i < observation_model_->ydim(); ++i) {
      auto observer = [this]() {this->observation_variance_current_ = false;};
      observation_model_->model(i)->Sigsq_prm()->add_observer(this, observer);
      observation_model_->model(i)->Nu_prm()->add_observer(this, observer);
    }
  }

  void StudentMvssRegressionModel::set_workspace_observers() {
    std::vector<Ptr<Params>> params = parameter_vector();
    data_policy_.set_observers(params);
  }

  void StudentMvssRegressionModel::set_parameter_observers(Model *model) {
    std::vector<Ptr<Params>> parameters = model->parameter_vector();
    for (auto &el : parameters) {
      el->add_observer(
          el.get(),
          [this](void) {
            this->get_filter().set_status(
                KalmanFilterBase::KalmanFilterStatus::NOT_CURRENT);
          });
    }
  }

  // If an observation is observed, subtract off the time series contribution
  // and add the residual bit to the regression model.
  void StudentMvssRegressionModel::observe_data_given_state(int time) {
    for (int series = 0; series < nseries(); ++series) {
      Vector shared_state_contribution =
          *observation_coefficients(time, dummy_selector_) * shared_state(time);
      if (is_observed(series, time)) {
        const Ptr<StudentMultivariateTimeSeriesRegressionData> &data_point(
            data_policy_.data_point(series, time));
        double regression_contribution = observed_data(series, time)
            - shared_state_contribution[series]
            - state_manager_.series_specific_state_contribution(series, time);
        CompleteDataStudentRegressionModel *obs_model =
            observation_model_->model(series);

        // There is an optimization opportunity here, because the regression
        // data point is reallocated every time.
        obs_model->add_data(new RegressionData(
                              new UnivData(regression_contribution),
                              data_point->Xptr()),
                            data_point->weight());
      }
    }
  }

  void StudentMvssRegressionModel::impute_shared_state_given_series_state(
      RNG &rng) {
    resize_subordinate_state();
    data_policy_.isolate_shared_state();
    MultivariateStateSpaceModelBase::impute_state(rng);
    data_policy_.unset_workspace();
  }

  void StudentMvssRegressionModel::impute_series_state_given_shared_state(
      RNG &rng) {
    if (has_series_specific_state()) {
      data_policy_.isolate_series_specific_state();
      for (int s = 0; s < nseries(); ++s) {
        if (state_manager_.series_specific_model(s)->state_dimension() > 0) {
          state_manager_.series_specific_model(s)->impute_state(rng);
        }
      }
      data_policy_.unset_workspace();
    }
  }

  void StudentMvssRegressionModel::resize_subordinate_state() {
    for (int series = 0; series < nseries(); ++series) {
      state_manager_.series_specific_model(series)->resize_state();
    }
  }


}  // namespace BOOM
