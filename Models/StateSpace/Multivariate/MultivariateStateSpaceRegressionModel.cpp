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

#include "Models/StateSpace/Multivariate/MultivariateStateSpaceRegressionModel.hpp"
#include "Models/StateSpace/Filters/KalmanFilterBase.hpp"
#include "distributions.hpp"
#include "numopt.hpp"
#include "numopt/Powell.hpp"

namespace BOOM {

  namespace {
    using MSSRM = MultivariateStateSpaceRegressionModel;
    using PSSSM = ProxyScalarStateSpaceModel;
    using TSRD = MultivariateTimeSeriesRegressionData;
  }  // namespace

  TSRD::MultivariateTimeSeriesRegressionData(
      double y, const Vector &x, int series, int timestamp)
      : RegressionData(y, x),
        which_series_(series),
        timestamp_index_(timestamp)
  {}

  TSRD::MultivariateTimeSeriesRegressionData(
      const Ptr<DoubleData> &y,
      const Ptr<VectorData> &x,
      int series,
      int timestamp)
      : RegressionData(y, x),
        which_series_(series),
        timestamp_index_(timestamp)
  {}

  //===========================================================================
  PSSSM::ProxyScalarStateSpaceModel(
      MultivariateStateSpaceRegressionModel *model,
      int which_series)
      : host_(model),
        which_series_(which_series)
  {}

  int PSSSM::time_dimension() const { return host_->time_dimension();}

  double PSSSM::adjusted_observation(int t) const {
    return host_->adjusted_observation(which_series_, t);
  }

  bool PSSSM::is_missing_observation(int t) const {
    return !host_->is_observed(which_series_, t);
  }

  double PSSSM::observation_variance(int t) const {
    return host_->single_observation_variance(t, which_series_);
  }

  void PSSSM::add_data(
      const Ptr<StateSpace::MultiplexedDoubleData> &data_point) {
    report_error("add_data is disabled.");
  }

  void PSSSM::add_data(const Ptr<Data> &data_point) {
    report_error("add_data is disabled.");
  }

  // This is StateSpaceModel::simulate_forecast, but with a zero residual
  // variance.
  Vector PSSSM::simulate_state_contribution_forecast(
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

  //===========================================================================
  MSSRM::MultivariateStateSpaceRegressionModel(int xdim, int nseries)
      : nseries_(nseries),
        time_dimension_(0),
        observation_model_(new IndependentRegressionModels(xdim, nseries)),
        adjusted_data_workspace_(nseries),
        workspace_time_index_(-1),
        workspace_status_(UNSET),
        observation_variance_(nseries),
        observation_variance_current_(false),
        dummy_selector_(nseries, true)
  {
    initialize_proxy_models();
    set_observation_variance_observers();
    set_workspace_observers();
    set_parameter_observers(observation_model_.get());
  }

  MSSRM * MSSRM::clone() const {
    report_error("Model cannot be copied");
    return nullptr;
  }

  MSSRM * MSSRM::deepclone() const {
    MSSRM *ans = clone();
    // Clone will throw an exception, so the code below will never be exectued.
    // But if we ever get around to allowing copying then the rest of this
    // function will work.
    if (ans) {
      ans->copy_samplers(*this);
    }
    return ans;
  }

  Matrix MSSRM::simulate_forecast(
      RNG &rng,
      const Matrix &forecast_data,
      const Vector &final_shared_state,
      const std::vector<Vector> &series_specific_final_state) {
    int horizon = forecast_data.nrow() / nseries();
    if (horizon * nseries() != forecast_data.nrow()) {
      report_error("The number of rows in forecast_data must be an "
                   "integer multiple of the number of series.");
    }
    Matrix forecast(nseries(), horizon, 0.0);
    // Add series specific component.
    if (has_series_specific_state()) {
      for (int j = 0; j < nseries(); ++j) {
        forecast.row(j) +=
            proxy_models_[j]->simulate_state_contribution_forecast(
                rng, horizon, series_specific_final_state[j]);
      }
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
        forecast(series, t) += observation_model()->model(series)->predict(
            forecast_data.row(index))
            + rnorm_mt(rng, 0, observation_model()->model(series)->sigma());
      }
    }
    return forecast;
  }

  void MSSRM::add_state(const Ptr<SharedStateModel> &state_model) {
    shared_state_models_.add_state(state_model);
    set_parameter_observers(state_model.get());
  }

  bool MSSRM::has_series_specific_state() const {
    for (int i = 0; i < proxy_models_.size(); ++i) {
      if (proxy_models_[i]->state_dimension() > 0) {
        return true;
      }
    }
    return false;
  }

  void MSSRM::impute_state(RNG &rng) {
    workspace_current_ = false;
    impute_shared_state_given_series_state(rng);
    impute_series_state_given_shared_state(rng);
    workspace_current_ = false;
  }

  void MSSRM::add_data(const Ptr<MultivariateTimeSeriesRegressionData> &dp) {
    if (dp->series() >= nseries()) {
      report_error("Series ID too large.");
    }

    time_dimension_ = std::max<int>(time_dimension_, 1 + dp->timestamp());
    data_indices_[dp->series()][dp->timestamp()] = dat().size();
    IID_DataPolicy<MultivariateTimeSeriesRegressionData>::add_data(dp);
    while (observed_.size() <= dp->timestamp()) {
      Selector all_missing(nseries(), false);
      observed_.push_back(all_missing);
    }
    observed_[dp->timestamp()].add(dp->series());
  }

  void MSSRM::add_data(const Ptr<Data> &dp) {
    this->add_data(dp.dcast<MultivariateTimeSeriesRegressionData>());
  }

  void MSSRM::add_data(MultivariateTimeSeriesRegressionData *dp) {
    this->add_data(Ptr<MultivariateTimeSeriesRegressionData>(dp));
  }

  int MSSRM::data_index(int series, int time) const {
    const auto &series_it = data_indices_.find(series);
    if (series_it == data_indices_.end()) {
      return -1;
    }
    const auto & time_it(series_it->second.find(time));
    if (time_it == series_it->second.end()) {
      return -1;
    }
    return time_it->second;
  }

  void MSSRM::clear_data() {
    time_dimension_ = 0;
    observed_.clear();
    data_indices_.clear();
    IID_DataPolicy<MultivariateTimeSeriesRegressionData>::clear_data();
  }

  void MSSRM::set_observed_status(int t, const Selector &observed) {
    if (observed.nvars_possible() != observed_[0].nvars_possible()) {
      report_error("Wrong size Selector passed to set_observed_status.");
    }
    observed_[t] = observed;
  }

  double MSSRM::adjusted_observation(int series, int time) const {
    return adjusted_observation(time)[series];
  }

  ConstVectorView MSSRM::adjusted_observation(int time) const {
    if (workspace_status_ == ISOLATE_SHARED_STATE) {
      isolate_shared_state(time);
    } else if (workspace_status_ == ISOLATE_SERIES_SPECIFIC_STATE) {
      isolate_series_specific_state(time);
    } else {
      report_error("The workspace_status_ flag must be set before calling "
                   "adjusted_observation so that the model can know which "
                   "adjustements are needed. ");
    }
    return adjusted_data_workspace_;
  }

  // The observation coefficients from the shared state portion of the model.
  // This does not include the regression coefficients from the regression
  // model, nor does it include the series-specific state.
  Ptr<SparseKalmanMatrix> MSSRM::observation_coefficients(
      int t, const Selector &observed) const {
    NEW(StackedMatrixBlock, ans)();
    for (int s = 0; s < number_of_state_models(); ++s) {
      ans->add_block(shared_state_models_[s]->observation_coefficients(
          t, observed));
    }
    return ans;
  }

  DiagonalMatrix MSSRM::observation_variance(int t) const {
    update_observation_variance();
    return observation_variance_;
  }

  DiagonalMatrix MSSRM::observation_variance(
      int t, const Selector &observed) const {
    update_observation_variance();
    if (observed.nvars() == observed.nvars_possible()) {
      return observation_variance_;
    } else {
      return DiagonalMatrix(observed.select(observation_variance_.diag()));
    }
  }

  Matrix MSSRM::state_contributions(int which_state_model) const {
    const SharedStateModel* model = state_model(which_state_model);
    Matrix ans(nseries(), time_dimension());
    for (int t = 0; t < time_dimension(); ++t) {
      ConstVectorView state(state_component(
          shared_state(t), which_state_model));
      ans.col(t) = *model->observation_coefficients(t, dummy_selector_) * state;
    }
    return ans;
  }

  //---------------------------------------------------------------------------
  // MLE and EM algorithm.
  //---------------------------------------------------------------------------

  namespace {
    class MultivariateStateSpaceTargetFun {
     public:
      explicit MultivariateStateSpaceTargetFun(
          MultivariateStateSpaceModelBase *model)
          : model_(model)
      {}

      double operator()(const Vector &parameters) {
        Vector old_parameters = model_->vectorize_params();
        model_->unvectorize_params(parameters);
        double ans = model_->log_likelihood();
        model_->unvectorize_params(old_parameters);
        return ans;
      }

     private:
      MultivariateStateSpaceModelBase *model_;
    };
  }  // namespace

  double MSSRM::mle(double epsilon, int ntries) {
    if (has_series_specific_state()) {
      report_error("Maximum likelihood estimation has not been implemented "
                   "in models with series-specific state.");
    }
    return MultivariateStateSpaceModelBase::mle(epsilon, ntries);
  }

  double MSSRM::Estep(bool save_state_distributions) {
    return average_over_latent_data(true, save_state_distributions, nullptr);
  }

  void MSSRM::Mstep(double epsilon) {
    if (observation_model()) {
      observation_model()->find_posterior_mode(epsilon);
    }
    for (int s = 0; s < number_of_state_models(); ++s) {
      state_model(s)->find_posterior_mode(epsilon);
    }
  }

  bool MSSRM::check_that_em_is_legal() const {
    if (observation_model()
        && !observation_model()->can_find_posterior_mode()) {
      return false;
    }
    for (int s = 0; s < number_of_state_models(); ++s) {
      if (!state_model(s)->can_find_posterior_mode()) {
        return false;
      }
    }
    return true;
  }

  //----------------------------------------------------------------------------
  // The math here comes from Durbin and Koopman 7.3.3.  The derivative of the
  // observed data log likelihood equals the expected derivative of the observed
  // data log likelihood, which is closely related to the objective function in
  // the E-step of the EM algorithm.  The expectation involves the full Kalman
  // filter and the disturbance smoother.
  //
  // The return value is the observed data log likelihood at the current
  // parameter values.  If 'update_sufficient_statistics' is true then this
  // implements the Estep of the EM algorithm by setting the complete data
  // sufficient statistics of the observation and state models, so that their
  // MLE's or MAP's can be computed in the M-step.
  //
  // If gradient is not the nullptr then it is filled with the gradient of log
  // likelihood.
  double MSSRM::average_over_latent_data(bool update_sufficient_statistics,
                                         bool save_state_distributions,
                                         Vector *gradient) {
    if (update_sufficient_statistics) {
      clear_client_data();
    }
    if (gradient) {
      *gradient = vectorize_params(true) * 0.0;
    }
    // Compute log likelihood (the return value) and fill the kalman filter with
    // current values.
    kalman_filter();

    // This is the disturbance smoother from Durbin and Koopman, equation
    // (4.69).
    Vector r(state_dimension(), 0.0);
    SpdMatrix N(state_dimension(), 0.0);
    for (int t = time_dimension() - 1; t >= 0; --t) {
      update_observation_model(r, N, t, save_state_distributions,
                               update_sufficient_statistics, gradient);

      // The E step contribution for the observation at time t involves the mean
      // and the variance of the state error from time t-1.
      //
      // The formula for the state error mean in Durbin and Koopman is
      // equation (4.41):   \hat \eta_t = Q_t R'_t r_t.
      //
      // state_error_mean is \hat eta[t-1]
      const Vector state_error_mean = (*state_error_variance(t - 1)) *
                                      state_error_expander(t - 1)->Tmult(r);

      // The formula for the state error variance in Durbin and Koopman is
      // equation (4.47):
      //
      // Var(\eta_t | Y) = Q - QR'NRQ  // all subscripted by _t
      //
      // state_error_posterior_variance is Var(\hat eta[t-1] | Y).
      SpdMatrix state_error_posterior_variance =
          state_error_expander(t - 1)->sandwich_transpose(N);  // transpose??
      state_error_variance(t - 1)->sandwich_inplace(
          state_error_posterior_variance);
      state_error_posterior_variance *= -1;
      state_error_variance(t - 1)->add_to(state_error_posterior_variance);

      if (update_sufficient_statistics) {
        update_state_level_complete_data_sufficient_statistics(
            t - 1, state_error_mean, state_error_posterior_variance);
      }

      if (gradient) {
        update_state_model_gradient(gradient, t - 1, state_error_mean,
                                    state_error_posterior_variance);
      }

      if (save_state_distributions) {
        // Now r is r_{t-1} and N is N_{t-1}.  From Durbin and Koopman (4.32)
        // E(alpha[t] | Y) = a[t] + P * r[t-1]
        // V(alpha[t] | Y) = P[t] - P[t] * N[t-1] * P[t]
        const SpdMatrix &P(get_filter()[t].state_variance());
        get_filter()[t].increment_state_mean(P * r);
        get_filter()[t].increment_state_variance(-1 * sandwich(P, N));
      }
    }
    // The kalman filter is not current because it contains smoothed values.
    double loglike = get_filter().log_likelihood();
    get_filter().set_status(KalmanFilterBase::NOT_CURRENT);
    return loglike;
  }


  void MSSRM::update_state_level_complete_data_sufficient_statistics(
      int t, const Vector &state_error_mean,
      const SpdMatrix &state_error_variance) {
    if (t >= 0) {
      for (int s = 0; s < number_of_state_models(); ++s) {
        state_model(s)->update_complete_data_sufficient_statistics(
            t, const_state_error_component(state_error_mean, s),
            state_error_variance_component(state_error_variance, s));
      }
    }
  }

  void MSSRM::update_state_model_gradient(
      Vector *gradient, int t, const Vector &state_error_mean,
      const SpdMatrix &state_error_variance) {
    if (t >= 0) {
      for (int s = 0; s < number_of_state_models(); ++s) {
        state_model(s)->increment_expected_gradient(
            state_parameter_component(*gradient, s), t,
            const_state_error_component(state_error_mean, s),
            state_error_variance_component(state_error_variance, s));
      }
    }
  }

  void MSSRM::update_observation_model_complete_data_sufficient_statistics(
      int t,
      const Vector &observation_error_mean,
      const Vector &observation_error_variances) {
    report_error("MSSRM::update_observation_model_complete_data_sufficient_statistics is not fully implemented.");
  }

  void MSSRM::update_observation_model_gradient(
      VectorView gradient,
      int t,
      const Vector &observation_error_mean,
      const Vector &observation_error_variances) {
    report_error("MSSRM::update_observation_model_gradient is not fully implemented.");
  }

  //---------------------------------------------------------------------------
  // Implementation of private member functions.
  //---------------------------------------------------------------------------

  void MSSRM::initialize_proxy_models() {
    proxy_models_.clear();
    proxy_models_.reserve(nseries_);
    for (int i = 0; i < nseries_; ++i) {
      proxy_models_.push_back(new ProxyScalarStateSpaceModel(this, i));
    }
  }

  void MSSRM::set_observation_variance_observers() {
    for (int i = 0; i < observation_model_->ydim(); ++i) {
      observation_model_->model(i)->Sigsq_prm()->add_observer(
          this,
          [this]() {this->observation_variance_current_ = false;});
    }
  }

  void MSSRM::set_workspace_observers() {
    std::vector<Ptr<Params>> params = parameter_vector();
    for (auto &el : params) {
      el->add_observer(
          this,
          [this]() {this->workspace_current_ = false;});
    }
  }

  void MSSRM::update_observation_variance() const {
    if (observation_variance_current_) return;
    VectorView elements(observation_variance_.diag());
    for (int i = 0; i < nseries(); ++i) {
      elements[i] = observation_model_->model(i)->sigsq();
    }
    observation_variance_current_ = true;
  }

  void MSSRM::resize_subordinate_state() {
    for (int series = 0; series < nseries(); ++series) {
      proxy_models_[series]->resize_state();
    }
  }

  void MSSRM::observe_state(int t) {
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

  void MSSRM::observe_initial_state() {
    for (int s = 0; s < number_of_state_models(); ++s) {
      ConstVectorView state(state_component(shared_state(0), s));
      state_model(s)->observe_initial_state(state);
    }
  }

  // If an observation is observed, subtract off the time series contribution
  // and add the residual bit to the regression model.
  void MSSRM::observe_data_given_state(int time) {
    for (int series = 0; series < nseries(); ++series) {
      Vector shared_state_contribution =
          *observation_coefficients(time, dummy_selector_) * shared_state(time);
      if (is_observed(series, time)) {
        int index = data_indices_[series][time];
        Ptr<MultivariateTimeSeriesRegressionData> dp = dat()[index];
        double regression_contribution = observed_data(series, time)
            - shared_state_contribution[series]
            - series_specific_state_contribution(series, time);
        observation_model_->model(series)->suf()->add_mixture_data(
            regression_contribution, dp->x(), 1.0);
      }
    }
  }

  // For the moment this function is never called.  It was originally written
  // when the implementation of 'impute_state' required fully observed y's.
  void MSSRM::impute_missing_observations(int t, RNG &rng) {
    // const Selector &observed(observed_status(t));
    // if (observed.nvars() == observed.nvars_possible()) {
    //   // If the observation is fully observed there is nothing to do.
    //   return;
    // } else {
    //   Selector missing(observed.complement());
    //   Vector imputed = *observation_coefficients(t, missing) * shared_state(t);
    //   for (int i = 0; i < missing.nvars(); ++i) {
    //     int series = missing.indx(i);
    //     double shared_effect = imputed[i];
    //     double series_effect = 0;
    //     if (this->has_series_specific_state()
    //         && series_state_dimension(series) > 0) {
    //       series_effect = proxy_models_[series]->observation_matrix(t).dot(
    //           proxy_models_[series]->state(t));
    //     }
    //     int data_index = data_indices_[series][t];
    //     const RegressionModel &regression(*observation_model_->model(series));
    //     double regression_effect = regression.predict(dat()[data_index]->x());
    //     double error = rnorm_mt(rng, 0, regression.sigma());
    //     dat()[data_index]->set_y(shared_effect
    //                              + series_effect
    //                              + regression_effect
    //                              + error);

    //     switch (workspace_status_) {
    //       case UNSET:
    //         // Do nothing.
    //       break;

    //       case ISOLATE_SHARED_STATE:
    //         adjusted_data_workspace_(series, t) = shared_effect + error;
    //         break;

    //       case ISOLATE_SERIES_SPECIFIC_STATE:
    //         adjusted_data_workspace_(series, t) = series_effect + error;
    //         break;

    //       default:
    //         report_error("Unrecognized status for adjusted_data_workspace.");
    //     }
    //   }
    // }
  }

  void MSSRM::impute_shared_state_given_series_state(RNG &rng) {
    resize_subordinate_state();
    workspace_status_ = ISOLATE_SHARED_STATE;
    MultivariateStateSpaceModelBase::impute_state(rng);
    workspace_status_ = UNSET;
  }

  void MSSRM::impute_series_state_given_shared_state(RNG &rng) {
    if (has_series_specific_state()) {
      workspace_status_ = ISOLATE_SERIES_SPECIFIC_STATE;
      for (int s = 0; s < nseries(); ++s) {
        if (proxy_models_[s]->state_dimension() > 0) {
          proxy_models_[s]->impute_state(rng);
        }
      }
      workspace_status_ = UNSET;
    }
  }

  // Set the adjusted data workspace by subtracting regression and
  // series-specific effects from the observed data.
  // void MSSRM::isolate_shared_state() {
  //   adjusted_data_workspace_.resize(nseries(), time_dimension());
  //   for (int time = 0; time < time_dimension(); ++time) {
  //     for (int series = 0; series < nseries(); ++series) {
  //       adjusted_data_workspace_(series, time) = observed_data(series, time);
  //       if (is_observed(series, time)) {
  //         adjusted_data_workspace_(series, time) -=
  //             series_specific_state_contribution(series, time);

  //         // Now subtract off the regression component.
  //         int index = data_indices_[series][time];
  //         Ptr<MultivariateTimeSeriesRegressionData> data_point = dat()[index];
  //         double regression_contribution = observation_model_->model(
  //             series)->predict(data_point->x());
  //         adjusted_data_workspace_(series, time) -= regression_contribution;
  //       }
  //     }
  //   }
  //   workspace_status_ = SHOWS_SHARED_EFFECTS;
  // }

  void MSSRM::isolate_shared_state() {
    workspace_status_ = ISOLATE_SHARED_STATE;
  }

  void MSSRM::isolate_series_specific_state() {
    workspace_status_ = ISOLATE_SERIES_SPECIFIC_STATE;
  }

  void MSSRM::isolate_shared_state(int time) const {
    if (workspace_current_
        && workspace_time_index_ == time
        && workspace_status_ == ISOLATE_SHARED_STATE) {
      return;
    }
    const Selector &observed(observed_status(time));
    adjusted_data_workspace_.resize(observed.nvars());
    for (int series = 0; series < nseries(); ++series) {
      if (observed[series]) {
        int s = observed.dense_index(series);
        adjusted_data_workspace_[s] =
            observed_data(series, time)
            - series_specific_state_contribution(series, time);

        int index = data_index(series, time);
        Ptr<MultivariateTimeSeriesRegressionData> data_point = dat()[index];
        double regression_contribution = observation_model_->model(
            series)->predict(data_point->x());
        adjusted_data_workspace_[s] -= regression_contribution;
      }
    }
    workspace_current_ = true;
    workspace_time_index_ = time;
    workspace_status_ = ISOLATE_SHARED_STATE;
  }

  void MSSRM::isolate_series_specific_state(int time) const {
    if (workspace_status_ == ISOLATE_SERIES_SPECIFIC_STATE
        && workspace_time_index_ == time
        && workspace_current_) {
      return;
    }
    const Selector &observed(observed_status(time));
    adjusted_data_workspace_.resize(observed.nvars());
    Vector shared_state_contribution =
        *observation_coefficients(time, observed) * shared_state(time);
    for (int s = 0; s < observed.nvars(); ++s) {
      int series = observed.sparse_index(s);
      int index = data_index(series, time);
      const Vector &predictors(dat()[index]->x());
      adjusted_data_workspace_[s] = observed_data(series, time)
          - shared_state_contribution[s]
          - observation_model_->model(series)->predict(predictors);
    }
    workspace_current_ = true;
    workspace_time_index_ = time;
    workspace_status_ = ISOLATE_SERIES_SPECIFIC_STATE;
  }

  double MSSRM::series_specific_state_contribution(int series, int time) const {
    if (proxy_models_.empty()) return 0;
    const ProxyScalarStateSpaceModel &proxy(*proxy_models_[series]);
    if (proxy.state_dimension() == 0) {
      return 0;
    } else {
      return proxy.observation_matrix(time).dot(proxy.state(time));
    }
  }

  void MSSRM::set_parameter_observers(Model *model) {
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

}  // namespace BOOM
