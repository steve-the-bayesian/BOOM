/*
  Copyright (C) 2005-2018 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#include "Models/StateSpace/StateModels/HierarchicalRegressionHolidayStateModel.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/StateSpace/StateSpaceRegressionModel.hpp"
#include "Models/StateSpace/DynamicInterceptRegression.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using HRHSM = HierarchicalRegressionHolidayStateModel;
  }  // namespace

  HRHSM::HierarchicalRegressionHolidayStateModel(
      const Date &time_of_first_observation,
      const Ptr<UnivParams> &residual_variance)
      : time_of_first_observation_(time_of_first_observation),
        residual_variance_(residual_variance),
        state_transition_matrix_(new IdentityMatrix(1)),
        state_variance_matrix_(new ZeroMatrix(1)),
        state_error_expander_(new EmptyMatrix),
        state_error_variance_(new EmptyMatrix),
        model_(nullptr),
        initial_state_mean_(1, 1.0),
        initial_state_variance_(1, 0.0)
  {}
  
  void HRHSM::add_holiday(const Ptr<Holiday> &holiday) {
    if (!holidays_.empty()) {
      if (holiday->maximum_window_width() !=
          holidays_[0]->maximum_window_width()) {
        report_error("All holidays must have the same window width.");
      }
    }
    holidays_.push_back(holiday);
    int dim = holiday->maximum_window_width();
    if (!model_) {
      NEW(MvnModel, prior_model)(dim);
      model_.reset(new HierarchicalGaussianRegressionModel(
          prior_model, residual_variance_));
    }
    NEW(RegressionModel, holiday_model)(
        holidays_.back()->maximum_window_width());
    model_->add_model(holiday_model);

    if (daily_dummies_.empty()) {
      for (int i = 0; i < dim; ++i) {
        Vector x(dim, 0.0);
        x[i] = 1.0;
        daily_dummies_.push_back(x);
      }
    }
  }

  void HRHSM::observe_time_dimension(int max_time) {
    Date date = time_of_first_observation_;
    which_holiday_.resize(max_time);
    which_day_.resize(max_time);
    for (int t = 0; t < max_time; ++t) {
      for (int h = 0; h < holidays_.size(); ++h) {
        if (holidays_[h]->active(date)) {
          which_holiday_[t] = h;
          which_day_[t] = holidays_[h]->days_into_influence_window(date);
        } else {
          which_holiday_[t] = -1;
          which_day_[t] = -1;
        }
      }
    }
  }
  
  void HRHSM::observe_state(const ConstVectorView &then,
                            const ConstVectorView &now,
                            int time_now,
                            ScalarStateSpaceModelBase *model) {
    int which_model = which_holiday_[time_now];
    if (which_model < 0) {
      return;
    }
    int day = which_day_[time_now];
    // The residual contains the observed data minus the contributions from all
    // state models but this one.
    double residual = model->adjusted_observation(time_now) - 
        model->observation_matrix(time_now).dot(model->state(time_now))
        + this->observation_matrix(time_now).dot(now);
    model_->data_model(which_model)->suf()->add_mixture_data(
        residual, daily_dummies(day), 1.0);
  }

  void HRHSM::observe_dynamic_intercept_regression_state(
      const ConstVectorView &then,
      const ConstVectorView &now,
      int time_now,
      DynamicInterceptRegressionModel *model) {
    int which_model = which_holiday_[time_now];
    if (which_model < 0) {
      return;
    }
    int day = which_day_[time_now];
    // The residual contains the observed data minus the contributions from all
    // state models but this one.
    Ptr<StateSpace::MultiplexedRegressionData>
        full_data = model->dat()[time_now];
    if (full_data->missing() == Data::missing_status::completely_missing) {
      return;
    }
    int nobs = full_data->total_sample_size();
    for (int i = 0; i < nobs; ++i) {
      if (full_data->regression_data(i).missing() ==
          Data::missing_status::observed) {
        double residual = full_data->regression_data(i).y()
            - model->conditional_mean(time_now, i)
            + this->observation_matrix(time_now).dot(now);
        model_->data_model(which_model)->suf()->add_mixture_data(
            residual, daily_dummies(day), 1.0);
      }
    }
  }

  SparseVector HRHSM::observation_matrix(int t) const {
    SparseVector ans(1);
    if (which_holiday_[t] < 0) {
      return ans;
    }
    ans[0] = model_->data_model(which_holiday_[t])->Beta()[which_day_[t]];
    return ans;
  }

  void HRHSM::clear_data() {
    if (!!model_) model_->clear_data_keep_models();
  }
  
}  // namespace BOOM
