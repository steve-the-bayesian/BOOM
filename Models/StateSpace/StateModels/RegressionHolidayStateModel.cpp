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

#include "Models/StateSpace/StateModels/RegressionHolidayStateModel.hpp"
#include "Models/StateSpace/DynamicInterceptRegression.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/StateSpace/StateSpaceRegressionModel.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using RHBI = RegressionHolidayBaseImpl;
  }  // namespace

  RHBI::RegressionHolidayBaseImpl(const Date &time_of_first_observation,
                                  const Ptr<UnivParams> &residual_variance)
      : time_of_first_observation_(time_of_first_observation),
        residual_variance_(residual_variance),
        state_transition_matrix_(new IdentityMatrix(1)),
        state_variance_matrix_(new ZeroMatrix(1)),
        state_error_expander_(new IdentityMatrix(1)),
        state_error_variance_(new ZeroMatrix(1)),
        initial_state_mean_(1, 1.0),
        initial_state_variance_(1, 0.0) {
    if (!residual_variance) {
      report_error("residual_variance must be non-NULL");
    }
  }

  void RHBI::observe_time_dimension(int max_time) {
    if (which_holiday_.size() == max_time) return;
    Date date = time_of_first_observation_;
    which_holiday_.resize(max_time);
    which_day_.resize(max_time);
    for (int t = 0; t < max_time; ++t, ++date) {
      which_holiday_[t] = -1;
      which_day_[t] = -1;
      for (int h = 0; h < holidays_.size(); ++h) {
        if (holidays_[h]->active(date)) {
          // It is possible (but rare) for multiple holidays to be active on the
          // same date.
          if (which_holiday_[t] >= 0) {
            std::ostringstream err;
            err << "More than one holiday is active on " << date
                << ".  This violates a model assumption that only one"
                << " holiday is active at a time.  If you really want to allow"
                << " this behavior, please place the co-occurring holidays in "
                << "different holiday state models.";
            report_error(err.str());
          }
          which_holiday_[t] = h;
          which_day_[t] = holidays_[h]->days_into_influence_window(date);
        }
      }
    }
  }

  void RHBI::add_holiday(const Ptr<Holiday> &holiday) {
    holidays_.push_back(holiday);
  }

  //===========================================================================
  namespace {
    using RHSM = RegressionHolidayStateModel;
  }  // namespace

  RHSM::RegressionHolidayStateModel(const Date &time_of_first_observation,
                                    const Ptr<UnivParams> &residual_variance,
                                    const Ptr<GaussianModel> &prior,
                                    RNG &seeding_rng)
      : impl_(time_of_first_observation, residual_variance),
        prior_(prior),
        rng_(seed_rng(seeding_rng)) {
    if (!prior_) {
      report_error("Prior must not be NULL.");
    }
  }

  RHSM *RHSM::clone() const { return new RHSM(*this); }

  void RHSM::add_holiday(const Ptr<Holiday> &holiday) {
    impl_.add_holiday(holiday);
    int dim = holiday->maximum_window_width();
    holiday_mean_contributions_.push_back(new VectorParams(dim));
    ParamPolicy::add_params(holiday_mean_contributions_.back());
    daily_totals_.push_back(Vector(dim, 0.0));
    daily_counts_.push_back(Vector(dim, 0.0));
  }

  RHSM::RegressionHolidayStateModel(const RHSM &rhs)
      : Model(rhs),
        StateModel(rhs),
        ManyParamPolicy(rhs),
        NullDataPolicy(rhs),
        NullPriorPolicy(rhs),
        impl_(rhs.impl_),
        holiday_mean_contributions_(rhs.holiday_mean_contributions_),
        daily_totals_(rhs.daily_totals_),
        daily_counts_(rhs.daily_counts_),
        prior_(rhs.prior_->clone()),
        rng_(rhs.rng_) {
    for (int i = 0; i < holiday_mean_contributions_.size(); ++i) {
      holiday_mean_contributions_[i] = holiday_mean_contributions_[i]->clone();
      ManyParamPolicy::add_params(holiday_mean_contributions_[i]);
    }
  }

  RHSM &RHSM::operator=(const RHSM &rhs) {
    if (&rhs != this) {
      Model::operator=(rhs);
      StateModel::operator=(rhs);
      ManyParamPolicy::operator=(rhs);
      NullDataPolicy::operator=(rhs);
      impl_ = rhs.impl_;
      holiday_mean_contributions_ = rhs.holiday_mean_contributions_;
      daily_totals_ = rhs.daily_totals_;
      daily_counts_ = rhs.daily_counts_;
      prior_ = rhs.prior_->clone();
      rng_ = rhs.rng_;
      for (int i = 0; i < holiday_mean_contributions_.size(); ++i) {
        holiday_mean_contributions_[i] =
            holiday_mean_contributions_[i]->clone();
        ManyParamPolicy::add_params(holiday_mean_contributions_[i]);
      }
    }
    return *this;
  }

  void RHSM::sample_posterior() {
    int number_of_holidays = holiday_mean_contributions_.size();
    for (int holiday = 0; holiday < number_of_holidays; ++holiday) {
      Vector holiday_pattern = holiday_mean_contributions_[holiday]->value();
      for (int day = 0; day < holiday_pattern.size(); ++day) {
        double posterior_precision =
            daily_counts_[holiday][day] / residual_variance() +
            1.0 / prior_->sigsq();
        double posterior_mean =
            daily_totals_[holiday][day] / residual_variance() +
            prior_->mu() / prior_->sigsq();
        posterior_mean /= posterior_precision;
        double posterior_sd = sqrt(1.0 / posterior_precision);
        holiday_pattern[day] = rnorm_mt(rng_, posterior_mean, posterior_sd);
      }
      holiday_mean_contributions_[holiday]->set(holiday_pattern);
    }
  }

  void RHSM::observe_time_dimension(int max_time) {
    impl_.observe_time_dimension(max_time);
  }

  void RHSM::observe_state(const ConstVectorView &then,
                           const ConstVectorView &now, int time_now,
                           ScalarStateSpaceModelBase *model) {
    int holiday = impl_.which_holiday(time_now);
    if (holiday < 0) return;
    int day = impl_.which_day(time_now);
    double residual =
        model->adjusted_observation(time_now) -
        model->observation_matrix(time_now).dot(model->state(time_now)) +
        this->observation_matrix(time_now).dot(now);
    daily_totals_[holiday][day] += residual;
    daily_counts_[holiday][day] += 1.0;
  }

  void RHSM::observe_dynamic_intercept_regression_state(
      const ConstVectorView &then, const ConstVectorView &now, int time_now,
      DynamicInterceptRegressionModel *model) {
    int holiday = impl_.which_holiday(time_now);
    if (holiday < 0) return;
    int day = impl_.which_day(time_now);
    Ptr<StateSpace::MultiplexedRegressionData> full_data =
        model->dat()[time_now];
    if (full_data->missing() == Data::missing_status::completely_missing) {
      return;
    }
    for (int i = 0; i < full_data->total_sample_size(); ++i) {
      if (full_data->regression_data(i).missing() ==
          Data::missing_status::observed) {
        double residual = full_data->regression_data(i).y() -
                          model->conditional_mean(time_now, i) +
                          this->observation_matrix(time_now).dot(now);
        daily_counts_[holiday][day] += 1.0;
        daily_totals_[holiday][day] += residual;
      }
    }
  }

  SparseVector RHSM::observation_matrix(int time_now) const {
    SparseVector ans(1);
    int holiday = impl_.which_holiday(time_now);
    if (holiday < 0) return ans;
    int day = impl_.which_day(time_now);
    ans[0] = holiday_mean_contributions_[holiday]->value()[day];
    return ans;
  }

}  // namespace BOOM
