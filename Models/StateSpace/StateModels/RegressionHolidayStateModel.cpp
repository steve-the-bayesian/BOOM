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
#include "Models/ZeroMeanGaussianModel.hpp"
#include "Models/Glm/TRegression.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using Impl = RegressionHolidayBaseImpl;
  }  // namespace

  Impl::RegressionHolidayBaseImpl(const Date &time_of_first_observation,
                                  const Ptr<UnivParams> &residual_variance)
      : time_of_first_observation_(time_of_first_observation),
        residual_variance_(residual_variance),
        state_transition_matrix_(new IdentityMatrix(1)),
        state_variance_matrix_(new ZeroMatrix(1)),
        state_error_expander_(new IdentityMatrix(1)),
        state_error_variance_(new ZeroMatrix(1)),
        initial_state_mean_(1, 1.0),
        initial_state_variance_(1, 0.0) {
  }

  void Impl::observe_time_dimension(int max_time) {
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

  void Impl::add_holiday(const Ptr<Holiday> &holiday) {
    holidays_.push_back(holiday);
  }

  Ptr<UnivParams> Impl::extract_residual_variance_parameter(
      ScalarStateSpaceModelBase &model) {
    if (ZeroMeanGaussianModel *gaussian =
        dynamic_cast<ZeroMeanGaussianModel *>(model.observation_model())) {
      return gaussian->Sigsq_prm();
    } else if (RegressionModel *reg =
               dynamic_cast<RegressionModel *>(model.observation_model())) {
      return reg->Sigsq_prm();
    } else if (TRegressionModel *student_reg =
               dynamic_cast<TRegressionModel *>(model.observation_model())) {
      return student_reg->Sigsq_prm();
      /////////////
      // TODO: expose the student variance inflator from the observation equation.
    } else {
      report_error("Cannot extract residual variance parameter.");
    }
    return Ptr<UnivParams>(nullptr);
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
        // TODO: Consider replacing 'residual_variance' with a set of weighted
        // Gaussian sufficient statistics, to be augmented when we
        // observe_data().  This is the only place where the residual_variance
        // is used.
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

  SparseVector RHSM::observation_matrix(int time_now) const {
    SparseVector ans(1);
    int holiday = impl_.which_holiday(time_now);
    if (holiday < 0) {
      return ans;
    }
    int day = impl_.which_day(time_now);
    ans[0] = holiday_mean_contributions_[holiday]->value()[day];
    return ans;
  }

  ScalarRegressionHolidayStateModel::ScalarRegressionHolidayStateModel(
      const Date &time_of_first_observation,
      ScalarStateSpaceModelBase *model,
      const Ptr<GaussianModel> &prior,
      RNG &seeding_rng)
      : RegressionHolidayStateModel(
            time_of_first_observation,
            Impl::extract_residual_variance_parameter(*model),
            prior,
            seeding_rng),
          model_(model)
    {}

  void ScalarRegressionHolidayStateModel::observe_state(
      const ConstVectorView &then, const ConstVectorView &now, int time_now) {
    if (!model_->is_missing_observation(time_now)) {
      int holiday = impl().which_holiday(time_now);
      if (holiday < 0) return;
      int day = impl().which_day(time_now);
      double residual =
          model_->adjusted_observation(time_now) -
          model_->observation_matrix(time_now).dot(model_->state(time_now)) +
          this->observation_matrix(time_now).dot(now);
      increment_daily_suf(holiday, day, residual, 1.0);
    }
  }

  namespace {
    using DIRHSM = DynamicInterceptRegressionHolidayStateModel;
  }
  
  DIRHSM::DynamicInterceptRegressionHolidayStateModel(
      const Date &time_of_first_observation,
      DynamicInterceptRegressionModel *model,
      const Ptr<GaussianModel> &prior,
      RNG &seeding_rng)
      : RegressionHolidayStateModel(
            time_of_first_observation,
            model->observation_model()->Sigsq_prm(),
            prior,
            seeding_rng),
        model_(model)
  {}

  
  void DIRHSM::observe_state(const ConstVectorView &then,
                             const ConstVectorView &now,
                             int time_now) {
    int holiday = impl().which_holiday(time_now);
    if (holiday < 0) return;
    int day = impl().which_day(time_now);
    Ptr<StateSpace::TimeSeriesRegressionData> data = model_->dat()[time_now];
    if (data->missing() == Data::missing_status::completely_missing) {
      return;
    }
    Vector residuals = data->response() - model_->conditional_mean(time_now);
    residuals += this->observation_matrix(time_now).dot(now);
    increment_daily_suf(holiday, day, sum(residuals), residuals.size());
  }

  Ptr<SparseMatrixBlock> DIRHSM::observation_coefficients(
      int t,
      const StateSpace::TimeSeriesRegressionData &data_point) const {
    return new IdenticalRowsMatrix(
        observation_matrix(t), data_point.sample_size());
  }

  
}  // namespace BOOM
