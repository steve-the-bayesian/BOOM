// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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
#include "Models/StateSpace/StateSpaceRegressionModel.hpp"
#include "Models/DataTypes.hpp"
#include "Models/StateSpace/Filters/SparseKalmanTools.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "cpputil/seq.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    typedef StateSpaceRegressionModel SSRM;
    typedef StateSpace::MultiplexedRegressionData MRD;
  }  // namespace

  MRD::MultiplexedRegressionData() : state_model_offset_(0) {}
  MRD::MultiplexedRegressionData(double y, const Vector &x)
      : state_model_offset_(0) {
    NEW(RegressionData, data_point)(y, x);
    add_data(data_point);
  }

  MRD::MultiplexedRegressionData(const std::vector<Ptr<RegressionData>> &data)
      : state_model_offset_(0) {
    for (const auto &d : data) {
      add_data(d);
    }
  }

  MRD *MRD::clone() const { return new MRD(*this); }

  std::ostream &MRD::display(std::ostream &out) const {
    out << "state model offset: " << state_model_offset_ << std::endl
        << std::setw(10) << " response "
        << " predictors " << std::endl;
    for (int i = 0; i < regression_data_.size(); ++i) {
      out << std::setw(10) << regression_data_[i]->y() << " "
          << regression_data_[i]->x() << std::endl;
    }
    return out;
  }

  void MRD::add_data(const Ptr<RegressionData> &dp) {
    MultiplexedData::add_data(dp);
    regression_data_.push_back(dp);
    predictors_.rbind(dp->x());
  }

  double MRD::adjusted_observation(const GlmCoefs &coefficients) const {
    if (missing() == Data::completely_missing || observed_sample_size() == 0) {
      return negative_infinity();
    }
    double ans = 0;
    for (int i = 0; i < regression_data_.size(); ++i) {
      const RegressionData &observation(regression_data(i));
      if (observation.missing() == Data::observed) {
        ans += observation.y() - coefficients.predict(observation.x());
      }
    }
    return ans / observed_sample_size();
  }

  const RegressionData &MRD::regression_data(int i) const {
    return *(regression_data_[i]);
  }

  Ptr<RegressionData> MRD::regression_data_ptr(int i) {
    return regression_data_[i];
  }

  //======================================================================
  void SSRM::setup() {
    regression_->only_keep_sufstats(true);
  }

  SSRM::StateSpaceRegressionModel(int xdim)
      : regression_(new RegressionModel(xdim)) {
    setup();
    // Note that in this constructor the regression model will still need to
    // have data added, so we can't call fix_xtx().  This means that the xtx
    // matrix will be re-computed with each trip through the data.
  }

  SSRM::StateSpaceRegressionModel(const Vector &y, const Matrix &X,
                                  const std::vector<bool> &observed)
      : regression_(new RegressionModel(ncol(X))) {
    setup();
    int n = y.size();
    if (nrow(X) != n) {
      ostringstream msg;
      msg << "X and y are incompatible in constructor for "
          << "StateSpaceRegressionModel." << endl
          << "length(y) = " << n << endl
          << "nrow(X) = " << nrow(X) << endl;
      report_error(msg.str());
    }

    for (int i = 0; i < n; ++i) {
      NEW(RegressionData, dp)(y[i], X.row(i));
      if (!(observed.empty()) && !observed[i]) {
        dp->set_missing_status(Data::missing_status::completely_missing);
      }
      add_data(dp);
    }

    // The cast is necessary because the regression model stores a Ptr
    // to a base class that does not supply fix_xtx();
    regression_->suf().dcast<NeRegSuf>()->fix_xtx();
  }

  SSRM::StateSpaceRegressionModel(const SSRM &rhs)
      : Model(rhs),
        ScalarStateSpaceModelBase(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        regression_(rhs.regression_->clone()) {
    setup();
  }

  SSRM *SSRM::clone() const { return new SSRM(*this); }

  void SSRM::add_data(const Ptr<Data> &dp) {
    Ptr<RegressionData> regression_data = dp.dcast<RegressionData>();
    if (!!regression_data) {
      add_regression_data(regression_data);
      return;
    }

    Ptr<MRD> multiplexed_data = dp.dcast<MRD>();
    if (!!multiplexed_data) {
      add_multiplexed_data(multiplexed_data);
      return;
    }
    report_error("Could not cast to an appropriate data type.");
  }

  void SSRM::add_regression_data(const Ptr<RegressionData> &dp) {
    NEW(MRD, multiplexed_data)();
    multiplexed_data->add_data(dp);
    multiplexed_data->set_missing_status(dp->missing());
    add_data(multiplexed_data);
  }

  void SSRM::add_multiplexed_data(const Ptr<MRD> &dp) {
    DataPolicy::add_data(dp);
    for (int i = 0; i < dp->total_sample_size(); ++i) {
      regression_model()->add_data(dp->regression_data_ptr(i));
    }
  }

  double SSRM::observation_variance(int t) const {
    const std::vector<Ptr<MRD>> &data(dat());
    double sigsq = regression_->sigsq();
    if (t >= data.size()) {
      return sigsq;
    } else {
      int n = data[t]->observed_sample_size();
      if (n == 0) ++n;
      return sigsq / n;
    }
  }

  double SSRM::adjusted_observation(int t) const {
    return dat()[t]->adjusted_observation(regression_->coef());
  }

  bool SSRM::is_missing_observation(int t) const {
    return dat()[t]->missing() == Data::completely_missing ||
           dat()[t]->observed_sample_size() == 0;
  }

  void SSRM::observe_data_given_state(int t) {
    if (!is_missing_observation(t)) {
      Ptr<MRD> dp(dat()[t]);
      double state_mean = observation_matrix(t).dot(state(t));
      for (int i = 0; i < dp->total_sample_size(); ++i) {
        const RegressionData &observation(dp->regression_data(i));
        if (observation.missing() == Data::observed) {
          regression_->suf()->add_mixture_data(observation.y() - state_mean,
                                               observation.x(), 1.0);
        }
      }
    }
  }

  Matrix SSRM::forecast(const Matrix &newX) {
    kalman_filter();
    Kalman::ScalarMarginalDistribution marg = get_filter().back();
    Matrix ans(nrow(newX), 2);
    int t0 = time_dimension();
    for (int t = 0; t < nrow(ans); ++t) {
      ans(t, 0) = regression_->predict(newX.row(t)) +
          observation_matrix(t + t0).dot(marg.state_mean());
      marg.update(0, true, t + t0);
      ans(t, 1) = sqrt(marg.prediction_variance());
    }
    return ans;
  }

  // TODO:  test simulate_forecast
  Vector SSRM::simulate_forecast(RNG &rng, const Matrix &newX,
                                 const Vector &final_state) {
    return simulate_multiplex_forecast(rng, newX, final_state,
                                       seq<int>(0, nrow(newX) - 1));
  }

  Vector SSRM::simulate_forecast(RNG &rng, const Matrix &newX) {
    ScalarStateSpaceModelBase::set_state_model_behavior(StateModel::MARGINAL);
    kalman_filter();
    // The Kalman filter produces the forecast distribution for the next time
    // period.  Since the observed data goes from 0 to t-1, the final element of
    // the filter contains the forecast distribution for time t.
    Vector final_state = rmvn_robust_mt(
        rng,
        get_filter().back().state_mean(),
        get_filter().back().state_variance());
    return simulate_forecast(rng, newX, final_state);
  }

  Matrix SSRM::simulate_forecast_components(RNG &rng, const Matrix &newX,
                                            const Vector &final_state) {
    ScalarStateSpaceModelBase::set_state_model_behavior(StateModel::MARGINAL);
    int forecast_horizon = newX.nrow();
    Matrix ans(number_of_state_models() + 2, forecast_horizon, 0.0);
    int t0 = time_dimension();
    Vector state = final_state;
    for (int t = 0; t < forecast_horizon; ++t) {
      state = simulate_next_state(rng, state, t + t0);
      for (int s = 0; s < number_of_state_models(); ++s) {
        ans(s, t) = state_model(s)->observation_matrix(t + t0).dot(
            state_component(state, s));
      }
      ans(number_of_state_models(), t) = regression_->predict(newX.row(t));
      ans.col(t).back() = rnorm_mt(rng,
                                   ans.col(t).sum(),
                                   observation_variance(t + t0));
    }
    return ans;
  }

  Vector SSRM::simulate_multiplex_forecast(RNG &rng,
                                           const Matrix &newX,
                                           const Vector &final_state,
                                           const std::vector<int> &timestamps) {
    ScalarStateSpaceModelBase::set_state_model_behavior(StateModel::MARGINAL);
    int forecast_dimension = timestamps.size();
    if (nrow(newX) != forecast_dimension) {
      report_error("Dimensions of timestamps and newX don't agree.");
    }
    Vector ans(forecast_dimension);
    int t0 = time_dimension();
    Vector state = final_state;
    // The time stamp of "final state" is t0 - 1.
    int time = -1;
    for (int i = 0; i < forecast_dimension; ++i) {
      advance_to_timestamp(rng, time, state, timestamps[i], i);
      ans[i] = rnorm_mt(rng, observation_matrix(t0 + time).dot(state),
                        sqrt(observation_variance(t0 + time)));
      ans[i] += regression_->predict(newX.row(i));
    }
    return ans;
  }

  Vector SSRM::one_step_holdout_prediction_errors(
      const Matrix &newX, const Vector &newY, const Vector &final_state,
      bool standardize) const {
    if (nrow(newX) != length(newY)) {
      report_error(
          "X and Y do not match in StateSpaceRegressionModel::"
          "one_step_holdout_prediction_errors");
    }

    Vector ans(nrow(newX));
    int t0 = time_dimension();
    Kalman::ScalarMarginalDistribution marg(this, nullptr, 0);
    marg.set_state_mean(*state_transition_matrix(t0 - 1) * final_state);
    marg.set_state_variance(SpdMatrix(state_variance_matrix(t0 - 1)->dense()));

    for (int t = 0; t < ans.size(); ++t) {
      bool missing = false;
      marg.update(newY[t] - regression_model()->predict(newX.row(t)),
                  missing, t + t0);
      ans[t] = marg.prediction_error();
      if (standardize) {
        ans[t] /= sqrt(marg.prediction_variance());
      }
    }
    return ans;
  }

  Matrix SSRM::simulate_holdout_prediction_errors(
      int niter, int cutpoint_number, bool standardize) {
    Matrix ans(niter, time_dimension());
    SubMatrix training_prediction_errors(
        ans, 0, niter - 1, 0, cutpoint_number - 1);
    SubMatrix holdout_prediction_errors(
        ans, 0, niter - 1, cutpoint_number, ncol(ans) - 1);
    std::vector<Ptr<Data>> training_data(dat().begin(), dat().begin() + cutpoint_number);
    std::vector<Ptr<StateSpace::MultiplexedRegressionData>> holdout_data(
        dat().begin() + cutpoint_number, dat().end());
    clear_data();
    for (const auto &data_point : training_data) {
      add_data(data_point);
    }
    Matrix holdout_predictors(holdout_data.size(), xdim());
    Vector holdout_response(holdout_data.size());
    for (int i = 0; i < holdout_data.size(); ++i) {
      if (holdout_data[i]->total_sample_size() != 1) {
        report_error("simulate_holdout_prediction_errors does "
                     "not work with multiplex data.");
      }
      holdout_response[i] = holdout_data[i]->regression_data(0).y();
      holdout_predictors.row(i) = holdout_data[i]->regression_data(0).x();
    }

    for (int i = 0; i < niter; ++i) {
      sample_posterior();
      training_prediction_errors.row(i) = one_step_prediction_errors(
          standardize);
      holdout_prediction_errors.row(i) = one_step_holdout_prediction_errors(
          holdout_predictors,
          holdout_response,
          state().last_col(),
          standardize);
    }
    return ans;
  }

  Vector SSRM::regression_contribution() const {
    const std::vector<Ptr<MRD>> &data(dat());
    Vector ans(data.size());
    for (int time = 0; time < data.size(); ++time) {
      Ptr<MRD> dp = data[time];
      double average_contribution = 0;
      for (int j = 0; j < data[time]->total_sample_size(); ++j) {
        average_contribution +=
            regression_model()->predict(dp->regression_data(j).x());
      }
      ans[time] = dp->total_sample_size() > 0
                      ? average_contribution /= dp->total_sample_size()
                      : 0;
    }
    return ans;
  }

}  // namespace BOOM
