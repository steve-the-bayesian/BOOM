// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#include "Models/StateSpace/StateSpaceStudentRegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/TDataImputer.hpp"
#include "Models/StateSpace/Filters/SparseKalmanTools.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/seq.hpp"
#include "distributions.hpp"
#include "stats/moments.hpp"

namespace BOOM {
  namespace {
    typedef StateSpaceStudentRegressionModel SSSRM;
    typedef StateSpace::AugmentedStudentRegressionData AugmentedData;
  }  // namespace

  AugmentedData::AugmentedStudentRegressionData(double y, const Vector &x)
      : RegressionData(y, x),
        weight_(1.0),
        state_model_offset_(0.0)
  {}

  AugmentedData *AugmentedData::clone() const {
    return new AugmentedData(*this);
  }

  std::ostream &AugmentedData::display(std::ostream &out) const {
    RegressionData::display(out)
        << "\n"
        << "state model offset: " << state_model_offset_ << std::endl
        << "weight: " << weight_ << std::endl;
    return out;
  }

  void AugmentedData::set_weight(double weight) {
    if (weight < 0 || !std::isfinite(weight)) {
      report_error("Weight must be finite and non-negative.");
    }
    weight_ = weight;
  }

  double AugmentedData::adjusted_observation(
      const GlmCoefs &coefficients) const {
    if (this->missing() == Data::observed) {
      return y() - coefficients.predict(x());
    } else {
      return negative_infinity();
    }
  }

  void AugmentedData::set_state_model_offset(double offset) {
    state_model_offset_ = offset;
  }

  //======================================================================
  SSSRM::StateSpaceStudentRegressionModel(int xdim)
      : StateSpaceNormalMixture(xdim > 1),
        observation_model_(new TRegressionModel(xdim)) {
  }

  SSSRM::StateSpaceStudentRegressionModel(const Vector &response,
                                          const Matrix &predictors,
                                          const std::vector<bool> &observed)
      : StateSpaceNormalMixture(ncol(predictors) > 0),
        observation_model_(new TRegressionModel(ncol(predictors))) {
    if ((ncol(predictors) == 1) &&
        (var(predictors.col(0)) < std::numeric_limits<double>::epsilon())) {
      set_regression_flag(false);
    }

    if (!observed.empty()) {
      if (observed.size() != response.size()) {
        report_error(
            "Argument size mismatch between response and observed in "
            "StateSpaceStudentRegressionModel constructor.");
      }
    }
    for (int i = 0; i < response.size(); ++i) {
      NEW(AugmentedData, data_point)(response[i], predictors.row(i));
      if (!observed.empty() && !observed[i]) {
        data_point->set_missing_status(
            Data::missing_status::completely_missing);
      }
      add_data(data_point);
    }
  }

  SSSRM::StateSpaceStudentRegressionModel(const SSSRM &rhs)
      : Model(rhs),
        StateSpaceNormalMixture(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        observation_model_(rhs.observation_model_->clone()) {}

  SSSRM *SSSRM::clone() const { return new SSSRM(*this); }

  int SSSRM::time_dimension() const { return dat().size(); }

  double SSSRM::observation_variance(int t) const {
    if (t >= time_dimension() || dat()[t]->missing() != Data::observed) {
      return student_marginal_variance();
    } else if (dat()[t]->weight() > 0) {
      return observation_model_->sigsq() / dat()[t]->weight();
    } else {
      return student_marginal_variance();
    }
  }

  double SSSRM::adjusted_observation(int t) const {
    if (is_missing_observation(t)) {
      return negative_infinity();
    }
    return dat()[t]->adjusted_observation(observation_model_->coef());
  }

  bool SSSRM::is_missing_observation(int t) const {
    return dat()[t]->missing() != Data::observed;
  }

  void SSSRM::observe_data_given_state(int t) {
    if (!is_missing_observation(t)) {
      dat()[t]->set_state_model_offset(observation_matrix(t).dot(state(t)));
      signal_complete_data_change(t);
    }
  }

  Matrix SSSRM::simulate_forecast_components(
      RNG &rng, const Matrix &predictors, const Vector &final_state) {
    set_state_model_behavior(StateModel::MARGINAL);
    int forecast_horizon = nrow(predictors);
    Matrix ans(number_of_state_models() + 2, forecast_horizon, 0.0);
    int t0 = time_dimension();
    Vector state = final_state;
    double sigma = observation_model_->sigma();
    double nu = observation_model_->nu();
    for (int t = 0; t < forecast_horizon; ++t) {
      state = simulate_next_state(rng, state, t + t0);
      for (int s = 0; s < number_of_state_models(); ++s) {
        ans(s, t) = state_model(s)->observation_matrix(t + t0).dot(
            state_component(state, s));
      }
      ans(number_of_state_models(), t) =
          observation_model_->predict(predictors.row(t));
      ans.col(t).back() = rstudent_mt(rng, ans.col(t).sum(), sigma, nu);
    }
    return ans;
  }

  Vector SSSRM::simulate_forecast(
      RNG &rng,
      const Matrix &predictors,
      const Vector &final_state) {
    set_state_model_behavior(StateModel::MARGINAL);
    Vector state = final_state;
    Vector ans(nrow(predictors));
    int t0 = dat().size();
    double sigma = observation_model_->sigma();
    double nu = observation_model_->nu();
    // The time stamp of "final state" is t0 - 1.
    int time = -1;
    for (int i = 0; i < nrow(predictors); ++i) {
      advance_to_timestamp(rng, time, state, i, i);
      double mu = observation_model_->predict(predictors.row(i)) +
                  observation_matrix(time + t0).dot(state);
      ans[i] = rstudent_mt(rng, mu, sigma, nu);
    }
    return ans;
  }

  // Find the one step prediction errors for a holdout sample.
  Vector SSSRM::one_step_holdout_prediction_errors(RNG &rng,
                                                   const Vector &response,
                                                   const Matrix &predictors,
                                                   const Vector &final_state,
                                                   bool standardize) {
    TDataImputer data_imputer;
    if (nrow(predictors) != response.size()) {
      report_error(
          "Size mismatch in arguments provided to "
          "one_step_holdout_prediction_errors.");
    }
    Vector ans(response.size());
    int t0 = dat().size();

    double sigma = observation_model_->sigma();
    double nu = observation_model_->nu();
    Kalman::ScalarMarginalDistribution marg(this, nullptr, 0);
    marg.set_state_mean(*state_transition_matrix(t0 - 1) * final_state);
    marg.set_state_variance(SpdMatrix(state_variance_matrix(t0 - 1)->dense()));

    for (int t = 0; t < ans.size(); ++t) {
      bool missing = false;
      // 1) simulate next state.
      // 2) simulate w_t given state
      // 3) kalman update state given w_t.
      double state_contribution =
          observation_matrix(t + t0).dot(marg.state_mean());
      double regression_contribution =
          observation_model_->predict(predictors.row(t));
      double mu = state_contribution + regression_contribution;
      ans[t] = response[t] - mu;
      if (standardize) {
        ans[t] /= sqrt(marg.prediction_variance());
      }

      // ans[t] is a random draw of the one step ahead prediction
      // error at time t0+t given observed data to time t0+t-1.  We
      // now proceed with the steps needed to update the Kalman filter
      // so we can compute ans[t+1].

      double weight = data_imputer.impute(rng, response[t] - mu, sigma, nu);
      // The latent state was drawn from its predictive distribution given Y[t0
      // + t -1] and used to impute the latent data for y[t0+t].  That latent
      // data is now used to update the Kalman filter for the next time period.
      // It is important that we discard the imputed state at this point.
      marg.update(response[t] - regression_contribution, missing,
                  t + t0, 1.0 / weight);
    }
    return ans;
  }

  double SSSRM::student_marginal_variance() const {
    double nu = observation_model_->nu();
    double sigsq = observation_model_->sigsq();
    return nu > 2 ? sigsq * nu / (nu - 2) : sigsq * 1e+8;
  }

  Matrix SSSRM::simulate_holdout_prediction_errors(
      int niter, int cutpoint_number, bool standardize) {
    Matrix ans(niter, time_dimension());
    SubMatrix training_prediction_errors(
        ans, 0, niter - 1, 0, cutpoint_number - 1);
    SubMatrix holdout_prediction_errors(
        ans, 0, niter - 1, cutpoint_number, ncol(ans) - 1);
    std::vector<Ptr<Data>> training_data(dat().begin(), dat().begin() + cutpoint_number);
    std::vector<Ptr<StateSpace::AugmentedStudentRegressionData>> holdout_data(
        dat().begin() + cutpoint_number, dat().end());
    clear_data();
    for (const auto &data_point : training_data) {
      add_data(data_point);
    }
    Matrix holdout_predictors(holdout_data.size(), xdim());
    Vector holdout_response(holdout_data.size());
    for (int i = 0; i < holdout_data.size(); ++i) {
      holdout_response[i] = holdout_data[i]->y();
      holdout_predictors.row(i) = holdout_data[i]->x();
    }

    for (int i = 0; i < niter; ++i) {
      sample_posterior();
      training_prediction_errors.row(i) = one_step_prediction_errors(
          standardize);
      holdout_prediction_errors.row(i) = one_step_holdout_prediction_errors(
          PriorPolicy::rng(),
          holdout_response,
          holdout_predictors,
          state().last_col(),
          standardize);
    }
    return ans;
  }

}  // namespace BOOM
