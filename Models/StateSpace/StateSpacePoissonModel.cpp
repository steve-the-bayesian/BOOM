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

#include "Models/StateSpace/StateSpacePoissonModel.hpp"
#include "Models/Glm/PosteriorSamplers/PoissonDataImputer.hpp"
#include "Models/StateSpace/Filters/SparseKalmanTools.hpp"
#include "cpputil/Constants.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/seq.hpp"
#include "distributions.hpp"
#include "stats/moments.hpp"

namespace BOOM {
  namespace {
    typedef StateSpacePoissonModel SSPM;
    typedef StateSpace::AugmentedPoissonRegressionData APRD;
  }  // namespace


  APRD::AugmentedPoissonRegressionData(double count, double exposure,
                                       const Vector &predictors)
      : PoissonRegressionData(count, predictors, exposure),
        latent_continuous_value_(0.0),
        precision_(1.0),
        state_model_offset_(0.0)
  {}

  APRD *APRD::clone() const { return new APRD(*this); }

  std::ostream &APRD::display(std::ostream &out) const {
    out << *this << std::endl
        << "latent continuous value: " << latent_continuous_value_ << std::endl
        << "precision:               " << precision_ << std::endl
        << "state model offset     : " << state_model_offset_ << std::endl;
    return out;
  }

  void APRD::set_latent_data(double value, double precision) {
    if (precision < 0) {
      report_error("precision must be non-negative.");
    }
    latent_continuous_value_ = value;
    precision_ = precision;
  }

  double APRD::latent_data_variance() const {
    return 1.0 / precision_;
  }

  double APRD::latent_data_value() const {
    return latent_continuous_value_;
  }

  double APRD::adjusted_observation(const GlmCoefs &coefficients) const {
    if (missing() != Data::observed) {
      return negative_infinity();
    }
    return latent_continuous_value_ - coefficients.predict(x());
  }

  void APRD::set_state_model_offset(double offset) {
    state_model_offset_ = offset;
  }

  //======================================================================
  SSPM::StateSpacePoissonModel(int xdim)
      : StateSpaceNormalMixture(xdim > 1),
        observation_model_(new PoissonRegressionModel(xdim)) {}

  SSPM::StateSpacePoissonModel(const Vector &counts,
                               const Vector &exposure,
                               const Matrix &design_matrix,
                               const std::vector<bool> &observed)
      : StateSpaceNormalMixture(ncol(design_matrix) > 0),
        observation_model_(new PoissonRegressionModel(ncol(design_matrix))) {
    if ((ncol(design_matrix) == 1) &&
        (var(design_matrix.col(0)) < std::numeric_limits<double>::epsilon())) {
      set_regression_flag(false);
    }
    bool all_observed = observed.empty();
    if (counts.size() != exposure.size() ||
        counts.size() != nrow(design_matrix) ||
        (!all_observed && counts.size() != observed.size())) {
      report_error(
          "Data sizes do not match in StateSpacePoissonModel "
          "constructor");
    }
    for (int i = 0; i < counts.size(); ++i) {
      bool missing = !(all_observed || observed[i]);
      NEW(APRD, dp)(missing ? 0 : counts[i],
                    missing ? 0 : exposure[i],
                    design_matrix.row(i));
      if (missing) {
        dp->set_missing_status(Data::missing_status::completely_missing);
      }
      add_data(dp);
    }
  }

  SSPM::StateSpacePoissonModel(const SSPM &rhs)
      : StateSpaceNormalMixture(rhs),
        observation_model_(rhs.observation_model_->clone()) {}

  SSPM *SSPM::clone() const { return new SSPM(*this); }

  int SSPM::time_dimension() const { return dat().size(); }

  double SSPM::observation_variance(int t) const {
    if (t >= time_dimension()) {
      // Variance of Poisson latent variable, on the log scale.
      return Constants::pi_squared_over_6;
    }
    return dat()[t]->latent_data_variance();
  }

  double SSPM::adjusted_observation(int t) const {
    if (is_missing_observation(t)) {
      return negative_infinity();
    }
    return dat()[t]->adjusted_observation(observation_model_->coef());
  }

  bool SSPM::is_missing_observation(int t) const {
    return t >= time_dimension() || dat()[t]->missing() != Data::observed;
  }

  void SSPM::observe_data_given_state(int t) {
    if (!is_missing_observation(t)) {
      dat()[t]->set_state_model_offset(observation_matrix(t).dot(state(t)));
      signal_complete_data_change(t);
    }
  }

  Vector SSPM::simulate_forecast(RNG &rng,
                                 const Matrix &forecast_predictors,
                                 const Vector &exposure,
                                 const Vector &final_state) {
    ScalarStateSpaceModelBase::set_state_model_behavior(StateModel::MARGINAL);
    Vector ans(nrow(forecast_predictors));
    Vector state = final_state;
    int t0 = time_dimension();
    // The time stamp of "final state" is t0 - 1.
    int time = -1;
    for (int i = 0; i < ans.size(); ++i) {
      advance_to_timestamp(rng, time, state, i, i);
      double eta = observation_matrix(time + t0).dot(state) +
                   observation_model_->predict(forecast_predictors.row(i));
      double mu = exp(eta);
      ans[i] = rpois_mt(rng, exposure[i] * mu);
    }
    return ans;
  }

  Matrix SSPM::simulate_forecast_components(RNG &rng,
                                            const Matrix &forecast_predictors,
                                            const Vector &exposure,
                                            const Vector &final_state) {
    ScalarStateSpaceModelBase::set_state_model_behavior(StateModel::MARGINAL);
    int forecast_horizon = nrow(forecast_predictors);
    Matrix ans(number_of_state_models() + 2, forecast_horizon, 0.0);
    Vector state = final_state;
    int t0 = time_dimension();
    for (int t = 0; t < forecast_horizon; ++t) {
      state = simulate_next_state(rng, state, t + t0);
      for (int s = 0; s < number_of_state_models(); ++s) {
        ans(s, t) = state_model(s)->observation_matrix(t + t0).dot(
            state_component(state, s));
      }
      ans(number_of_state_models(), t) = observation_model_->predict(
          forecast_predictors.row(t));
      double mu = exp(sum(ans.col(t)));
      ans.col(t).back() = rpois_mt(rng, exposure[t] * mu);
    }
    return ans;
  }

  Vector SSPM::one_step_holdout_prediction_errors(
      RNG &rng, PoissonDataImputer &data_imputer, const Vector &counts,
      const Vector &exposure, const Matrix &predictors,
      const Vector &final_state) {
    if (nrow(predictors) != counts.size() || exposure.size() != counts.size()) {
      report_error(
          "Size mismatch in arguments provided to "
          "one_step_holdout_prediction_errors.");
    }
    Vector ans(counts.size());
    int t0 = dat().size();
    Kalman::ScalarMarginalDistribution marg(this, nullptr, 0);
    marg.set_state_mean(*state_transition_matrix(t0 - 1) * final_state);
    marg.set_state_variance(state_variance_matrix(t0 - 1)->dense());

    // This function differs from one_step_holdout_prediction_errors
    // in StateSpaceRegressionModel because the response is on the
    // Poisson scale, and the state needs a non-linear (exp) transform
    // to get it on the scale of the data.  We handle this by imputing
    // the latent data for each observation, using the latent data to
    // sample an observation on which the one-step holdout will be
    // computed, and then updating the Kalman filter to draw the next
    // time point.
    for (int t = 0; t < ans.size(); ++t) {
      bool missing = false;
      // 1) simulate next state.
      // 2) simulate w_t given state
      // 3) kalman update state given w_t.
      Vector state = rmvn_mt(rng,
                             marg.state_mean(),
                             marg.state_variance());

      double state_contribution = observation_matrix(t + t0).dot(state);
      double regression_contribution =
          observation_model_->predict(predictors.row(t));
      double mu = state_contribution + regression_contribution;
      double prediction = exposure[t] * exp(mu);
      ans[t] = counts[t] - prediction;

      // ans[t] is a random draw of the one step ahead prediction
      // error at time t0+t given observed data to time t0+t-1.  We
      // now proceed with the steps needed to update the Kalman filter
      // so we can compute ans[t+1].

      double internal_neglog_final_event_time = 0;
      double internal_mixture_mean = 0;
      double internal_mixture_precision = 0;
      double neglog_final_interarrival_time = 0;
      double external_mixture_mean = 0;
      double external_mixture_precision = 0;

      data_imputer.impute(rng, counts[t], exposure[t], mu,
                          &internal_neglog_final_event_time,
                          &internal_mixture_mean, &internal_mixture_precision,
                          &neglog_final_interarrival_time,
                          &external_mixture_mean, &external_mixture_precision);

      double total_precision = external_mixture_precision;
      double precision_weighted_sum =
          neglog_final_interarrival_time - external_mixture_mean;
      precision_weighted_sum *= external_mixture_precision;
      if (counts[t] > 0) {
        precision_weighted_sum +=
            (internal_neglog_final_event_time - internal_mixture_mean) *
            internal_mixture_precision;
        total_precision += internal_mixture_precision;
      }
      double latent_observation = precision_weighted_sum / total_precision;
      double latent_variance = 1.0 / total_precision;

      // The latent state was drawn from its predictive distribution
      // given Y[t0 + t -1] and used to impute the latent data for
      // y[t0+t].  That latent data is now used to update the Kalman
      // filter for the next time period.  It is important that we
      // discard the imputed state at this point.
      marg.update(latent_observation - regression_contribution, missing,
                  t + t0, latent_variance / observation_variance(t + t0));
    }
    return ans;
  }

  Matrix SSPM::simulate_holdout_prediction_errors(
      int niter, int cutpoint_number, bool standardize) {
    Matrix ans(niter, time_dimension());
    SubMatrix training_prediction_errors(
        ans, 0, niter - 1, 0, cutpoint_number - 1);
    SubMatrix holdout_prediction_errors(
        ans, 0, niter - 1, cutpoint_number, ncol(ans) - 1);
    std::vector<Ptr<Data>> training_data(dat().begin(), dat().begin() + cutpoint_number);
    std::vector<Ptr<StateSpace::AugmentedPoissonRegressionData>> holdout_data(
        dat().begin() + cutpoint_number, dat().end());
    clear_data();
    for (const auto &data_point : training_data) {
      add_data(data_point);
    }
    Matrix holdout_predictors(holdout_data.size(), xdim());
    Vector holdout_counts(holdout_data.size());
    Vector holdout_exposure(holdout_data.size());
    for (int i = 0; i < holdout_data.size(); ++i) {
      const PoissonRegressionData &poisson_data(*holdout_data[i]);
      holdout_counts[i] = poisson_data.y();
      holdout_exposure[i] = poisson_data.exposure();
      holdout_predictors.row(i) = poisson_data.x();
    }

    PoissonDataImputer imputer;

    for (int i = 0; i < niter; ++i) {
      sample_posterior();
      training_prediction_errors.row(i) = one_step_prediction_errors(
          standardize);
      holdout_prediction_errors.row(i) = one_step_holdout_prediction_errors(
          PriorPolicy::rng(),
          imputer,
          holdout_counts,
          holdout_exposure,
          holdout_predictors,
          state().last_col());
    }
    return ans;
  }

}  // namespace BOOM
