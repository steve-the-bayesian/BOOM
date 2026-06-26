/*
  Copyright (C) 2005-2026 Steven L. Scott

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

#include "Bandits/LogitBandit.hpp"
#include "Bandits/bandit_functions.hpp"
#include "stats/logit.hpp"
#include "stats/optimal_arm_probabilities.hpp"
#include "distributions.hpp"

namespace BOOM {

  LogitBandit::LogitBandit(const Ptr<BinomialLogitModel> &model,
                           const Ptr<LinearBanditEncoder> &encoder)
      : model_(model),
        encoder_(encoder),
        last_thompson_row_(-1),
        last_thompson_arm_(-1)
  {}
  
  void LogitBandit::observe_data(int arm,
                                 int num_successes,
                                 int num_trials,
                                 const MixedMultivariateData &context) {
    Vector predictor_vector = encoder_->encode_row(arm, context);
    NEW(BinomialRegressionData, data_point)(
        num_successes, num_trials, predictor_vector);
    model_->add_data(data_point);
  }

  void LogitBandit::update_posterior(int ndraws) {
    coefficient_draws_.resize(ndraws, model_->xdim());
    log_likelihood_.resize(ndraws);
    for (int i = 0; i < ndraws; ++i) {
      model_->sample_posterior();
      coefficient_draws_.row(i) = model_->Beta();
      log_likelihood_[i] = model_->log_likelihood();
    }
  }

  double LogitBandit::value(int arm, const MixedMultivariateData &context) const {
    return model_->predict(encoder_->encode_row(arm, context));
  }

  Matrix LogitBandit::arm_predictors(const MixedMultivariateData &context) const {
    Matrix ans(number_of_arms(), model_->xdim());
    for (int i = 0; i < number_of_arms(); ++i) {
      ans.row(i) = encoder_->encode_row(i, context);
    }
    return ans;
  }
  
  Vector LogitBandit::optimal_arm_probabilities(
      const MixedMultivariateData &context,
      RNG &rng) const {
    Matrix predictors = arm_predictors(context);
    Matrix value_draws = coefficient_draws_.multT(predictors);
    return ComputeOptimalArmProbabilities(value_draws, rng);
  }

  std::vector<std::string> LogitBandit::thompson(
      const MixedMultivariateData &context,
      RNG &rng) const {
    Matrix predictors = arm_predictors(context);
    last_thompson_row_ = rmulti_mt(rng, 0, draws().nrow() - 1);
    const ConstVectorView coefficients = coefficient_draws_.row(
        last_thompson_row_);
    Matrix value_draws(1, number_of_arms());
    value_draws.row(0) = predictors * coefficients;
    Vector probs = ComputeOptimalArmProbabilities(value_draws, rng);
    last_thompson_arm_ = argmax_random_ties(probs, rng);
    return encoder()->arm_values(last_thompson_arm_);
  }

  Vector LogitBandit::value_remaining_distribution(
      const MixedMultivariateData &context,
      RNG &rng) const {
    Matrix predictors = arm_predictors(context);
    Matrix value_draws = coefficient_draws_.multT(predictors);
    for (int i = 0; i < value_draws.nrow(); ++i) {
      for (int j = 0; j < value_draws.ncol(); ++j) {
        value_draws(i, j) = logit_inv(value_draws(i, j));
      }
    }
    return ValueRemainingDistribution(value_draws, rng);
  }
  
  void LogitBandit::set_draws(const Matrix &draws) {
    if (draws.ncol() != model_->xdim()) {
      std::ostringstream err;
      err << "A matrix with " << draws.ncol()
          << " columns was pased to LogitBandit::set_draws, but the model "
          << "has " << model_->xdim() << " predictors.";
      report_error(err.str());
    }
    coefficient_draws_ = draws;
    model_->set_Beta(draws.last_row());
  }

  void LogitBandit::set_log_likelihood(const Vector &log_likelihood) {
    if (coefficient_draws_.nrow() > 0
        && coefficient_draws_.nrow() != log_likelihood.size()) {
      std::ostringstream err;
      err << "Each log likelihood value is associated with one MCMC "
          "draw of a coefficient vector, but you have loaded a set of "
          "log likelihood values with length that fails to match the "
          "number of rows in coefficient_draws_.";
      report_error(err.str());
    }
    log_likelihood_ = log_likelihood;
  }
  
}  // namespace BOOM
