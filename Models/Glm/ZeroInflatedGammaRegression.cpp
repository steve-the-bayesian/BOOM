// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

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

#include "Models/Glm/ZeroInflatedGammaRegression.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    typedef ZeroInflatedGammaRegressionModel ZIGRM;
  }  // namespace

  ZIGRM::ZeroInflatedGammaRegressionModel(int xdim, double zero_threshold)
      : gamma_model_(new GammaRegressionModelConditionalSuf(xdim)),
        logit_model_(new BinomialLogitModel(xdim)),
        zero_threshold_(zero_threshold) {
    ParamPolicy::add_model(gamma_model_);
    ParamPolicy::add_model(logit_model_);
  }

  ZIGRM::ZeroInflatedGammaRegressionModel(const ZIGRM &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        PriorPolicy(rhs),
        gamma_model_(rhs.gamma_model_->clone()),
        logit_model_(rhs.logit_model_->clone()),
        zero_threshold_(rhs.zero_threshold_) {
    ParamPolicy::add_model(gamma_model_);
    ParamPolicy::add_model(logit_model_);
  }

  ZIGRM *ZIGRM::clone() const { return new ZIGRM(*this); }

  double ZIGRM::expected_value(const Vector &x) const {
    double prob_positive = logit_model_->success_probability(x);
    double conditional_mean = gamma_model_->expected_value(x);
    return prob_positive * conditional_mean;
  }

  // At this value of x, let mu, alpha, and p, denote the mean and
  // shape parameters of the gamma distribution, and the probability
  // of a nonzero value, respectively.
  //
  // Let Z = 1 with probability p, and let y > 0 iff Z==1.
  //
  // Var(y) = E(Var(Y | Z)) + Var(E(Y | Z))
  //     Var(Y | Z) = Z * mu^2/alpha, and E(Y | Z) = Z * mu, so
  // Var(y) = (p * mu^2 / alpha) + mu^2 * p * (1 - p)
  double ZIGRM::variance(const Vector &x) const {
    double mu = gamma_model_->expected_value(x);
    double alpha = gamma_model_->shape_parameter();
    double p = logit_model_->success_probability(x);
    double mu2 = mu * mu;
    return (p * mu2 / alpha) + p * mu2 * (1 - p);
  }

  double ZIGRM::standard_deviation(const Vector &x) const {
    return sqrt(variance(x));
  }

  double ZIGRM::probability_nonzero(const Vector &x) const {
    return logit_model_->success_probability(x);
  }

  double ZIGRM::probability_zero(const Vector &x) const {
    return logit_model_->failure_probability(x);
  }

  void ZIGRM::add_data(const Ptr<RegressionData> &dp) {
    double y = dp->y();
    Ptr<VectorData> xptr = dp->Xptr();
    NEW(BinomialRegressionData, logit_data)(y > zero_threshold_, 1, xptr);
    logit_model_->add_data(logit_data);
    if (y > zero_threshold_) {
      gamma_model_->add_data(dp);
    }
  }

  void ZIGRM::add_data(const Ptr<Data> &dp) {
    add_data(dp.dcast<RegressionData>());
  }

  void ZIGRM::increment_sufficient_statistics(
      int number_zeros, int number_nonzero, double sum_of_nonzero,
      double sum_of_logs_of_nonzero, const Ptr<VectorData> &predictors) {
    NEW(BinomialRegressionData, logit_data)
    (number_nonzero, number_nonzero + number_zeros, predictors);
    logit_model_->add_data(logit_data);
    gamma_model_->increment_sufficient_statistics(
        number_nonzero, sum_of_nonzero, sum_of_logs_of_nonzero, predictors);
  }

  void ZIGRM::clear_data() {
    gamma_model_->clear_data();
    logit_model_->clear_data();
  }

  void ZIGRM::combine_data(const Model &rhs, bool just_suf) {
    const ZIGRM &other_model(dynamic_cast<const ZIGRM &>(rhs));
    gamma_model_->combine_data(*other_model.gamma_model_, just_suf);
    logit_model_->combine_data(*other_model.logit_model_, just_suf);
  }

  void ZIGRM::mle() {
    gamma_model_->mle();
    logit_model_->mle();
  }

  double ZIGRM::sim(const Vector &x, RNG &rng) const {
    double p = probability_nonzero(x);
    double u = runif_mt(rng, 0, 1);
    return (u > p ? 0 : gamma_model_->sim(x, rng));
  }

}  // namespace BOOM
