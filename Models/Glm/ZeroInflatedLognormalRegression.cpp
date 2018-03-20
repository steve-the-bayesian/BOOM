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
#include "Models/Glm/ZeroInflatedLognormalRegression.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef ZeroInflatedLognormalRegressionModel ZILRM;
  }  // namespace

  ZILRM::ZeroInflatedLognormalRegressionModel(int dimension,
                                              double zero_threshold)
      : ParamPolicy(new GlmCoefs(dimension), new UnivParams(1.0),
                    new GlmCoefs(dimension)),
        DataPolicy(new NeRegSuf(dimension)),
        zero_threshold_(zero_threshold) {}

  ZILRM *ZILRM::clone() const { return new ZILRM(*this); }

  double ZILRM::expected_value(const Vector &x) const {
    double mu = regression_coefficients().predict(x);
    return probability_nonzero(x) * exp(mu + 0.5 * sigsq());
  }

  // The variance of the mixture is calcualted by EVVE's rule.
  //
  // The variance of the lognormal part is [exp(sigsq) - 1] * exp(2*mu
  // + sigsq).
  //
  // Var(y) = E(Var(y | z)) + Var(E(y|z)), where z is an indicator of
  // the event y==0.  So
  //
  // Var(y) = (1-p(x)) * lognormal_variance
  //           + Var(lognormal_mean * Bernoulli(1 - p))
  //        = (1-p(x)) * lognormal_variance
  //           + p(x)*(1-p(x))*lognormal_mean^2
  double ZILRM::variance(const Vector &x) const {
    double mu = regression_coefficients().predict(x);
    double lognormal_variance = expm1(sigsq()) * exp(2 * mu + sigsq());
    double lognormal_mean = exp(mu + 0.5 * sigsq());
    double p = probability_zero(x);
    return (1 - p) * lognormal_variance + p * (1 - p) * square(lognormal_mean);
  }

  double ZILRM::standard_deviation(const Vector &x) const {
    return sqrt(variance(x));
  }

  double ZILRM::probability_nonzero(const Vector &x) const {
    double eta = logit_coefficients().predict(x);
    return plogis(eta);
  }

  double ZILRM::probability_zero(const Vector &x) const {
    return 1.0 - probability_nonzero(x);
  }

  void ZILRM::add_data(const Ptr<Data> &dp) { add_data(DAT(dp)); }

  void ZILRM::add_data(const Ptr<RegressionData> &dp) {
    if (dp->y() > zero_threshold_) {
      suf()->add_mixture_data(log(dp->y()), dp->x(), 1.0);
    }
    IID_DataPolicy<RegressionData>::add_data(dp);
  }

  Ptr<GlmCoefs> ZILRM::regression_coefficient_ptr() { return prm1(); }

  const GlmCoefs &ZILRM::regression_coefficients() const { return prm1_ref(); }

  Ptr<UnivParams> ZILRM::sigsq_prm() { return prm2(); }

  double ZILRM::sigsq() const { return prm2_ref().value(); }

  double ZILRM::sigma() const { return sqrt(sigsq()); }

  void ZILRM::set_sigsq(double sigsq) {
    if (sigsq <= 0) {
      report_error("sigsq must be positive.");
    }
    prm2_ref().set(sigsq);
  }

  Ptr<GlmCoefs> ZILRM::logit_coefficient_ptr() { return prm3(); }

  const GlmCoefs &ZILRM::logit_coefficients() const { return prm3_ref(); }

  double ZILRM::log_likelihood(const Vector &logit_coefficients,
                               const Vector &regression_coefficients,
                               double sigsq) const {
    double loglike =
        RegressionModel::log_likelihood(regression_coefficients, sigsq, *suf());
    const std::vector<Ptr<RegressionData>> &data(dat());
    for (int i = 0; i < data.size(); ++i) {
      bool success = data[i]->y() > zero_threshold_;
      double log_odds = logit_coefficients.dot(data[i]->x());
      // log [(p/q)^y * q] = y * eta + log(q)
      loglike += success * log_odds +
                 plogis(log_odds,
                        0,      // location
                        1,      // scale
                        false,  // lower tail, because we want log q not log p.
                        true);  // logscale
    }
    return loglike;
  }

  double ZILRM::sim(const Vector &x, RNG &rng) const {
    double p = probability_zero(x);
    double u = runif_mt(rng);
    if (u < p) return 0;
    double logy = rnorm_mt(rng, regression_coefficients().predict(x), sigma());
    return exp(logy);
  }

  HierarchicalZeroInflatedGammaData ZILRM::simulate_sufficient_statistics(
      const Vector &x, int64_t n, RNG &rng) const {
    double p = probability_zero(x);
    double y_hat = regression_coefficients().predict(x);
    int number_of_positives = rbinom(n, p);
    int number_of_zeros = n - number_of_positives;
    double sum = 0.0;
    double sum_of_logs_of_positives = 0.0;
    for (int i = 0; i < number_of_positives; ++i) {
      double log_of_value = rnorm_mt(rng, y_hat, sigma());
      sum += exp(log_of_value);
      sum_of_logs_of_positives += log_of_value;
    }
    return HierarchicalZeroInflatedGammaData(
        number_of_zeros, number_of_positives, sum, sum_of_logs_of_positives);
  }

}  // namespace BOOM
