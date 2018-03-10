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

#include "Models/Glm/ZeroInflatedPoissonRegression.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  ZeroInflatedPoissonRegressionData::ZeroInflatedPoissonRegressionData(
      int64_t y, const Vector &x, int64_t total_number_of_trials,
      int64_t number_of_zero_trials)
      : PoissonRegressionData(y, x, total_number_of_trials),
        number_of_zeros_(number_of_zero_trials),
        number_of_trials_(total_number_of_trials) {
    if (number_of_zero_trials < 0) {
      report_error("Number of trials must be non-negative.");
    }
    if (total_number_of_trials < number_of_zero_trials) {
      report_error("total_number_of_trials must exceed number_of_zero_trials.");
    }
  }

  int64_t ZeroInflatedPoissonRegressionData::number_of_zero_trials() const {
    return number_of_zeros_;
  }

  int64_t ZeroInflatedPoissonRegressionData::number_of_positive_trials() const {
    return number_of_trials_ - number_of_zeros_;
  }

  int64_t ZeroInflatedPoissonRegressionData::total_number_of_trials() const {
    return number_of_trials_;
  }

  void ZeroInflatedPoissonRegressionData::add_incremental_data(
      int64_t incremental_event_count, int64_t incremental_number_of_trials,
      int64_t incremental_number_of_zeros) {
    if (incremental_number_of_zeros > incremental_number_of_trials) {
      report_error(
          "Number of trials producing zero events cannot "
          "exceed the total number of trials.");
    }
    if (incremental_number_of_trials < 0) {
      report_error("The number of trials must be non-negative.");
    }
    if (incremental_event_count < 0) {
      report_error("The number of incremental events must be non-negative");
    }
    if (incremental_number_of_zeros < 0) {
      report_error(
          "The number of incremental zero-trials must "
          "be non-negative");
    }
    if (incremental_number_of_trials == 0 && incremental_event_count > 0) {
      report_error(
          "Postive incremental event count "
          "but zero incremental trials.");
    }
    number_of_zeros_ += incremental_number_of_zeros;
    number_of_trials_ += incremental_number_of_trials;
    PoissonRegressionData::set_exposure(exposure() +
                                        incremental_number_of_trials);
    set_y(y() + incremental_event_count);
  }
  //======================================================================
  typedef ZeroInflatedPoissonRegressionModel ZIPRM;
  ZIPRM::ZeroInflatedPoissonRegressionModel(int dimension)
      : ParamPolicy(new GlmCoefs(dimension), new GlmCoefs(dimension)) {}

  ZIPRM *ZIPRM::clone() const { return new ZIPRM(*this); }

  double ZIPRM::expected_value(const Vector &x) const {
    return probability_unconstrained(x) * poisson_mean(x);
  }

  double ZIPRM::poisson_mean(const Vector &x) const {
    return exp(poisson_coefficients().predict(x));
  }

  double ZIPRM::probability_unconstrained(const Vector &x) const {
    return plogis(logit_coefficients().predict(x));
  }

  double ZIPRM::probability_forced_to_zero(const Vector &x) const {
    return plogis(logit_coefficients().predict(x), 0, 1, false);
  }

  Ptr<GlmCoefs> ZIPRM::poisson_coefficient_ptr() { return prm1(); }

  const GlmCoefs &ZIPRM::poisson_coefficients() const { return prm1_ref(); }

  Ptr<GlmCoefs> ZIPRM::logit_coefficient_ptr() { return prm2(); }

  const GlmCoefs &ZIPRM::logit_coefficients() const { return prm2_ref(); }

  double ZIPRM::sim(const Vector &x, RNG &rng) const {
    if (runif_mt(rng) < probability_forced_to_zero(x)) {
      return 0.0;
    }
    return rpois(poisson_mean(x));
  }

  ZeroInflatedPoissonSuf ZIPRM::simulate_sufficient_statistics(const Vector &x,
                                                               int64_t n,
                                                               RNG &rng) const {
    double number_of_zeros = rbinom_mt(rng, n, probability_forced_to_zero(x));
    double number_of_positives = n - number_of_zeros;
    double sum_of_positives =
        rpois_mt(rng, number_of_positives * poisson_mean(x));
    return ZeroInflatedPoissonSuf(number_of_zeros, number_of_positives,
                                  sum_of_positives);
  }
}  // namespace BOOM
