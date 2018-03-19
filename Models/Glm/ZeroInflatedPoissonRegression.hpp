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

#ifndef BOOM_GLM_ZERO_INFLATED_POISSON_REGRESSION_MODEL_HPP_
#define BOOM_GLM_ZERO_INFLATED_POISSON_REGRESSION_MODEL_HPP_

#include "Models/Glm/PoissonRegressionData.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/ZeroInflatedPoissonModel.hpp"

namespace BOOM {

  class ZeroInflatedPoissonRegressionData : public PoissonRegressionData {
   public:
    // Args:
    //   event_count: Total number of events represented by this
    //     observation. Must be non-negative.
    //   x: Vector of predictor variables.
    //   number_of_trials: The number of trials for this observation.
    //     Must be positive.
    //   number_of_zeros: The number of trials that produced zero
    //     events.  Must be non-negative, and cannot exceed
    //     number_of_trials.
    ZeroInflatedPoissonRegressionData(int64_t event_count, const Vector &x,
                                      int64_t number_of_trials,
                                      int64_t number_of_zeros);

    // Number of trials that each produced zero events.
    int64_t number_of_zero_trials() const;

    // Number of trials that each produced a positive number of events.
    int64_t number_of_positive_trials() const;

    // Total number of trials for this observation.
    int64_t total_number_of_trials() const;

    // Add more trials, events, and zero-events to this observation.
    void add_incremental_data(int64_t incremental_event_count,
                              int64_t incremental_number_of_trials,
                              int64_t incremental_number_of_zeros);

   private:
    // The number of trials that each produced zero events.
    int64_t number_of_zeros_;

    // The number_of_trials_ is the same as 'exposure' in the
    // underlying PoissonRegressionData.  Stored here as an integer to
    // save the effort of double -> int64_t conversion later.
    int64_t number_of_trials_;
  };

  // y | x ~  [ p(x) * Poisson(lambda(x)) + (1-p(x)) * I(0) ]
  //   where x is a vector of predictor variables,
  //   log lambda(x) = beta.dot(x)
  //   and logit p(x) = delta.dot(x)
  class ZeroInflatedPoissonRegressionModel
      : public ParamPolicy_2<GlmCoefs, GlmCoefs>,
        public IID_DataPolicy<ZeroInflatedPoissonRegressionData>,
        public PriorPolicy {
   public:
    // Create a new model with the specified dimension.  All predictor
    // variables are included, but with zero coefficients.
    // Args:
    //   dimension:  The dimension of the predictor variables.
    explicit ZeroInflatedPoissonRegressionModel(int dimension);
    ZeroInflatedPoissonRegressionModel *clone() const override;

    // Returns the conditional expected value per trial, given the
    // specified vector of predictor variables x.  The conditional
    // expectation is p(x) * lambda(x).
    double expected_value(const Vector &x) const;

    // The first function returns the probability that an observation
    // with the given set of predictors (x) comes from the portion of
    // the distribution that is constrained to be zero.  The second
    // returns the complement of this probability.
    //
    // The logistic regression part of this model directly gives
    // probability_unconstrained().
    double probability_forced_to_zero(const Vector &x) const;
    double probability_unconstrained(const Vector &x) const;

    // Returns exp(poisson_coefficients().dot(x));
    double poisson_mean(const Vector &x) const;

    Ptr<GlmCoefs> poisson_coefficient_ptr();
    const GlmCoefs &poisson_coefficients() const;

    Ptr<GlmCoefs> logit_coefficient_ptr();
    const GlmCoefs &logit_coefficients() const;

    // Simulates a single trial for the given vector of predictor variables.
    double sim(const Vector &x, RNG &rng = BOOM::GlobalRng::rng) const;

    // Simulates the specified number of trials for the given vector of
    // predictor variables and returns a structure containing the aggregated
    // results.
    //
    // Args:
    //   x:  A vector of predictor variables.
    //   n:  The number of trials to simulate.
    // Returns:
    //   Aggregated data for the all the requested observations.
    ZeroInflatedPoissonSuf simulate_sufficient_statistics(
        const Vector &x, int64_t n, RNG &rng = GlobalRng::rng) const;
  };

}  // namespace BOOM
#endif  //  BOOM_GLM_ZERO_INFLATED_POISSON_REGRESSION_MODEL_HPP_
