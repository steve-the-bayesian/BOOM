// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#ifndef BOOM_BINOMIAL_LOGIT_DATA_IMPUTER_HPP_
#define BOOM_BINOMIAL_LOGIT_DATA_IMPUTER_HPP_

#include "Models/Glm/PosteriorSamplers/NormalMixtureApproximation.hpp"

namespace BOOM {

  // For imputing latent data from the truncated logistic
  // distribution.
  class BinomialLogitDataImputer {
   public:
    virtual ~BinomialLogitDataImputer() {}

    // Args:
    //   rng:  The random number generator.
    //   number_of_trials: The number of trials in this binomial
    //     observation.
    //   number_of_successes: The number of successes in this binomial
    //     observation.
    //   log_odds: The log odds of a success (e.g. beta * x in a
    //     logistic regression).
    //
    // Returns:
    //   The first element in the returned pair is the information
    //   weighted sum of the latent logits.  The second element is the
    //   sum of the information weights.
    virtual std::pair<double, double> impute(RNG &rng, double number_of_trials,
                                             double number_of_successes,
                                             double log_odds) const = 0;

    // A finite mixture approximation to the logistic distribution.
    static const LogitMixtureApproximation mixture_approximation;

    // Rather than impute the exact latent data for each trial,
    // approximate methods can be used to draw the sum of the latent
    // data based on a normal approximation.  The minimal number of
    // trials at which the normal approximation should be used is the
    // clt_threshold ("clt = central limit theorem").
    virtual int clt_threshold() const = 0;

   protected:
    // Adds a human readable message to 'err'.
    void debug_status_message(std::ostream &err, double number_of_trials,
                              double number_of_successes, double eta) const;
  };

  //=======================================================================
  // An approximate data imputer.  If number_of_trials is less than
  // the CLT threshold, then the imputation will be done exactly.
  // Otherwise, the sum of the latent utilities will be imputed.
  class BinomialLogitPartialAugmentationDataImputer
      : public BinomialLogitDataImputer {
   public:
    // Args:
    //   clt_threshold: The minimal sample size where approximate
    //     augmentation begins to take place.
    explicit BinomialLogitPartialAugmentationDataImputer(
        int clt_threshold = 10);

    // Impute the latent quasi-sufficient statistics for a single
    // observation.
    //
    // Args:
    //   rng:  The random number generator.
    //   number_of_trials: The number of binomial trials for this
    //     observation.  Must be non-negative.
    //   number_of_successes: The number of binomial successes for
    //     this observation.  Must be non-negative and must not exceed
    //     number_of_trials.
    //   log_odds: The log of the odds of a success in each binomial
    //     trial.
    //
    // Returns:
    //   A pair with elements 'information_weighted_sum', and
    //   'information', constructed so that the sufficient statistics
    //   can be computed as xty += x * information_weighted_sum, and
    //   xtx += information * outer_product(x).
    //
    //   information_weighted_sum: If number_of_trials is
    //     clt_threshold or larger then this is
    //
    //                sum_j (z[i, j] / pi^2/3).
    //
    //     If number_of_trials is less than the clt_threshold then
    //     'information_weighted_sum' is a weighted sum of the
    //     simulated logit values:
    //
    //                sum_j z[i, j] / v[i, j]
    //
    //     where v[i, j] is the variance from the mixture
    //     approximation to the logistic distribution.  The mixture
    //     indicator is imputed, and the variance is conditional on
    //     the imputed mixture indicator.
    //
    //   information: If number_of_trials exceeds the clt_threshold
    //     then this is the reciprocal variance of the sum of
    //     number_of_trials logits: (number_of_trials / (pi^2 / 3)).
    //     Otherwise, it is the sum of the reciprocal variances in the
    //     normal mixture approximation to the logistic distribution,
    //     i.e.
    //
    //               sum_j (1.0 / v[i, j]).
    std::pair<double, double> impute(RNG &rng, double number_of_trials,
                                     double number_of_successes,
                                     double log_odds) const override;

    // The smallest number_of_trials where approximate augmentation
    // takes place.
    int clt_threshold() const override;

   private:
    int clt_threshold_;
  };

  //=======================================================================
  // A nearly exact data imputer.  If number_of_trials is large, then
  // the number of observations from each mixture component is sampled
  // directly from the multinomial distribution.  Then the information
  // weighted sum is sampled conditionally from the large sample
  // approximation given by the central limit theorem.
  class BinomialLogitCltDataImputer : public BinomialLogitDataImputer {
   public:
    // Args:
    //   clt_threshold: The smallest number_of_trials where
    //   approximate augmentation takes place.
    explicit BinomialLogitCltDataImputer(int clt_threshold = 10);

    // Args:
    //   rng:  The random number generator.
    //   number_of_trials: The number of binomial trials for this
    //     observation.  Must be non-negative.
    //   number_of_successes: The number of binomial successes for
    //     this observation.  Must be non-negative and must not exceed
    //     number_of_trials.
    //   log_odds: The log of the odds of a success in each binomial
    //     trial.
    //
    // Returns:
    //   Like the other classes documented above, the first element in
    //   the return pair is information_weighted_sum.  The second is
    //   information.  If number_of_trials < clt_threshold then each
    //   Bernoulli observation is imputed individually.  Otherwise the
    //   number of observations in each mixture component are sampled
    //   by a multinomial draw (using exact probabilities).  So in
    //   either case the 'information' number will be exact.
    //
    //   Likewise, in both cases 'information_weighted_sum' is a weighted
    //   sum of the simulated logit values:
    //
    //                sum_j z[i, j] / v[i, j]
    //
    //   where v[i, j] is the variance from the mixture approximation
    //   to the logistic distribution.  If number_of_trials >=
    //   clt_threshold then the sum is obtained by a single random
    //   draw from the normal distribution with the appropriate
    //   moments.
    std::pair<double, double> impute(RNG &rng, double number_of_trials,
                                     double number_of_successes,
                                     double log_odds) const override;

    // The smallest number_of_trials for which approximate
    // augmentation takes place.
    int clt_threshold() const override;

   private:
    int clt_threshold_;

    // Specific cases used to implement the public impute() method.
    std::pair<double, double> impute_small_sample(RNG &rng,
                                                  double number_of_trials,
                                                  double number_of_successes,
                                                  double eta) const;

    std::pair<double, double> impute_large_sample(RNG &rng,
                                                  double number_of_trials,
                                                  double number_of_successes,
                                                  double eta) const;
  };

}  // namespace BOOM

#endif  // BOOM_BINOMIAL_LOGIT_DATA_IMPUTER_HPP_
