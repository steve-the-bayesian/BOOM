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

#ifndef BOOM_BINOMIAL_PROBIT_DATA_IMPUTER_HPP_
#define BOOM_BINOMIAL_PROBIT_DATA_IMPUTER_HPP_

#include <ostream>
#include "distributions/rng.hpp"

namespace BOOM {

  //=======================================================================
  // A nearly exact data imputer.  If number_of_trials is large, then
  // the mean and variance of the two relevant truncated normals is
  // used to sample from the normal approximation to sum(z) obtained
  // through the central limit theorem.
  class BinomialProbitDataImputer {
   public:
    // Args:
    //   clt_threshold: The smallest number_of_trials where
    //   approximate augmentation takes place.
    explicit BinomialProbitDataImputer(int clt_threshold = 10);

    // Args:
    //   rng:  The random number generator.
    //   number_of_trials: The number of binomial trials for this
    //     observation.  Must be non-negative.
    //   number_of_successes: The number of binomial successes for
    //     this observation.  Must be non-negative and must not exceed
    //     number_of_trials.
    //   linear_predictor: The linear predictor (probit of success) in each
    //     binomial trial.
    //
    // Returns:
    //   The sum of the latent Gaussian responses associated with this
    //   observation.  y[i] will be from the distribution truncated to
    //   the upper tail, and n[i] - y[i] will be from the distribution
    //   truncated to the lower tail.
    double impute(RNG &rng, double number_of_trials, double number_of_successes,
                  double linear_predictor) const;

    // The smallest number_of_trials for which approximate
    // augmentation takes place.
    int clt_threshold() const { return clt_threshold_; }

   private:
    int clt_threshold_;
  };

}  // namespace BOOM

#endif  // BOOM_BINOMIAL_PROBIT_DATA_IMPUTER_HPP_
