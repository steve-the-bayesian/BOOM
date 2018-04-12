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

#include "Models/Glm/PosteriorSamplers/PoissonDataImputer.hpp"
#include <cmath>
#include "Models/Glm/PosteriorSamplers/poisson_mixture_approximation_table.hpp"
#include "distributions.hpp"
#include "cpputil/lse.hpp"

namespace BOOM {

  // Returns true if x can be exponentiated without numerical problems.
  inline bool safe_to_exp(double x) {
    return fabs(x) < 600;
  }
  
  NormalMixtureApproximationTable PoissonDataImputer::mixture_table_(
      create_poisson_mixture_approximation_table());

  void PoissonDataImputer::impute(RNG &rng,
                                  int response,
                                  double exposure,
                                  double log_lambda,
                                  double *internal_neglog_final_event_time,
                                  double *internal_mu,
                                  double *internal_weight,
                                  double *neglog_final_interarrival_time,
                                  double *external_mu,
                                  double *external_weight) {
    double time_of_final_internal_event =
          response > 0 ? exposure * (rbeta_mt(rng, response, 1)) : 0;

    // Delta is the time between the final interior event time and the end of
    // the interval.
    double delta = exposure - time_of_final_internal_event;
    double z_external;
    if (safe_to_exp(log_lambda)) {
      z_external = -log(delta + rexp_mt(rng, exp(log_lambda)));
    } else {
      if (delta > 0) {
        // Handle cases where log lambda is really big or small.  Really big
        // cases are fine.  We just need to keep the algorithm from crashing.
        // Fast mixing will take you back to mortal sized values soon.  When
        // log-lambda is small the algorithm can be drawn in to a slow-mixing
        // cycle because very small log-lambda values lead to very large latent
        // data.
        //
        // Here's the logic: Let D be the distance between the final event time
        // in the interval, and the end of the interval.  You're going to add an
        // E(lambda) = E(1) / lambda to D to get to the final event, then take
        // the negative log of D + E/lambda to get z.  The negative log of an
        // E(1) is standard extreme value, so the negative of a standard extreme
        // value is the log of an E(1).
        //
        // z = -log(D + E/lambda)
        //   = -lse2( log(D), log(E) - log_lambda)
        //   = -lse2( log(D),  -StandardExv - log(lambda))
        double error = rexv_mt(rng, 0, 1);
        z_external = -lse2(log(delta), -error - log_lambda);
      } else {
        z_external = log_lambda + rexv_mt(rng, 0, 1);
      }
    }
    double mu = 0;
    double sigsq = 1;
    unmix_poisson_augmented_data(rng, z_external - log_lambda, 1, &mu, &sigsq,
                                 &mixture_table_);

    *neglog_final_interarrival_time = z_external;
    *external_mu = mu;
    *external_weight = 1.0 / sigsq;

    if (response > 0) {
      double z_internal = -log(time_of_final_internal_event);
      double z_internal_residual = z_internal - log_lambda;
      unmix_poisson_augmented_data(rng, z_internal_residual, response, &mu,
                                   &sigsq, &mixture_table_);
      *internal_neglog_final_event_time = z_internal;
      *internal_mu = mu;
      *internal_weight = 1.0 / sigsq;
    }
  }

  void PoissonDataImputer::saturate_mixture_table() {
    // Force the table to fill itself with interpolated values, so that it does
    // not change later, while being accessed from separate threads.
    for (int n = mixture_table_.smallest_index();
         n < mixture_table_.largest_index(); ++n) {
      mixture_table_.approximate(n);
    }
  }
}  // namespace BOOM
