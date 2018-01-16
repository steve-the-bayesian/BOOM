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

#include <Models/Glm/PosteriorSamplers/PoissonDataImputer.hpp>
#include <Models/Glm/PosteriorSamplers/poisson_mixture_approximation_table.hpp>
#include <cmath>
#include <distributions.hpp>

namespace BOOM {

  NormalMixtureApproximationTable
  PoissonDataImputer::mixture_table_(
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

    double time_of_final_internal_event = response > 0 ?
        exposure * (rbeta_mt(rng, response, 1)) : 0;
    double final_interarrival_time = exposure - time_of_final_internal_event
        + rexp_mt(rng, exp(log_lambda));
    double z_external = -log(final_interarrival_time);
    double mu = 0;
    double sigsq = 1;
    unmix_poisson_augmented_data(
        rng, z_external - log_lambda, 1, &mu, &sigsq, &mixture_table_);

    *neglog_final_interarrival_time = z_external;
    *external_mu = mu;
    *external_weight = 1.0 / sigsq;

    if (response > 0) {
      double z_internal = -log(time_of_final_internal_event);
      double z_internal_residual = z_internal - log_lambda;
      unmix_poisson_augmented_data(
          rng, z_internal_residual, response, &mu, &sigsq, &mixture_table_);
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
}  // namespace BOOM:
