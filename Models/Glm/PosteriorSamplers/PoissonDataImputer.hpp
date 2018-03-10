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

#ifndef BOOM_POISSON_DATA_IMPUTER_HPP_
#define BOOM_POISSON_DATA_IMPUTER_HPP_

#include "Models/Glm/PosteriorSamplers/NormalMixtureApproximation.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  // This class manages latent data imputation for Poisson regression
  // models using the auxiliary mixture sampling algorithm from
  // Fruhwirth-Schnatter, Fruhwirth, Held, and Rue "Improved Auxiliary
  // Mixture Sampling for Hierarchical Models of Non-Gaussian Data".
  // The algorithm has been modified to account for data with
  // different exposure periods.
  //
  // During the first pass through the data, the object can modify the
  // state of a singleton table by adding table elements for
  // previously unseen data values.  This means that the first pass
  // through a data augmentation should be done in a single thread.
  // Subsequent augmentations can be safely multi-threaded.
  class PoissonDataImputer {
   public:
    // Impute the required variances and excess residuals given
    // observed data.  Pointer arguments are used for output, and may
    // not be NULL.
    // Args:
    //   rng:  A random number generator.
    //   response:  The Poisson count to be augmented.
    //   exposure: The length of the exposure window associated with
    //     'response'.
    //   log_lambda: The log of the mean response, per unit of
    //     exposure.  The expected value of response is exposure *
    //     exp(log_lambda).
    //   internal_neglog_final_event_time: The negative logarithm of
    //     the time (in [0, exposure)) of the final event in the
    //     interior of the exposure interval.  If response == 0 then
    //     this is not set.
    //   internal_mu: The mean of the mixture component imputed for
    //     this observation.  If response == 0 this is not set.
    //   internal_weight: The reciprocal variance (information) for
    //     the uncensored observation.  If response == 0 then this is
    //     not set.
    //   neglog_final_interarrival_time: The negative logarithm of the
    //     time between the final event in the exposure interval and
    //     the first event beyond it.
    //   external_mu: The mean of the mixture component imputed for
    //     the final interarrival time.
    //   external_weight: The reciprocal variance (information) for
    //     the mixture component imputed for the final interarrival time.
    void impute(RNG &rng, int response, double exposure, double log_lambda,
                double *internal_neglog_final_event_time, double *internal_mu,
                double *internal_weight, double *neglog_final_interarrival_time,
                double *external_mu, double *external_weight);

    // Fills the mixture_table_ with all values that could possibly be
    // required by unmix_poisson_augmented_data, so that usage of this
    // class is thread-safe afterwards.  Saturating the table might
    // take a long time, however.
    static void saturate_mixture_table();

    // Save the values in the mixture table to a Vector that can be
    // used to restore the table later.
    static Vector serialize_mixture_table() {
      return mixture_table_.serialize();
    }

    // Restore the table to a state that was saved earlier.
    static void deserialize_mixture_table(const Vector &serialized_state) {
      mixture_table_.deserialize(serialized_state);
    }

   private:
    // The NormalMixtureApproximationTable is really big.  It is
    // static so that multiple samplers (e.g. in a hierarchical model)
    // don't all need their own copy.  Note that during the first MCMC
    // iteration the table can get modified, so we need to run in
    // single threaded mode for a single iteration.  After all
    // possible values of y (the "response") have been observed it is
    // safe to run in multi-threaded mode.
    static NormalMixtureApproximationTable mixture_table_;
  };

}  // namespace BOOM
#endif  // BOOM_POISSON_DATA_IMPUTER_HPP_
