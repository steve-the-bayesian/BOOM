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

#include "Models/Glm/PosteriorSamplers/poisson_mixture_approximation_table.hpp"

namespace BOOM {
  void fill_poisson_mixture_approximation_table_1(
      NormalMixtureApproximationTable *table);
  void fill_poisson_mixture_approximation_table_2(
      NormalMixtureApproximationTable *table);
  void fill_poisson_mixture_approximation_table_3(
      NormalMixtureApproximationTable *table);

  void fill_poisson_mixture_approximation_table(
      NormalMixtureApproximationTable *table) {
    fill_poisson_mixture_approximation_table_1(table);
    fill_poisson_mixture_approximation_table_2(table);
    fill_poisson_mixture_approximation_table_3(table);
  }

  BOOM::NormalMixtureApproximationTable
  create_poisson_mixture_approximation_table() {
    BOOM::NormalMixtureApproximationTable table;
    fill_poisson_mixture_approximation_table(&table);
    return table;
  }

  void unmix_poisson_augmented_data(
      RNG &rng, double negative_log_interevent_time_residual,
      int number_of_events, double *mu, double *sigsq,
      NormalMixtureApproximationTable *table) {
    if (number_of_events >= table->largest_index()) {
      // If n is very large then the distribution very close to
      // Gaussian.  The mode can be obtained analytically, and the
      // curvature at the mode is just 1.0/n.
      *mu = -log(number_of_events);
      *sigsq = 1.0 / number_of_events;
      return;
    } else {
      NormalMixtureApproximation approximation(
          table->approximate(number_of_events));
      approximation.unmix(rng, negative_log_interevent_time_residual, mu,
                          sigsq);
    }
  }

}  // namespace BOOM
