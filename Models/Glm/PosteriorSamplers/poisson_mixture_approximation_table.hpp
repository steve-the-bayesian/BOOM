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

#ifndef BOOM_FILL_POISSON_MIXTURE_APPROXIMATION_TABLE_HPP_
#define BOOM_FILL_POISSON_MIXTURE_APPROXIMATION_TABLE_HPP_

#include "Models/Glm/PosteriorSamplers/NormalMixtureApproximation.hpp"

namespace BOOM {
  void fill_poisson_mixture_approximation_table(
      NormalMixtureApproximationTable *table);

  NormalMixtureApproximationTable create_poisson_mixture_approximation_table();

  void unmix_poisson_augmented_data(RNG &rng,
                                    double negative_log_interevent_time,
                                    int number_of_events, double *mu,
                                    double *sigsq,
                                    NormalMixtureApproximationTable *table);

}  // namespace BOOM
#endif  //  BOOM_FILL_POISSON_MIXTURE_APPROXIMATION_TABLE_HPP_
