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

#ifndef BOOM_SAMPLERS_IMPORTANCE_RESAMPLER_HPP_
#define BOOM_SAMPLERS_IMPORTANCE_RESAMPLER_HPP_

#include <functional>
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Samplers/DirectProposal.hpp"

namespace BOOM {

  // An implementation of sampling with importance resampling.
  class ImportanceResampler {
   public:
    // Create an importance resampler that generates draws a target
    // distribution by first generating draws from a proposal
    // distribution, and then resampling with weight proportional to
    // the target density / proposal density.
    ImportanceResampler(
        const std::function<double(const Vector &)> &log_target_density,
        const Ptr<DirectProposal> &proposal);

    // Args:
    //   number_of_draws: The number of draws to be simulated.  Some
    //     draws will be duplicated.
    //   rng: The random number generator that provides the source of
    //     randomness for the simulation.
    //
    // Returns:
    //   A Matrix, with each row corresponding to a unique draw from
    //   the target distribution, and a Vector of weights with length
    //   equal to the number of rows in the Matrix.  The Vector's
    //   entries are the number of times the corresponding row of
    //   the Matrix occurred in the resample.
    std::pair<Matrix, Vector> draw(int number_of_draws,
                                   RNG &rng = GlobalRng::rng);

    // Returns a matrix with number_of_draws rows, each of which is a
    // draw from the target distribution.  The number of draws
    Matrix draw_and_resample(int number_of_draws, RNG &rng = GlobalRng::rng);

   private:
    std::function<double(const Vector &)> log_target_density_;
    Ptr<DirectProposal> proposal_;
  };

}  // namespace BOOM

#endif  //  BOOM_SAMPLERS_IMPORTANCE_RESAMPLER_HPP_
