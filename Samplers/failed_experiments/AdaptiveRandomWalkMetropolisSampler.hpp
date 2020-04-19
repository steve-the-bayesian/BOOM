#ifndef BOOM_ADAPTIVE_RANDOM_WALK_METROPOLIS_SAMPLER_HPP_
#define BOOM_ADAPTIVE_RANDOM_WALK_METROPOLIS_SAMPLER_HPP_
/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include <functional>

#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Samplers/Sampler.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  class AdaptiveRandomWalkMetropolisSampler : public Sampler {
   public:
    typedef std::function<double(const Vector &)> LogDensity;
    explicit AdaptiveRandomWalkMetropolisSampler(
        const LogDensity &log_density,
        double smoothing_weight_on_past = .95,
        RNG *rng = nullptr);
    Vector draw(const Vector &old) override;

   private:
    void update_proposal_distribution(const Vector &cand, const Vector &old,
                                      bool accepted);

    LogDensity log_density_;
    double smoothing_weight_;
    SpdMatrix smoothed_sum_of_squares_;
    double smoothed_sample_size_;
  };

}  // namespace BOOM

#endif  //  BOOM_ADAPTIVE_RANDOM_WALK_METROPOLIS_SAMPLER_HPP_
