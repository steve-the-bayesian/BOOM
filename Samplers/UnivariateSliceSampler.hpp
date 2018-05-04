// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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

#ifndef BOOM_SAMPLERS_UNIVARIATE_SLICE_SAMPLER_HPP_
#define BOOM_SAMPLERS_UNIVARIATE_SLICE_SAMPLER_HPP_

#include <functional>
#include "LinAlg/Vector.hpp"
#include "Samplers/Sampler.hpp"
#include "Samplers/ScalarSliceSampler.hpp"
#include "TargetFun/TargetFun.hpp"

namespace BOOM {

  // A "Univariate" slice sampler draws a vector one component at a
  // time.  If you just want to draw a scalar quantity then you want
  // a ScalarSliceSampler instead.
  class UnivariateSliceSampler : public Sampler {
   public:
    typedef std::function<double(const Vector &x)> Target;
    // Args:
    //   logdensity: The log of the un-normalized density function to
    //     be sampled.
    //   dim:  The dimension of the density to be sampled.
    //   suggested_dx: The initial suggested step size to use for each
    //     scalar slice sampler.
    //   unimodal: If 'true' the density is known to be unimodal.  If
    //     'false' then the density is potentially multi-modal.
    //   rng: A pointer to the random number generator that supplies
    //     randomness to this sampler.
    explicit UnivariateSliceSampler(const Target &logdensity,
                                    double suggested_dx = 1.0,
                                    bool unimodal = false,
                                    RNG *rng = nullptr);
    Vector draw(const Vector &x) override;

    // Set lower and upper limits for the domain of each variable.
    // negative_infinity() and infinity() are legal values, but
    // lower[i] < upper[i] is a requirement for all i.
    void set_limits(const Vector &lower, const Vector &upper);

   private:
    // Vector valued members start out empty until the first call to
    // draw() or set_limits, at which point the dimension of the
    // problem becomes apparent.  At that point, initialize is called
    // to set up the vector of samplers and scalar target adapters
    // with the right dimension.
    void initialize(int dimension);

    Target f_;
    double suggested_dx_;
    bool unimodal_;
    std::vector<ScalarTargetFunAdapter> scalar_targets_;
    std::vector<ScalarSliceSampler> scalar_samplers_;
    Vector theta_;
  };

}  // namespace BOOM
#endif  // BOOM_SAMPLERS_UNIVARIATE_SLICE_SAMPLER_HPP_
