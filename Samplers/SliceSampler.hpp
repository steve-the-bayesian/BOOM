// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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
#ifndef BOOM_SLICE_SAMPLER_HPP
#define BOOM_SLICE_SAMPLER_HPP

#include <functional>
#include "LinAlg/Vector.hpp"
#include "Samplers/Sampler.hpp"

namespace BOOM {

  typedef std::function<double(const Vector &)> Func;

  class SliceSampler : public Sampler {
   public:
    explicit SliceSampler(const Func &log_density, bool unimodal = false);
    Vector draw(const Vector &x) override;

   private:
    // lo and hi, last_position_, and random_direction_ define slice boundaries.
    // The "left" edge of the slice is last_position_ - lo_ * random_direction_.
    // The "right " edge is last_position_ + hi * random_direction_;
    //
    // Both hi_ and lo_ are positive numbers.
    double lo_, logplo_;
    double hi_, logphi_;

    // When the class is first constructed, scale_ is set to 1.  After
    // the first call to draw(), scale_ remembers the distance from
    // the argument to the returned value, which helps get things on
    // the right scale the next time draw() is called.  Thus the first
    // draw might be slightly inefficient.
    double scale_;

    // The height of the log density determining the slice.
    double log_p_slice_;

    // The argument to draw() is stored as last_position_, which is
    // needed in several places throughout the call.
    Vector last_position_;

    // When draw() is called a random direction is chosen, and a
    // univariate slice sample is taken along that direction.
    Vector random_direction_;

    bool unimodal_;
    Func logp_;
    void doubling(bool);
    void contract(double lam, double p);
    void find_limits();

    // Point "random_direction_" in a uniformly chosen random direction.
    void set_random_direction();

    void initialize();
  };

}  // namespace BOOM

#endif  // BOOM_SLICE_SAMPLER_HPP
