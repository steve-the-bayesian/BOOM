// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#ifndef BOOM_FIXED_SPD_SAMPLER_HPP_
#define BOOM_FIXED_SPD_SAMPLER_HPP_
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/SpdParams.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {
  class FixedSpdSampler : public PosteriorSampler {
   public:
    FixedSpdSampler(const Ptr<SpdParams> &spd, double value,
                    int which_diagonal_element,
                    RNG &seeding_rng = GlobalRng::rng)
        : PosteriorSampler(seeding_rng),
          spd_(spd),
          value_(value),
          i_(which_diagonal_element),
          j_(which_diagonal_element) {}

    FixedSpdSampler(const Ptr<SpdParams> &spd, double value, int which_i,
                    int which_j, RNG &seeding_rng = GlobalRng::rng)
        : PosteriorSampler(seeding_rng),
          spd_(spd),
          value_(value),
          i_(which_i),
          j_(which_j) {}

    void draw() override {
      if (spd_->var()(i_, j_) == value_) return;
      SpdMatrix Sigma = spd_->var();
      Sigma(i_, j_) = value_;
      if (i_ != j_) Sigma(j_, i_) = value_;
      spd_->set_var(Sigma);
    }

    double logpri() const override {
      if (spd_->var()(i_, j_) == value_) return 0;
      return BOOM::negative_infinity();
    }

   private:
    Ptr<SpdParams> spd_;
    double value_;
    int i_;
    int j_;
  };

}  // namespace BOOM

#endif  // BOOM_FIXED_SPD_SAMPLER_HPP_
