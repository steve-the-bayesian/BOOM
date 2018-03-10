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

#ifndef BOOM_FIXED_UNIVARIATE_SAMPLER_HPP_
#define BOOM_FIXED_UNIVARIATE_SAMPLER_HPP_
#include "Models/ParamTypes.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {
  class FixedUnivariateSampler : public PosteriorSampler {
   public:
    FixedUnivariateSampler(const Ptr<UnivParams> &prm, double value,
                           RNG &seeding_rng = GlobalRng::rng)
        : PosteriorSampler(seeding_rng), prm_(prm), value_(value) {}

    void draw() override {
      if (prm_->value() == value_) return;
      prm_->set(value_);
    }

    double logpri() const override {
      if (prm_->value() == value_) return 0;
      return BOOM::negative_infinity();
    }

   private:
    Ptr<UnivParams> prm_;
    double value_;
  };

}  // namespace BOOM

#endif  // BOOM_FIXED_UNIVARIATE_SAMPLER_HPP_
