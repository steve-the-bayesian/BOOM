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

#ifndef BOOM_DISCRETE_UNIFORM_MODEL_HPP_
#define BOOM_DISCRETE_UNIFORM_MODEL_HPP_

#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/NullParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  // A uniform distribution on the integers {lo, ..., hi}.
  class DiscreteUniformModel : public NullParamPolicy,
                               public NullDataPolicy,
                               public PriorPolicy,
                               public IntModel {
   public:
    DiscreteUniformModel(int lo, int hi);
    DiscreteUniformModel *clone() const override;
    double logp(int x) const override;

    // Smallest number in support.
    int lo() const { return lo_; }

    // Largest number in support.
    int hi() const { return hi_; }

    // Simulate a random number between lo() and hi();
    int sim(RNG &rng = GlobalRng::rng) const;

   private:
    int lo_, hi_;
    double log_normalizing_constant_;
  };
}  // namespace BOOM

#endif  //  BOOM_DISCRETE_UNIFORM_MODEL_HPP_
