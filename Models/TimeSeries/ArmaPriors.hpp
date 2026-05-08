#ifndef BOOM_ARMA_PRIORS_HPP_
#define BOOM_ARMA_PRIORS_HPP_

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

#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/NullParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/VectorModel.hpp"

namespace BOOM {

  // A UniformMaPrior is a prior distribution that is uniform over the set of
  // "causal" MA coefficients.
  class UniformMaPrior : public VectorModel,
                         public NullParamPolicy,
                         public NullDataPolicy,
                         public PriorPolicy {
   public:
    explicit UniformMaPrior(int dim) : dim_(dim) {}
    UniformMaPrior *clone() const override { return new UniformMaPrior(*this); }

    // Return 0 if x corresponds to a set of causal MA coefficients, and return
    // negative infinity otherwise.
    double logp(const Vector &x) const override;

    // Simulate MA coefficients uniformly from the [-1, 1] box until a causal
    // set is obtained.  This can fail (resulting in an exception) if the
    // maximum number of simulation attempts is exceeded.
    Vector sim(RNG &rng = GlobalRng::rng) const override;

   private:
    int dim_;
  };

  // A UniformArPrior is a prior distribution over the set of invertible AR
  // coefficients.  
  class UniformArPrior : public VectorModel,
                         public NullParamPolicy,
                         public NullDataPolicy,
                         public PriorPolicy {
   public:
    explicit UniformArPrior(int dim) : dim_(dim) {}
    UniformArPrior *clone() const override { return new UniformArPrior(*this); }

    // Return 0 if x corresponds to a set of invertible AR coefficients,
    // negative infinity otherwise.
    double logp(const Vector &x) const override;

    // Simulate AR coefficients by rejection sampling uniformly from the [-1, 1]
    // box until an invertible set is obtained.  This can fail (resulting in an
    // exception) if the maximum number of simulation attempts is exceeded.
    Vector sim(RNG &rng = GlobalRng::rng) const override;

   private:
    int dim_;
  };

}  // namespace BOOM

#endif  //  BOOM_ARMA_PRIORS_HPP_
