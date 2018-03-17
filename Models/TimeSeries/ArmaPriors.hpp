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

  class UniformMaPrior : public VectorModel,
                         public NullParamPolicy,
                         public NullDataPolicy,
                         public PriorPolicy {
   public:
    explicit UniformMaPrior(int dim) : dim_(dim) {}
    UniformMaPrior *clone() const override { return new UniformMaPrior(*this); }
    double logp(const Vector &x) const override;
    Vector sim(RNG &rng = GlobalRng::rng) const override;

   private:
    int dim_;
  };

  class UniformArPrior : public VectorModel,
                         public NullParamPolicy,
                         public NullDataPolicy,
                         public PriorPolicy {
   public:
    explicit UniformArPrior(int dim) : dim_(dim) {}
    UniformArPrior *clone() const override { return new UniformArPrior(*this); }
    double logp(const Vector &x) const override;
    Vector sim(RNG &rng = GlobalRng::rng) const override;

   private:
    int dim_;
  };

}  // namespace BOOM

#endif  //  BOOM_ARMA_PRIORS_HPP_
