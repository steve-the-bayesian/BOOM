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

#ifndef BOOM_UNIVARIATE_LANGEVIN_SAMPLER_HPP_
#define BOOM_UNIVARIATE_LANGEVIN_SAMPLER_HPP_

#include "LinAlg/Vector.hpp"
#include "Samplers/Sampler.hpp"
#include "Samplers/ScalarLangevinSampler.hpp"

namespace BOOM {

  class UnivariateLangevinSampler : public Sampler {
   public:
    UnivariateLangevinSampler(const Ptr<dScalarEnabledTargetFun> &f, int xdim,
                              double step_size, RNG *rng);
    Vector draw(const Vector &x) override;

    // If 'okay_to_adapt' is true then the sampler will attempt to
    // adjust the stepsize, shortening it if too few proposals are
    // accepted, and lengthening it if too many are accepted.
    //
    // The default is that 'okay_to_adapt' is false.
    void allow_adaptation(bool okay_to_adapt);

   private:
    Ptr<dScalarEnabledTargetFun> f_;
    Vector x_;
    std::vector<Ptr<dScalarTargetFunAdapter> > scalar_targets_;
    std::vector<ScalarLangevinSampler> scalar_samplers_;
  };

}  // namespace BOOM

#endif  //  BOOM_UNIVARIATE_LANGEVIN_SAMPLER_HPP_
