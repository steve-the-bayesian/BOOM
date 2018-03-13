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

#ifndef BOOM_SCALAR_LANGEVIN_SAMPLER_HPP_
#define BOOM_SCALAR_LANGEVIN_SAMPLER_HPP_

#include <functional>
#include "Samplers/Sampler.hpp"
#include "TargetFun/TargetFun.hpp"
#include "cpputil/Ptr.hpp"

namespace BOOM {

  class ScalarLangevinSampler : public ScalarSampler {
   public:
    // The initial innovation sd is set to the square root of the step
    // size.
    ScalarLangevinSampler(const Ptr<dScalarTargetFun> &logf,
                          double initial_step_size, RNG *rng = nullptr);
    double draw(double x) override;

    // The Metropolis proposal is centered on 0.5 * step_size() *
    // gradient.
    double step_size() const { return step_size_; }
    void set_step_size(double new_step_size);

    double logf(double x) const { return (*logf_)(x); }
    double logf(double x, double &g) const { return (*logf_)(x, g); }

    // NOTE: Adaptive stepsize selection is an experimental feature.
    // Its correctness should be mathematically proved.
    //
    // If okay_to_adapt is true then the sampler will attempt to
    // adjust the stepsize, shortening it if too few proposals are
    // accepted, and lengthening it if too many are accepted.
    //
    // The default for this class is no adaptation.
    void allow_adaptation(bool okay_to_adapt);

   private:
    Ptr<dScalarTargetFun> logf_;
    double step_size_;
    double sd_;
    bool adapt_;
    int consecutive_accepts_;
    int consecutive_rejects_;
  };

}  // namespace BOOM

#endif  // BOOM_SCALAR_LANGEVIN_SAMPLER_HPP_
