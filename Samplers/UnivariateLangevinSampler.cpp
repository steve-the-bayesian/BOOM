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
#include "Samplers/UnivariateLangevinSampler.hpp"
#include "TargetFun/TargetFun.hpp"

namespace BOOM {

  UnivariateLangevinSampler::UnivariateLangevinSampler(
      const Ptr<dScalarEnabledTargetFun> &f, int xdim, double step_size,
      RNG *rng)
      : Sampler(rng), f_(f), x_(xdim) {
    for (int i = 0; i < xdim; ++i) {
      scalar_targets_.push_back(new dScalarTargetFunAdapter(f_, &x_, i));
      scalar_samplers_.emplace_back(
          ScalarLangevinSampler(scalar_targets_.back(), step_size, rng));
    }
  }

  Vector UnivariateLangevinSampler::draw(const Vector &x) {
    x_ = x;
    for (int i = 0; i < x_.size(); ++i) {
      x_[i] = scalar_samplers_[i].draw(x_[i]);
    }
    return x_;
  }

  void UnivariateLangevinSampler::allow_adaptation(bool okay_to_adapt) {
    for (int i = 0; i < scalar_samplers_.size(); ++i) {
      scalar_samplers_[i].allow_adaptation(okay_to_adapt);
    }
  }

}  // namespace BOOM
