// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#include "Samplers/Sampler.hpp"

namespace BOOM {

  SamplerBase::SamplerBase() : rng_(0), owns_rng_(false) {}

  SamplerBase::SamplerBase(RNG *rng) : rng_(rng), owns_rng_(false) {}

  SamplerBase::~SamplerBase() {
    if (owns_rng_) delete rng_;
  }

  RNG &SamplerBase::rng() const {
    if (rng_) return *rng_;
    return GlobalRng::rng;
  }

  void SamplerBase::set_rng(RNG *r, bool owner) {
    rng_ = r;
    owns_rng_ = owner;
  }

  SamplerBase::SamplerBase(const SamplerBase &rhs)
      : RefCounted(rhs), owns_rng_(false) {
    if (!rhs.owns_rng_) {
      rng_ = rhs.rng_;
    }
  }

  void SamplerBase::set_seed(unsigned long s) {
    if (rng_) {
      rng_->seed(s);
    }  // do nothing if no rng is set.
  }

}  // namespace BOOM
