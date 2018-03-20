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
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {
  typedef PriorPolicy PP;

  void PP::set_method(const Ptr<PosteriorSampler> &sam) {
    samplers_.push_back(sam);
  }

  void PP::sample_posterior() {
    for (uint i = 0; i < samplers_.size(); ++i) {
      samplers_[i]->draw();
    }
  }

  double PP::logpri() const {
    double ans = 0;
    for (uint i = 0; i < samplers_.size(); ++i) ans += samplers_[i]->logpri();
    return ans;
  }

  void PP::clear_methods() { samplers_.clear(); }

  int PP::number_of_sampling_methods() const { return samplers_.size(); }
}  // namespace BOOM
