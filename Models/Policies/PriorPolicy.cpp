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
#include "cpputil/report_error.hpp"

namespace BOOM {

  void PriorPolicy::sample_posterior() {
    for (uint i = 0; i < samplers_.size(); ++i) {
      samplers_[i]->draw();
    }
  }

  double PriorPolicy::logpri() const {
    double ans = 0;
    for (uint i = 0; i < samplers_.size(); ++i) ans += samplers_[i]->logpri();
    return ans;
  }

  void PriorPolicy::set_method(const Ptr<PosteriorSampler> &sampler) {
    samplers_.push_back(sampler);
  }

  void PriorPolicy::clear_methods() { samplers_.clear(); }

  int PriorPolicy::number_of_sampling_methods() const {
    return samplers_.size();
  }

  RNG &PriorPolicy::rng() {
    if (samplers_.empty()) {
      report_error("There are no Samplers from which to obtain a "
                   "random number generator.");
    }
    return samplers_[0]->rng();
  }

}  // namespace BOOM
