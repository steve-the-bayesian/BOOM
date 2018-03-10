// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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

#include "Samplers/UnivariateSliceSampler.hpp"
#include <cassert>
#include <cmath>
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef UnivariateSliceSampler USS;
  }  // namespace

  USS::UnivariateSliceSampler(const Target &logpost, double suggested_dx,
                              bool unimodal, RNG *rng)
      : Sampler(rng),
        f_(logpost),
        suggested_dx_(suggested_dx),
        unimodal_(unimodal) {}

  Vector USS::draw(const Vector &x) {
    theta_ = x;
    if (scalar_samplers_.empty()) {
      initialize(x.size());
    }
    for (int i = 0; i < scalar_samplers_.size(); ++i) {
      theta_[i] = scalar_samplers_[i].draw(theta_[i]);
    }
    return theta_;
  }

  void USS::initialize(int dim) {
    for (int i = 0; i < dim; ++i) {
      scalar_targets_.push_back(ScalarTargetFunAdapter(f_, &theta_, i));
      scalar_samplers_.push_back(ScalarSliceSampler(
          scalar_targets_.back(), unimodal_, suggested_dx_, &rng()));
    }
  }

  void USS::set_limits(const Vector &lower, const Vector &upper) {
    if (scalar_samplers_.empty()) {
      initialize(lower.size());
    }
    if (lower.size() != scalar_samplers_.size() ||
        upper.size() != scalar_samplers_.size()) {
      report_error(
          "Limits are wrong dimension in "
          "UnivariateSliceSampler::set_limits.");
    }
    for (size_t i = 0; i < lower.size(); ++i) {
      if (upper[i] <= lower[i]) {
        report_error("Upper limit must be larger than lower limit.");
      }
      if (std::isfinite(lower[i])) {
        scalar_samplers_[i].set_lower_limit(lower[i]);
      }
      if (std::isfinite(upper[i])) {
        scalar_samplers_[i].set_upper_limit(upper[i]);
      }
    }
  }

}  // namespace BOOM
