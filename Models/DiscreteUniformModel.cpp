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

#include "Models/DiscreteUniformModel.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  DiscreteUniformModel::DiscreteUniformModel(int lo, int hi)
      : lo_(lo), hi_(hi) {
    if (hi < lo) {
      report_error("hi must be >= lo in DiscreteUniformModel.");
    }
    log_normalizing_constant_ = log(1 + hi_ - lo_);
  }

  DiscreteUniformModel *DiscreteUniformModel::clone() const {
    return new DiscreteUniformModel(*this);
  }

  double DiscreteUniformModel::logp(int x) const {
    if (x >= lo_ && x <= hi_) {
      return -log_normalizing_constant_;
    }
    return negative_infinity();
  }

  int DiscreteUniformModel::sim(RNG &rng) const {
    return random_int_mt(rng, lo_, hi_);
  }

}  // namespace BOOM
