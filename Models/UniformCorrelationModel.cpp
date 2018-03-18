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

#include "Models/UniformCorrelationModel.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef UniformCorrelationModel UCM;

  UCM::UniformCorrelationModel(uint dim) : dim_(dim) {}

  UCM::UniformCorrelationModel(const UCM &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        CorrelationModel(rhs),
        dim_(rhs.dim_) {}

  UCM *UCM::clone() const { return new UCM(*this); }
  void UCM::initialize_params() {}

  double UCM::logp(const CorrelationMatrix &m) const {
    return m.is_pos_def() ? 0.0 : BOOM::negative_infinity();
  }

  double UCM::pdf(const Ptr<Data> &dp, bool logscale) const {
    double ans = logp(CorrelationMatrix(DAT(dp)->value()));
    return logscale ? ans : exp(ans);
  }

  uint UCM::dim() const { return dim_; }

  CorrelationMatrix UCM::sim(RNG &rng) const {
    return random_cor_mt(rng, dim());
  }
}  // namespace BOOM
