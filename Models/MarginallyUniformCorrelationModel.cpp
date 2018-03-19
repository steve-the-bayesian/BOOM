// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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

#include "Models/MarginallyUniformCorrelationModel.hpp"
#include "distributions.hpp"

namespace BOOM {

  typedef MarginallyUniformCorrelationModel MUCM;

  MUCM::MarginallyUniformCorrelationModel(uint dim) : dim_(dim) {}

  MUCM *MUCM::clone() const { return new MUCM(*this); }

  double MUCM::pdf(const Ptr<Data> &dp, bool logscale) const {
    // un-normalized!!!
    Ptr<SpdParams> d = DAT(dp);
    double ans = logp(CorrelationMatrix(d->value()));
    return logscale ? ans : exp(ans);
  }

  double MUCM::logp(const CorrelationMatrix &R) const {
    // un-normalized
    uint k = R.dim();
    double ldR = R.logdet();
    double nu = k + 1;
    SpdMatrix Rinv = R.inv();
    double ans = -.5 * (nu + k + 1) * ldR - .5 * sum(log(Rinv.diag()));
    return ans;
  }

  uint MUCM::dim() const { return dim_; }

  CorrelationMatrix MUCM::sim(RNG &rng) const {
    uint d = dim();
    SpdMatrix I(d, 1.0);
    SpdMatrix Sigma = rWish_mt(rng, d + 1, I, true);  // inverse wishart
    return var2cor(Sigma);
  }
}  // namespace BOOM
