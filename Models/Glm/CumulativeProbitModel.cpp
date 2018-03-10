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
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
   USA
 */

#include "Models/Glm/CumulativeProbitModel.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef CumulativeProbitModel CPM;

  CPM::CumulativeProbitModel(const Vector &beta, const Vector &delta)
      : OrdinalCutpointModel(beta, delta) {}

  CPM::CumulativeProbitModel(const Matrix &X, const Vector &y)
      : OrdinalCutpointModel(X, y) {}

  CPM::CumulativeProbitModel(const CPM &rhs)
      : Model(rhs), OrdinalCutpointModel(rhs) {}

  CPM *CPM::clone() const { return new CPM(*this); }

  double CPM::link_inv(double eta) const { return pnorm(eta); }

  double CPM::dlink_inv(double eta) const { return dnorm(eta); }

  double CPM::simulate_latent_variable(RNG &rng) const { return rnorm_mt(rng); }
}  // namespace BOOM
