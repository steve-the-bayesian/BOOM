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

#include "Models/Glm/CumulativeLogitModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef CumulativeLogitModel CLM;
  typedef OrdinalCutpointModel OCM;

  CLM::CumulativeLogitModel(const Vector &beta, const Vector &delta)
      : OCM(beta, delta) {}

  CLM::CumulativeLogitModel(const Matrix &X, const Vector &y) : OCM(X, y) {}

  CLM::CumulativeLogitModel(const CLM &rhs) : Model(rhs), OCM(rhs) {}

  CLM *CLM::clone() const { return new CLM(*this); }

  double CLM::link_inv(double eta) const { return plogis(eta); }

  double CLM::dlink_inv(double eta) const { return dlogis(eta); }

  double CLM::simulate_latent_variable(RNG &rng) const {
    return rlogis_mt(rng);
  }

}  // namespace BOOM
