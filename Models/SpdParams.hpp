#ifndef BOOM_SPD_PARAMS_HPP
#define BOOM_SPD_PARAMS_HPP

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#include "Models/ParamTypes.hpp"
#include "Models/SpdData.hpp"

namespace BOOM {
  /*
   * There are several storage classes that one could use for Spd
   * params.  Store Sigma, Sigma_inverse, chol(Sigma), chol(Sigma_inv),
   * S and R, S and chol(R), S and chol(R_inv), etc.
   */

  class SpdParams : public SpdData, virtual public Params {
   public:
    explicit SpdParams(uint p, double diag = 1.0, bool ivar = false);
    explicit SpdParams(const SpdMatrix &V, bool ivar = false);
    SpdParams *clone() const override;

    uint size(bool minimal = true) const override {
      return SpdData::size(minimal);
    }
    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    using Params::unvectorize;
  };

}  // namespace BOOM
#endif  // BOOM_SPD_PARAMS_HPP
