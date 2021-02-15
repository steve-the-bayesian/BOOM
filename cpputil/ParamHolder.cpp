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
#include "cpputil/ParamHolder.hpp"
#include "Models/ParamTypes.hpp"

namespace BOOM {
  typedef ParamHolder PH;

  PH::ParamHolder(const Ptr<Params> &held, Vector &Wsp)
      : storage_(Wsp), prm_(held) {
    storage_ = prm_->vectorize(true);
  }

  PH::ParamHolder(const Vector &x, const Ptr<Params> &held, Vector &Wsp)
      : storage_(Wsp), prm_(held) {
    storage_ = prm_->vectorize(true);
    prm_->unvectorize(x, true);
  }

  PH::~ParamHolder() { prm_->unvectorize(storage_, true); }

  //------------------------------------------------------------

  typedef ParamVectorHolder PVH;
  PVH::ParamVectorHolder(const std::vector<Ptr<Params>> &held, Vector &Wsp)
      : v(Wsp), prm(held) {}

  PVH::ParamVectorHolder(const Vector &x,
                         const std::vector<Ptr<Params>> &held,
                         Vector &Wsp)
      : v(Wsp), prm(held) {
    v = vectorize(prm, true);
    unvectorize(prm, x, true);
  }

  PVH::~ParamVectorHolder() { unvectorize(prm, v, true); }

}  // namespace BOOM
