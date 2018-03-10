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

#include "Samplers/DirectProposal.hpp"

namespace BOOM {

  void intrusive_ptr_add_ref(DirectProposal *d) { d->up_count(); }

  void intrusive_ptr_release(DirectProposal *d) {
    d->down_count();
    if (d->ref_count() == 0) {
      delete d;
    }
  }

  MvnDirectProposal::MvnDirectProposal(const Vector &mu, const SpdMatrix &Sigma)
      : model_(mu, Sigma) {}

  Vector MvnDirectProposal::draw(RNG &rng) { return model_.sim(rng); }

  double MvnDirectProposal::logp(const Vector &x) const {
    return model_.logp(x);
  }

  //======================================================================

  MvtDirectProposal::MvtDirectProposal(const Vector &mu, const SpdMatrix &Sigma,
                                       double nu)
      : model_(mu, Sigma, nu) {}

  Vector MvtDirectProposal::draw(RNG &rng) { return model_.sim(rng); }

  double MvtDirectProposal::logp(const Vector &x) const {
    return model_.logp(x);
  }

}  // namespace BOOM
