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

#ifndef BOOM_SAMPLERS_DIRECT_PROPOSAL_HPP_
#define BOOM_SAMPLERS_DIRECT_PROPOSAL_HPP_

#include "Models/MvnModel.hpp"
#include "Models/MvtModel.hpp"

namespace BOOM {

  // A proposal distribution that can be sampled directly (rather than
  // depending on a previous state, as in MCMC).
  class DirectProposal : private RefCounted {
   public:
    // Returns a draw from the proposal distribution.
    virtual Vector draw(RNG &rng) = 0;

    // Evaluates the log of the proposal density at x.
    virtual double logp(const Vector &x) const = 0;

    friend void intrusive_ptr_add_ref(DirectProposal *);
    friend void intrusive_ptr_release(DirectProposal *);
  };

  void intrusive_ptr_add_ref(DirectProposal *);
  void intrusive_ptr_release(DirectProposal *);

  //----------------------------------------------------------------------
  class MvnDirectProposal : public DirectProposal {
   public:
    MvnDirectProposal(const Vector &mu, const SpdMatrix &Sigma);
    Vector draw(RNG &rng) override;
    double logp(const Vector &x) const override;

   private:
    MvnModel model_;
  };

  //----------------------------------------------------------------------
  class MvtDirectProposal : public DirectProposal {
   public:
    MvtDirectProposal(const Vector &mu, const SpdMatrix &Sigma, double nu);
    Vector draw(RNG &rng) override;
    double logp(const Vector &x) const override;

   private:
    MvtModel model_;
  };

}  // namespace BOOM

#endif  // BOOM_SAMPLERS_DIRECT_PROPOSAL_HPP_
