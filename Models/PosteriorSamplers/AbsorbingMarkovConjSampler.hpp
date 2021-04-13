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

#ifndef BOOM_ABSORBING_MARKOV_CONJUGATE_SAMPLER_HPP
#define BOOM_ABSORBING_MARKOV_CONJUGATE_SAMPLER_HPP

#include "LinAlg/Selector.hpp"
#include "Models/MarkovModel.hpp"
#include "Models/PosteriorSamplers/MarkovConjSampler.hpp"

namespace BOOM {

  class AbsorbingMarkovConjSampler : public MarkovConjSampler {
   public:
    AbsorbingMarkovConjSampler(MarkovModel *Mod,
                               const Ptr<ProductDirichletModel> &Q,
                               const Ptr<DirichletModel> &pi0,
                               const std::vector<uint> &absorbing_states,
                               RNG &seeding_rng = GlobalRng::rng);
    AbsorbingMarkovConjSampler(MarkovModel *Mod,
                               const Ptr<ProductDirichletModel> &Q,
                               const std::vector<uint> &absorbing_states,
                               RNG &seeding_rng = GlobalRng::rng);
    AbsorbingMarkovConjSampler(MarkovModel *Mod, const Matrix &Nu,
                               const std::vector<uint> &absorbing_states,
                               RNG &seeding_rng = GlobalRng::rng);
    AbsorbingMarkovConjSampler(MarkovModel *Mod, const Matrix &Nu,
                               const Vector &nu,
                               const std::vector<uint> &absorbing_states,
                               RNG &seeding_rng = GlobalRng::rng);

    AbsorbingMarkovConjSampler *clone_to_new_host(Model *new_host) const override;

    double logpri() const override;
    void draw() override;
    void find_posterior_mode(double epsilon = 1e-5) override;
    bool can_find_posterior_mode() const override { return true; }

   private:
    MarkovModel *mod_;
    Selector abs_;
    Selector trans_;
  };

}  // namespace BOOM

#endif  // BOOM_ABSORBING_MARKOV_CONJUGATE_SAMPLER_HPP
