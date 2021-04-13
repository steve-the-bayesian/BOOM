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

#ifndef BOOM_MARKOV_CONJ_SHRINKAGE_SAMPLER_HPP
#define BOOM_MARKOV_CONJ_SHRINKAGE_SAMPLER_HPP

#include "Models/DirichletModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/ProductDirichletModel.hpp"

namespace BOOM {
  class MarkovConjShrinkageSampler;
  class MarkovModel;

  class MarkovConjShrinkageSampler : public PosteriorSampler {
   public:
    explicit MarkovConjShrinkageSampler(uint dim,
                                        RNG &seeding_rng = GlobalRng::rng);
    explicit MarkovConjShrinkageSampler(const Matrix &Nu,
                                        RNG &seeding_rng = GlobalRng::rng);
    MarkovConjShrinkageSampler(const Matrix &Nu, const Vector &nu,
                               RNG &seeding_rng = GlobalRng::rng);
    explicit MarkovConjShrinkageSampler(const Ptr<ProductDirichletModel> &Nu,
                                        RNG &seeding_rng = GlobalRng::rng);
    MarkovConjShrinkageSampler(const Ptr<ProductDirichletModel> &Nu,
                               const Ptr<DirichletModel> &nu,
                               RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    uint Nmodels() const;
    uint dim() const;
    MarkovConjShrinkageSampler *add_model(MarkovModel *);
    // The sampler manages a collection of MarkovModels.  Its 'draw'
    // method will sample all their initial distributions and transition
    // matrices, and update the sufficient statistics for pri_.  It does
    // not draw the parameters of pri_.

   private:
    std::vector<MarkovModel *> models_;
    Ptr<ProductDirichletModel> pri_;
    Ptr<DirichletModel> ipri_;
    void check_dim(uint d);
  };

}  // namespace BOOM

#endif  // BOOM_MARKOV_CONJ_SHRINKAGE_SAMPLER_HPP
