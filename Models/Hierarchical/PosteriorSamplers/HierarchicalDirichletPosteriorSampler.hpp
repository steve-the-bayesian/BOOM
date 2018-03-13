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

#ifndef BOOM_HIERARCHICAL_DIRICHLET_SAMPLER_HPP_
#define BOOM_HIERARCHICAL_DIRICHLET_SAMPLER_HPP_

#include "Models/DirichletModel.hpp"
#include "Models/PosteriorSamplers/DirichletPosteriorSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/Hierarchical/HierarchicalDirichletModel.hpp"
namespace BOOM {

  class HierarchicalDirichletPosteriorSampler : public PosteriorSampler {
   public:
    HierarchicalDirichletPosteriorSampler(
        HierarchicalDirichletModel *model,
        const Ptr<DiffVectorModel> &mean_prior,
        const Ptr<DiffDoubleModel> &content_prior,
        RNG &seeding_rng = GlobalRng::rng);

    double logpri() const override;
    void draw() override;

   private:
    HierarchicalDirichletModel *model_;
    Ptr<DiffVectorModel> dirichlet_mean_prior_;
    Ptr<DiffDoubleModel> dirichlet_sample_size_prior_;
    Ptr<DirichletPosteriorSampler> sampler_;
  };

}  // namespace BOOM

#endif  // BOOM_HIERARCHICAL_DIRICHLET_SAMPLER_HPP_
