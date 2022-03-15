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
#ifndef BOOM_COMPOSITE_MODEL_SAMPLER_HPP_
#define BOOM_COMPOSITE_MODEL_SAMPLER_HPP_

#include "Models/CompositeModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
namespace BOOM {
  // A CompositeModelSampler is the default sampling method for a
  // CompositeModel.  It implements logpri() and draw() by passing calls
  // down to the individual components of the composite model.
  class CompositeModelSampler : public PosteriorSampler {
   public:
    explicit CompositeModelSampler(CompositeModel *model,
                                   RNG &seeding_rng = GlobalRng::rng);
    CompositeModelSampler *clone_to_new_host(Model *new_host) const override;
    double logpri() const override;
    void draw() override;

   private:
    CompositeModel *m_;
  };
}  // namespace BOOM
#endif  //  BOOM_COMPOSITE_MODEL_SAMPLER_HPP_
