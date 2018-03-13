// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

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

#ifndef BOOM_BINOMIAL_PROBIT_TIM_SAMPLER_HPP_
#define BOOM_BINOMIAL_PROBIT_TIM_SAMPLER_HPP_

#include "LinAlg/Selector.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/Glm/BinomialProbitModel.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Samplers/TIM.hpp"

namespace BOOM {

  class BinomialProbitTimSampler : public PosteriorSampler {
   public:
    BinomialProbitTimSampler(BinomialProbitModel *model,
                             const Ptr<MvnBase> &prior, double proposal_df = 3,
                             RNG &seeding_rng = GlobalRng::rng);
    double logpri() const override;
    void draw() override;

   private:
    BinomialProbitModel *model_;
    Ptr<MvnBase> prior_;
    double proposal_df_;
    std::map<Selector, TIM> samplers_;
  };

}  // namespace BOOM

#endif  //  BOOM_BINOMIAL_PROBIT_TIM_SAMPLER_HPP_
