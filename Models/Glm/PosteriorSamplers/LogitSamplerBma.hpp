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

#ifndef BOOM_LOGIT_SAMPLER_BMA_HPP
#define BOOM_LOGIT_SAMPLER_BMA_HPP

#include "Models/Glm/PosteriorSamplers/LogitSampler.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"

namespace BOOM {

  class LogitSamplerBma : public LogitSampler {
   public:
    LogitSamplerBma(LogisticRegressionModel *mod, const Ptr<MvnBase> &slab,
                    const Ptr<VariableSelectionPrior> &spike,
                    RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    void limit_model_selection(uint n);
    double log_model_prob(const Selector &inc) const;

   private:
    void draw_beta_given_gamma();
    void draw_gamma();

    LogisticRegressionModel *mod_;
    Ptr<MvnBase> slab_;
    Ptr<VariableSelectionPrior> spike_;
    uint max_nflips_;

    mutable SpdMatrix Ominv, iV_tilde_;
  };

}  // namespace BOOM
#endif  // BOOM_LOGIT_SAMPLER_BMA_HPP
