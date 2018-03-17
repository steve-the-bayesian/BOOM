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

#include "Models/Glm/BinomialLogitModel.hpp"
#include "Models/Glm/GammaRegressionModel.hpp"
#include "Models/Glm/ZeroInflatedGammaRegression.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  // This is a convenience class for the
  // ZeroInflatedGammaRegressionModel.  It assumes that posterior
  // samplers have been set for the logit_model and gamma_regression
  // components of the ZeroInflatedGammaRegressionModel.
  class ZeroInflatedGammaRegressionPosteriorSampler : public PosteriorSampler {
   public:
    explicit ZeroInflatedGammaRegressionPosteriorSampler(
        ZeroInflatedGammaRegressionModel *model,
        RNG &seeding_rng = GlobalRng::rng)
        : PosteriorSampler(seeding_rng), model_(model) {}

    void draw() override {
      model_->logit_model()->sample_posterior();
      model_->gamma_regression()->sample_posterior();
    }

    double logpri() const override {
      return model_->logit_model()->logpri() +
             model_->gamma_regression()->logpri();
    }

   private:
    ZeroInflatedGammaRegressionModel *model_;
  };

}  // namespace BOOM
