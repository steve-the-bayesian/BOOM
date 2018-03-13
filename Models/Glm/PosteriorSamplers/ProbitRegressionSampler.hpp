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

#ifndef BOOM_PROBIT_REGRESSION_SAMPLER_HPP_
#define BOOM_PROBIT_REGRESSION_SAMPLER_HPP_

#include "Models/Glm/PosteriorSamplers/BinomialProbitDataImputer.hpp"
#include "Models/Glm/ProbitRegression.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {
  class ProbitRegressionSampler : public PosteriorSampler {
   public:
    ProbitRegressionSampler(ProbitRegressionModel *model,
                            const Ptr<MvnBase> &prior,
                            RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double logpri() const override;

    // Call refresh_xtx when the model has gained or lost data.
    // Otherwise, it is assumed that xtx_ is fixed between iterations
    void refresh_xtx();

    void impute_latent_data();
    const Vector &xtz() const;
    const SpdMatrix &xtx() const;

   protected:
    virtual void draw_beta();

   private:
    ProbitRegressionModel *model_;
    Ptr<MvnBase> prior_;

    BinomialProbitDataImputer imputer_;

    // Complete data sufficient statistics.
    SpdMatrix xtx_;
    Vector xtz_;
  };
}  // namespace BOOM

#endif  // BOOM_PROBIT_REGRESSION_SAMPLER_HPP_
