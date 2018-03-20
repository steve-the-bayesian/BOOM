#ifndef BOOM_ARMA_SLICE_SAMPLER_HPP_
#define BOOM_ARMA_SLICE_SAMPLER_HPP_

/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "Models/DoubleModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/TimeSeries/ArmaModel.hpp"
#include "Models/VectorModel.hpp"
#include "Samplers/UnivariateSliceSampler.hpp"

namespace BOOM {

  // A posterior sampler for ARMA models based on the slice sampler.  This is
  // computationally slow, but other approaches seem to be even slower.
  class ArmaSliceSampler : public PosteriorSampler {
   public:
    ArmaSliceSampler(ArmaModel *model, const Ptr<VectorModel> &ar_prior,
                     const Ptr<VectorModel> &ma_prior,
                     const Ptr<DoubleModel> &precision_prior,
                     RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    double log_posterior(const Vector &ar_coefficients,
                         const Vector &ma_coefficients, double precision) const;
    double vectorized_log_posterior(const Vector &parameters) const;

   private:
    ArmaModel *model_;
    Ptr<VectorModel> ar_prior_;
    Ptr<VectorModel> ma_prior_;
    Ptr<DoubleModel> precision_prior_;

    UnivariateSliceSampler sampler_;
  };

}  // namespace BOOM

#endif  // BOOM_ARMA_SLICE_SAMPLER_HPP_
