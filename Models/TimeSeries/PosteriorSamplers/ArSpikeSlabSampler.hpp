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

#ifndef BOOM_AR_SPIKE_SLAB_SAMPLER_HPP_
#define BOOM_AR_SPIKE_SLAB_SAMPLER_HPP_

#include "Models/GammaModel.hpp"
#include "Models/Glm/PosteriorSamplers/SpikeSlabSampler.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/TimeSeries/ArModel.hpp"

// A posterior sampler for ArModel's where the prior distribution for
// the AR coefficients is a spike and slab.
//
//   phi | gamma ~ N(phi0, V_g)
//
// where V_g^{-1} is is the subset of V^{-1} where gamma == 1.  In
// most cases phi0 == 0, but a general value of phi0 is assumed here.
//
// The support of the prior is optionally truncated to the stationary
// region of the parameter space.  This is somewhat dicey, as the
// truncation is ignored by the portion of the spike and slab
// algorithm that draws inclusion indicators.
//
// TODO: Consider extending this to use the prior by Huerta and West (1999), who
// put spike and slab priors on roots of the AR polynomial.
namespace BOOM {
  class ArSpikeSlabSampler : public PosteriorSampler {
   public:
    ArSpikeSlabSampler(ArModel *model,
                       const Ptr<MvnBase> &slab,
                       const Ptr<VariableSelectionPrior> &spike,
                       const Ptr<GammaModelBase> &residual_precision_prior,
                       bool truncate_support_to_stationary_region = true,
                       RNG &seeding_rng = GlobalRng::rng);

    // The posterior draw conditions on sigma^2.  It makes no effort
    // to enforce stationarity.
    void draw() override;
    double logpri() const override;

    void allow_model_selection(bool allow) {
      spike_slab_sampler_.allow_model_selection(allow);
    }

    void limit_model_selection(uint nflips) {
      spike_slab_sampler_.limit_model_selection(nflips);
    }

    // Truncate the support of the residual standard deviation prior so that it
    // is less than an upper limit.  The limit can be infinity().
    void set_sigma_upper_limit(double sigma_upper_limit) {
      sigsq_sampler_.set_sigma_max(sigma_upper_limit);
    }

    // Args:
    //   truncate: If true then trunacate the support of the prior distribution
    //     to coefficients implying stationarity.  If false then remove any such
    //     restriction that might be in place.
    //
    // Effect:
    //   If the current set of coefficients are outside the stationary region,
    //   they are shrunk towards zero until stationarity is acheived.
    void truncate_support(bool truncate);

   private:
    void draw_phi();
    void draw_phi_univariate();
    void draw_sigma_full_conditional();
    void set_sufficient_statistics();

    // A utility to call when phi is outside the bounds of stationarity.  Shrink
    // phi towards zero until stationarity is achieved.
    //
    // Returns:
    //  true if phi could be shrunk to a value implying stationarity.  False
    //  otherwise.
    bool shrink_phi(Vector &phi);

    ArModel *model_;
    Ptr<MvnBase> slab_;
    Ptr<VariableSelectionPrior> spike_;
    Ptr<GammaModelBase> residual_precision_prior_;

    // A flag indicating whether the support of the prior should be
    // truncated to the stationary distribution.
    bool truncate_;
    int max_number_of_regression_proposals_;
    SpikeSlabSampler spike_slab_sampler_;
    GenericGaussianVarianceSampler sigsq_sampler_;
    WeightedRegSuf suf_;
  };
}  // namespace BOOM

#endif  //  BOOM_AR_SPIKE_SLAB_SAMPLER_HPP_
