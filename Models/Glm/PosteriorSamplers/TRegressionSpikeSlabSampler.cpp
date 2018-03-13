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

#include "Models/Glm/PosteriorSamplers/TRegressionSpikeSlabSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  TRegressionSpikeSlabSampler::TRegressionSpikeSlabSampler(
      TRegressionModel *model, const Ptr<MvnBase> &coefficient_slab_prior,
      const Ptr<VariableSelectionPrior> &coefficient_spike_prior,
      const Ptr<GammaModelBase> &siginv_prior, const Ptr<DoubleModel> &nu_prior,
      RNG &seeding_rng)
      : TRegressionSampler(model, coefficient_slab_prior, siginv_prior,
                           nu_prior, seeding_rng),
        model_(model),
        sam_(model, coefficient_slab_prior, coefficient_spike_prior),
        coefficient_slab_prior_(coefficient_slab_prior),
        coefficient_spike_prior_(coefficient_spike_prior),
        siginv_prior_(siginv_prior),
        nu_prior_(nu_prior) {}

  void TRegressionSpikeSlabSampler::draw() {
    impute_latent_data();
    draw_model_indicators();
    draw_included_coefficients();
    draw_sigsq_full_conditional();
    draw_nu_given_observed_data();
  }

  double TRegressionSpikeSlabSampler::logpri() const {
    return sam_.logpri() + nu_prior_->logp(model_->nu()) +
           siginv_prior_->logp(1.0 / model_->sigsq());
  }

  void TRegressionSpikeSlabSampler::draw_model_indicators() {
    sam_.draw_model_indicators(rng(), complete_data_sufficient_statistics(),
                               model_->sigsq());
  }

  void TRegressionSpikeSlabSampler::draw_included_coefficients() {
    sam_.draw_beta(rng(), complete_data_sufficient_statistics(),
                   model_->sigsq());
  }

  void TRegressionSpikeSlabSampler::allow_model_selection(bool allow) {
    sam_.allow_model_selection(allow);
  }

  void TRegressionSpikeSlabSampler::limit_model_selection(int max_flips) {
    sam_.limit_model_selection(max_flips);
  }

}  // namespace BOOM
