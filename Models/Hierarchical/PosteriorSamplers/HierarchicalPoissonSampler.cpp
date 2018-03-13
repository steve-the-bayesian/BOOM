// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#include "Models/Hierarchical/PosteriorSamplers/HierarchicalPoissonSampler.hpp"
#include "Models/PosteriorSamplers/GammaPosteriorSampler.hpp"
#include "Models/PosteriorSamplers/PoissonGammaSampler.hpp"

namespace BOOM {

  HierarchicalPoissonSampler::HierarchicalPoissonSampler(
      HierarchicalPoissonModel *model, const Ptr<DoubleModel> &gamma_mean_prior,
      const Ptr<DoubleModel> &gamma_sample_size_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        gamma_mean_prior_(gamma_mean_prior),
        gamma_sample_size_prior_(gamma_sample_size_prior) {
    GammaModel *prior = model_->prior_model();
    prior->clear_methods();
    NEW(GammaPosteriorSampler, prior_sampler)
    (prior, gamma_mean_prior_, gamma_sample_size_prior_, rng());
    prior->set_method(prior_sampler);
  }

  double HierarchicalPoissonSampler::logpri() const {
    const GammaModel *prior = model_->prior_model();
    return gamma_mean_prior_->logp(prior->mean()) +
           gamma_sample_size_prior_->logp(prior->alpha());
  }

  void HierarchicalPoissonSampler::draw() {
    GammaModel *prior = model_->prior_model();
    prior->clear_data();
    for (int i = 0; i < model_->number_of_groups(); ++i) {
      PoissonModel *data_model = model_->data_model(i);
      if (data_model->number_of_sampling_methods() != 1) {
        data_model->clear_methods();
        NEW(PoissonGammaSampler, data_model_sampler)
        (data_model, Ptr<GammaModel>(prior), rng());
        data_model->set_method(data_model_sampler);
      }
      int number_attempts = 0;
      do {
        data_model->sample_posterior();
        if (++number_attempts > 1000) {
          report_error(
              "Too many attempts to draw a positive mean in "
              "HierarchicalPoissonSampler::draw");
        }
      } while (data_model->lam() == 0);
      prior->suf()->update_raw(data_model->lam());
    }
    prior->sample_posterior();
  }

}  // namespace BOOM
