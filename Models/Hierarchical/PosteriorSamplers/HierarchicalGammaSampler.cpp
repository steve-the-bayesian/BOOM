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

#include "Models/Hierarchical/PosteriorSamplers/HierarchicalGammaSampler.hpp"

namespace BOOM {

  HierarchicalGammaSampler::HierarchicalGammaSampler(
      HierarchicalGammaModel *model,
      const Ptr<DoubleModel> &gamma_mean_mean_prior,
      const Ptr<DoubleModel> &gamma_mean_shape_prior,
      const Ptr<DoubleModel> &gamma_shape_mean_prior,
      const Ptr<DoubleModel> &gamma_shape_shape_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        gamma_mean_mean_prior_(gamma_mean_mean_prior),
        gamma_mean_shape_prior_(gamma_mean_shape_prior),
        gamma_shape_mean_prior_(gamma_shape_mean_prior),
        gamma_shape_shape_prior_(gamma_shape_shape_prior),
        gamma_mean_sampler_(new GammaPosteriorSampler(
            model_->prior_for_mean_parameters(), gamma_mean_mean_prior,
            gamma_mean_shape_prior)),
        gamma_shape_sampler_(new GammaPosteriorSampler(
            model_->prior_for_shape_parameters(), gamma_shape_mean_prior,
            gamma_shape_shape_prior)) {
    model_->prior_for_mean_parameters()->set_method(gamma_mean_sampler_);
    model_->prior_for_shape_parameters()->set_method(gamma_shape_sampler_);
  }

  double HierarchicalGammaSampler::logpri() const {
    return gamma_mean_sampler_->logpri() + gamma_shape_sampler_->logpri();
  }

  // The draw() method will draw values of the gamma_mean and gamma_shape
  void HierarchicalGammaSampler::draw() {
    model_->prior_for_mean_parameters()->clear_data();
    model_->prior_for_shape_parameters()->clear_data();

    for (int i = 0; i < model_->number_of_groups(); ++i) {
      GammaModel *data_model = model_->data_model(i);
      ensure_posterior_sampling_method(data_model);
      data_model->sample_posterior();
      model_->prior_for_mean_parameters()->suf()->update_raw(
          data_model->mean());
      model_->prior_for_shape_parameters()->suf()->update_raw(
          data_model->alpha());
    }

    model_->prior_for_mean_parameters()->sample_posterior();
    model_->prior_for_shape_parameters()->sample_posterior();
  }

  void HierarchicalGammaSampler::ensure_posterior_sampling_method(
      GammaModel *data_model) {
    if (data_model->number_of_sampling_methods() == 0) {
      NEW(GammaPosteriorSampler, sampler)
      (data_model, model_->prior_for_mean_parameters(),
       model_->prior_for_shape_parameters());
      data_model->set_method(sampler);
    }
  }

}  // namespace BOOM
