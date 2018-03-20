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

#include "Models/Hierarchical/PosteriorSamplers/HierarchicalDirichletPosteriorSampler.hpp"
#include "Models/Hierarchical/HierarchicalDirichletModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  namespace {
    typedef HierarchicalDirichletPosteriorSampler HDPS;
  }

  HDPS::HierarchicalDirichletPosteriorSampler(
      HierarchicalDirichletModel *model,
      const Ptr<DiffVectorModel> &dirichlet_mean_prior,
      const Ptr<DiffDoubleModel> &dirichlet_sample_size_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        dirichlet_mean_prior_(dirichlet_mean_prior),
        dirichlet_sample_size_prior_(dirichlet_sample_size_prior),
        sampler_(new DirichletPosteriorSampler(
            model_->prior_model(), dirichlet_mean_prior_,
            dirichlet_sample_size_prior_, rng())) {
    model_->prior_model()->set_method(sampler_);
  }

  double HDPS::logpri() const {
    const DirichletModel *prior = model_->prior_model();
    double ans = dirichlet_mean_prior_->logp(prior->pi());
    ans += dirichlet_sample_size_prior_->logp(sum(prior->nu()));
    return ans;
  }

  void HDPS::draw() {
    DirichletModel *prior = model_->prior_model();
    prior->clear_data();
    for (int i = 0; i < model_->number_of_groups(); ++i) {
      MultinomialModel *data_model = model_->data_model(i);
      if (data_model->number_of_sampling_methods() != 1) {
        data_model->clear_methods();
        NEW(MultinomialDirichletSampler, data_model_sampler)
        (data_model, Ptr<DirichletModel>(prior), rng());
        data_model->set_method(data_model_sampler);
      }
      data_model->sample_posterior();
      prior->suf()->update(*(data_model->Pi_prm()));
    }
    prior->sample_posterior();
  }

}  // namespace BOOM
