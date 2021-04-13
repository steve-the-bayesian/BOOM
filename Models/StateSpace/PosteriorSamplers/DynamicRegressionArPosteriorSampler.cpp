// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#include "Models/StateSpace/PosteriorSamplers/DynamicRegressionArPosteriorSampler.hpp"

namespace BOOM {

  namespace {
    typedef DynamicRegressionArPosteriorSampler DRARPS;
  }

  DRARPS::DynamicRegressionArPosteriorSampler(
      DynamicRegressionArStateModel *model,
      const std::vector<Ptr<GammaModelBase>> &siginv_priors, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), model_(model) {
    if (siginv_priors.size() != model_->xdim()) {
      report_error("Wrong number of variance priors supplied.");
    }
    samplers_.reserve(model_->xdim());
    for (int i = 0; i < model_->xdim(); ++i) {
      NEW(ArPosteriorSampler, sampler)
      (model_->coefficient_model(i).get(), siginv_priors[i], seeding_rng);
      model_->coefficient_model(i)->set_method(sampler);
      samplers_.push_back(sampler);
    }
  }

  DRARPS *DRARPS::clone_to_new_host(Model *new_host) const {
    std::vector<Ptr<GammaModelBase>> siginv_priors;
    for (const auto &el : samplers_) {
      siginv_priors.push_back(el->residual_precision_prior());
    }
    return new DRARPS(dynamic_cast<DynamicRegressionArStateModel *>(new_host),
                      siginv_priors,
                      rng());
  }

  void DRARPS::draw() {
    for (int i = 0; i < model_->xdim(); ++i) {
      samplers_[i]->draw();
    }
  }

  double DRARPS::logpri() const {
    double ans = 0;
    for (int i = 0; i < model_->xdim(); ++i) {
      ans += samplers_[i]->logpri();
    }
    return ans;
  }

}  // namespace BOOM
