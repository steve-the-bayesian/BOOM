// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include "Models/StateSpace/PosteriorSamplers/DynamicRegressionPosteriorSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using DRIPS = DynamicRegressionIndependentPosteriorSampler;
  }

  DRIPS::DynamicRegressionIndependentPosteriorSampler(
      DynamicRegressionStateModel *model,
      const std::vector<Ptr<GammaModelBase>> &innovation_precision_priors,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        priors_(innovation_precision_priors) {
    if (innovation_precision_priors.size() == 1) {
      for (int i = 1; i < model_->xdim(); ++i) {
        priors_.push_back(priors_[0]->clone());
      }
    }
    if (priors_.size() != model_->xdim()) {
      report_error(
          "The number of prior distributions must be the same "
          "as the number of coefficients in the dynamic regression.");
    }
    for (int i = 0; i < priors_.size(); ++i) {
      samplers_.push_back(GenericGaussianVarianceSampler(priors_[i]));
    }
  }

  DRIPS *DRIPS::clone_to_new_host(Model *new_host) const {
    std::vector<Ptr<GammaModelBase>> new_priors;
    for (const auto &el : priors_) {
      new_priors.push_back(el->clone());
    }
    DRIPS *ans = new DRIPS(
        dynamic_cast<DynamicRegressionStateModel *>(new_host),
        new_priors,
        rng());
    for (int i = 0; i < samplers_.size(); ++i) {
      ans->set_sigma_max(i, samplers_[i].sigma_max());
    }
    return ans;
  }

  void DRIPS::draw() {
    for (int i = 0; i < samplers_.size(); ++i) {
      double sigsq = samplers_[i].draw(rng(), model_->suf(i)->n(),
                                       model_->suf(i)->sumsq());
      model_->set_sigsq(sigsq, i);
    }
  }

  double DRIPS::logpri() const {
    double ans = 0;
    for (int i = 0; i < samplers_.size(); ++i) {
      ans += samplers_[i].log_prior(model_->sigsq(i));
    }
    return ans;
  }

  void DRIPS::set_sigma_max(int coefficient, double value) {
    samplers_[coefficient].set_sigma_max(value);
  }

  DynamicRegressionPosteriorSampler::DynamicRegressionPosteriorSampler(
      DynamicRegressionStateModel *model, const Ptr<GammaModel> &siginv_prior,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        siginv_prior_(siginv_prior),
        sigsq_sampler_(siginv_prior_),
        handle_siginv_prior_separately_(false) {}

  DynamicRegressionPosteriorSampler *
  DynamicRegressionPosteriorSampler::clone_to_new_host(Model *model) const {
    DynamicRegressionPosteriorSampler *ans =
        new DynamicRegressionPosteriorSampler(
            dynamic_cast<DynamicRegressionStateModel *>(model),
            siginv_prior_,
            rng());
    if (handle_siginv_prior_separately_) {
      ans->handle_siginv_prior_separately();
    }
    ans->set_sigma_max(sigsq_sampler_.sigma_max());
    return ans;
  }

  void DynamicRegressionPosteriorSampler::handle_siginv_prior_separately() {
    handle_siginv_prior_separately_ = true;
  }

  void DynamicRegressionPosteriorSampler::draw() {
    siginv_prior_->clear_data();
    for (int i = 0; i < model_->state_dimension(); ++i) {
      const GaussianSuf *suf = model_->suf(i);
      double sumsq = suf->sumsq() * model_->predictor_variance()[i];
      double sigsq = sigsq_sampler_.draw(rng(), suf->n(), sumsq);
      model_->set_sigsq(sigsq, i);
      siginv_prior_->suf()->update_raw(1.0 / sigsq);
    }
    if (!handle_siginv_prior_separately_) {
      siginv_prior_->sample_posterior();
    }
  }

  double DynamicRegressionPosteriorSampler::logpri() const {
    double ans = 0;
    for (int i = 0; i < model_->state_dimension(); ++i) {
      sigsq_sampler_.log_prior(model_->sigsq(i));
    }
    if (!handle_siginv_prior_separately_) {
      ans += siginv_prior_->logpri();
    }
    return ans;
  }

}  // namespace BOOM
