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

#include "Models/PosteriorSamplers/IndependentMvnVarSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  IndependentMvnVarSampler::IndependentMvnVarSampler(
      IndependentMvnModel *model,
      const std::vector<Ptr<GammaModelBase>> &priors,
      Vector sd_max_values,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), model_(model), priors_(priors) {
    if (priors.size() != model->dim()) {
      report_error(
          "Prior dimension does not match model in "
          "IndependentMvnVarSampler");
    }
    if (sd_max_values.empty()) {
      sd_max_values.resize(model->dim(), infinity());
    }
    if (sd_max_values.size() != model->dim()) {
      report_error(
          "sd_max_values.size() != model->dim() in "
          "IndependentMvnVarSampler");
    }
    for (int i = 0; i < model->dim(); ++i) {
      samplers_.push_back(
          GenericGaussianVarianceSampler(priors_[i], sd_max_values[i]));
    }
  }

  IndependentMvnVarSampler *IndependentMvnVarSampler::clone_to_new_host(
      Model *new_host) const {
    std::vector<Ptr<GammaModelBase>> priors;
    Vector sd_max_values;
    for (int i = 0; i < priors_.size(); ++i) {
      priors.push_back(priors_[i]->clone());
      sd_max_values.push_back(samplers_[i].sigma_max());
    }
    return new IndependentMvnVarSampler(
        dynamic_cast<IndependentMvnModel *>(new_host),
        priors,
        sd_max_values,
        rng());
  }

  double IndependentMvnVarSampler::logpri() const {
    const Vector &sigsq(model_->sigsq());
    double ans = 0;
    for (int i = 0; i < sigsq.size(); ++i) {
      ans += samplers_[i].log_prior(sigsq[i]);
    }
    return ans;
  }

  void IndependentMvnVarSampler::draw() {
    Ptr<IndependentMvnSuf> suf = model_->suf();
    for (int i = 0; i < model_->dim(); ++i) {
      double sigsq = samplers_[i].draw(rng(), suf->n(i),
                                       suf->centered_sumsq(i, model_->mu()[i]));
      model_->set_sigsq_element(sigsq, i);
    }
  }

  void IndependentMvnVarSampler::set_sigma_max(const Vector &sigma_max) {
    if (sigma_max.size() != samplers_.size()) {
      std::ostringstream err;
      err << "Size mismatch in set_sigma_match.  Vector of samplers has size "
          << samplers_.size() << " but vector of limits has size "
          << sigma_max.size() << ".\n";
      report_error(err.str());
    }
    for (int i = 0; i < sigma_max.size(); ++i) {
      samplers_[i].set_sigma_max(sigma_max[i]);
    }
  }

}  // namespace BOOM
