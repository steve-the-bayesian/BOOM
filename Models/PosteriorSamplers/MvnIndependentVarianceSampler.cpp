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

#include "Models/PosteriorSamplers/MvnIndependentVarianceSampler.hpp"

#include "distributions.hpp"
#include "distributions/trun_gamma.hpp"

namespace BOOM {

  MvnIndependentVarianceSampler::MvnIndependentVarianceSampler(
      MvnModel *model,
      const std::vector<Ptr<GammaModelBase> > &siginv_priors,
      const Vector &sigma_max_values,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), model_(model), priors_(siginv_priors) {
    if (model->dim() != siginv_priors.size()) {
      report_error(
          "The model and siginv_priors arguments do not conform in "
          "the MvnIndependentVarianceSampler constructor.");
    }

    if (model->dim() != sigma_max_values.size()) {
      report_error(
          "The model and sigma_max_values arguments do "
          "not conform in the MvnIndependentVarianceSampler "
          "constructor.");
    }

    for (int i = 0; i < model->dim(); ++i) {
      if (sigma_max_values[i] < 0) {
        report_error(
            "All elements of sigma_max_values must be "
            "non-negative in "
            "MvnIndependentVarianceSampler constructor.");
      }
    }

    for (int i = 0; i < model->dim(); ++i) {
      GenericGaussianVarianceSampler sampler(priors_[i], sigma_max_values[i]);
      sigsq_samplers_.push_back(sampler);
    }
  }

  MvnIndependentVarianceSampler *
  MvnIndependentVarianceSampler::clone_to_new_host(Model *new_host) const {
    std::vector<Ptr<GammaModelBase>> priors;
    Vector sigma_max_values;
    for (int i = 0; i < priors_.size(); ++i) {
      priors.push_back(priors_[i]->clone());
      sigma_max_values.push_back(sigsq_samplers_[i].sigma_max());
    }

    return new MvnIndependentVarianceSampler(
        dynamic_cast<MvnModel *>(new_host),
        priors,
        sigma_max_values,
        rng());
  }

  void MvnIndependentVarianceSampler::draw() {
    SpdMatrix diagonal_inverse_variance = model_->siginv();
    SpdMatrix sumsq = model_->suf()->center_sumsq(model_->mu());
    // Because the variance matrix is diagonal, we can simply draw its
    // diagonal elements one at a time.
    for (int i = 0; i < model_->dim(); ++i) {
      double sigsq =
          sigsq_samplers_[i].draw(rng(), model_->suf()->n(), sumsq(i, i));
      diagonal_inverse_variance(i, i) = 1.0 / sigsq;
    }
    model_->set_siginv(diagonal_inverse_variance);
  }

  double MvnIndependentVarianceSampler::logpri() const {
    double ans = 0;
    for (int i = 0; i < priors_.size(); ++i) {
      double siginv = model_->siginv()(i, i);
      ans += sigsq_samplers_[i].log_prior(1.0 / siginv);
    }
    return ans;
  }

}  // namespace BOOM
