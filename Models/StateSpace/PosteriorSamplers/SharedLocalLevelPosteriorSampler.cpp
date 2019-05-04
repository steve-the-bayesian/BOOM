/*
  Copyright (C) 2018 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/StateSpace/PosteriorSamplers/SharedLocalLevelPosteriorSampler.hpp"

namespace BOOM {

  namespace {
    using SLLPS = SharedLocalLevelPosteriorSampler;
  }  // namespace

  SLLPS::SharedLocalLevelPosteriorSampler(
      SharedLocalLevelStateModel *model,
      const std::vector<Ptr<GammaModelBase>> &innovation_precision_priors,
      const Matrix &coefficient_prior_mean,
      double observation_coefficient_prior_sample_size,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        innovation_precision_priors_(innovation_precision_priors)
  {
    if (innovation_precision_priors_.size() != model_->state_dimension()) {
      std::ostringstream err;
      err << "The model has state dimension " << model_->state_dimension()
          << " but " << innovation_precision_priors_.size()
          << " priors were passed.  They should match.";
      report_error(err.str());
    }

    for (int i = 0; i < innovation_precision_priors_.size(); ++i) {
      variance_samplers_.emplace_back(innovation_precision_priors_[i]);
    }

    observation_coefficient_sampler_.reset(
        new MultivariateRegressionSampler(
            model_->coefficient_model().get(),
            coefficient_prior_mean.transpose(),
            observation_coefficient_prior_sample_size,
            1.0,
            SpdMatrix(innovation_precision_priors_.size()),
            rng()));
  }

  double SLLPS::logpri() const {
    double ans = 0;
    for (int i = 0; i < innovation_precision_priors_.size(); ++i) {
      ans += variance_samplers_[i].log_prior(
          model_->innovation_model(i)->sigsq());
    }
    ans += observation_coefficient_sampler_->logpri();
    return ans;
  }
  
  void SLLPS::draw() {
    for (int i = 0; i < variance_samplers_.size(); ++i) {
      Ptr<ZeroMeanGaussianModel> innovation_model = model_->innovation_model(i);
      const auto suf = innovation_model->suf();
      double sigsq = variance_samplers_[i].draw(
          rng(), suf->n(), suf->sumsq());
      innovation_model->set_sigsq(sigsq);
    }

    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    ////////// Need to copy the variance from the observation model into the
    ////////// coefficient model, or use a different prior.
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    observation_coefficient_sampler_->draw_Beta();
    model_->impose_identifiability_constraint();
  }
  
}  // namespace BOOM
