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

  DynamicRegressionPosteriorSampler::DynamicRegressionPosteriorSampler(
      DynamicRegressionStateModel *model,
      const Ptr<GammaModel> &siginv_prior,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        siginv_prior_(siginv_prior),
        sigsq_sampler_(siginv_prior_),
        handle_siginv_prior_separately_(false)
  {}

  void DynamicRegressionPosteriorSampler::handle_siginv_prior_separately(){
    handle_siginv_prior_separately_ = true;
  }

  void DynamicRegressionPosteriorSampler::draw(){
    siginv_prior_->clear_data();
    for (int i = 0; i < model_->state_dimension(); ++i) {
      const GaussianSuf *suf = model_->suf(i);
      double sumsq = suf->sumsq() * model_->predictor_variance()[i];
      double sigsq = sigsq_sampler_.draw(rng(), suf->n(), sumsq);
      model_->set_sigsq(sigsq, i);
      siginv_prior_->suf()->update_raw(1.0/sigsq);
    }
    if (!handle_siginv_prior_separately_) {
      siginv_prior_->sample_posterior();
    }
  }

  double DynamicRegressionPosteriorSampler::logpri()const{
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
