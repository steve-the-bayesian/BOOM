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
#ifndef BOOM_DYNAMIC_REGRESSION_POSTERIOR_SAMPLER_HPP_
#define BOOM_DYNAMIC_REGRESSION_POSTERIOR_SAMPLER_HPP_

#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/StateSpace/StateModels/DynamicRegressionStateModel.hpp"
#include "Models/GammaModel.hpp"

namespace BOOM {

class DynamicRegressionPosteriorSampler : public PosteriorSampler {
 public:

  // The prior distribution for siginv_prior should be set before
  // passing it to this constructor.
  DynamicRegressionPosteriorSampler(DynamicRegressionStateModel *model,
                                    const Ptr<GammaModel> &siginv_prior,
                                    RNG &seeding_rng = GlobalRng::rng);

  // By default the class will take control of siginv_prior, updating
  // it when draw() is called, and adding its contribution to
  // logpri().  If you want to avoid this behavior and manage
  // siginv_prior outside of this class then call
  // handle_siginv_prior_separately().
  void handle_siginv_prior_separately();

  // logpri() returns the prior with respect to sigma[i].  It does not
  // return the hyperprior of siginv_prior.  A separate call to
  // siginv_prior->logpri() will return that.
  double logpri() const override;

  // draw() will update each sigma[i] and update the sufficient
  // statistics for siginv_prior_.  You still need to call
  // siginv_prior_->sample_posterior().
  void draw() override;

 private:
  DynamicRegressionStateModel *model_;
  Ptr<GammaModel> siginv_prior_;
  GenericGaussianVarianceSampler sigsq_sampler_;
  bool handle_siginv_prior_separately_;
};

}  // namespace BOOM

#endif //  BOOM_DYNAMIC_REGRESSION_POSTERIOR_SAMPLER_HPP_
