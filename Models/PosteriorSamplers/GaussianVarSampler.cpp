// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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
#include "Models/PosteriorSamplers/GaussianVarSampler.hpp"
#include "Models/GammaModel.hpp"
#include "Models/GaussianModel.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  typedef GaussianVarSampler GVS;
  GVS::GaussianVarSampler(GaussianModel *model,
                          const Ptr<GammaModelBase> &precision_prior,
                          RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        prior_(precision_prior),
        model_(model),
        sampler_(prior_) {}

  inline double sumsq(double nu, double sig) { return nu * sig * sig; }

  GVS::GaussianVarSampler(GaussianModel *model, double prior_df,
                          double prior_sigma_guess, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        prior_(new GammaModel(prior_df / 2.0,
                              sumsq(prior_df, prior_sigma_guess) / 2.0)),
        model_(model),
        sampler_(prior_) {}

  GVS *GVS::clone_to_new_host(Model *new_host) const {
    GVS *ans = new GVS(dynamic_cast<GaussianModel *>(new_host),
                       prior_->clone(),
                       rng());
    ans->set_sigma_upper_limit(sampler_.sigma_max());
    return ans;
  }

  void GVS::draw() {
    double n = model_->suf()->n();
    double sumsq = model_->suf()->centered_sumsq(model_->mu());
    double sigsq = sampler_.draw(rng(), n, sumsq);
    model_->set_sigsq(sigsq);
  }

  double GVS::logpri() const { return sampler_.log_prior(model_->sigsq()); }

  void GVS::set_sigma_upper_limit(double max_sigma) {
    sampler_.set_sigma_max(max_sigma);
  }

  const Ptr<GammaModelBase> GVS::ivar() const { return prior_; }

  void GVS::find_posterior_mode(double) {
    double n = model_->suf()->n();
    double sumsq = model_->suf()->centered_sumsq(model_->mu());
    model_->set_sigsq(sampler_.posterior_mode(n, sumsq));
  }

}  // namespace BOOM
