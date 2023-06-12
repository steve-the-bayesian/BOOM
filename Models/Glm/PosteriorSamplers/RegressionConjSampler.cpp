// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#include "Models/Glm/PosteriorSamplers/RegressionConjSampler.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef RegressionConjSampler RCS;
  RCS::RegressionConjSampler(
      RegressionModel *model, const Ptr<MvnGivenScalarSigmaBase> &coefficient_prior,
      const Ptr<GammaModelBase> &residual_precision_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        coefficient_prior_(coefficient_prior),
        residual_precision_prior_(residual_precision_prior),
        sigsq_sampler_(residual_precision_prior_) {}

  void RCS::set_posterior_suf() {
    const Vector &prior_mean(coefficient_prior_->mu());
    SpdMatrix unscaled_prior_precision = coefficient_prior_->unscaled_precision();
    posterior_precision_ = unscaled_prior_precision + model_->suf()->xtx();
    posterior_mean_ = model_->suf()->xty() + unscaled_prior_precision * prior_mean;
    posterior_mean_ = posterior_precision_.solve(posterior_mean_);
    SS_ = model_->suf()->relative_sse(posterior_mean_)
        + unscaled_prior_precision.Mdist(posterior_mean_, prior_mean);
    DF_ = model_->suf()->n();
  }

  void RCS::draw() {
    set_posterior_suf();
    double sigsq = sigsq_sampler_.draw(rng(), DF_, SS_);
    model_->set_sigsq(sigsq);
    posterior_precision_ /= sigsq;
    Vector beta = rmvn_ivar_mt(
        rng(), posterior_mean_, posterior_precision_);
    // if (beta[0] < -10 || beta[0] > 10) {
    //   std::cout << "Unusual draw of beta[0] in RegressionConjSampler...\n"
    //             << "xtx = " << model_->suf()->xtx()
    //             << "xty = " << model_->suf()->xty() << "\n"
    //             << "posterior_mean = " << posterior_mean_ << "\n"
    //             << "posterior_precision_ = " << posterior_precision_
    //             << "sigsq = " << sigsq << "\n" ;
    // }
    model_->set_Beta(beta);
  }

  void RCS::find_posterior_mode(double) {
    set_posterior_suf();
    model_->set_Beta(posterior_mean_);
    double DF = DF_ + prior_df();
    double SS = SS_ + prior_ss();
    if (DF <= 2)
      model_->set_sigsq(0.0);  // mode = (alpha-1)/beta
    else
      model_->set_sigsq(SS / (DF - 2));  //   alpha = df/2  beta = ss/2
  }

  double RCS::logpri() const {
    double ans = coefficient_prior_->logp(model_->Beta());
    ans += sigsq_sampler_.log_prior(model_->sigsq());
    return ans;
  }

}  // namespace BOOM
