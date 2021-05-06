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
#include "Models/PosteriorSamplers/GaussianConjSampler.hpp"
#include "Models/GaussianModel.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  typedef GaussianConjSampler GCS;
  GCS::GaussianConjSampler(GaussianModel *m,
                           const Ptr<GaussianModelGivenSigma> &mu,
                           const Ptr<GammaModelBase> &sig,
                           RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        mod_(m),
        mu_(mu),
        siginv_(sig),
        sigsq_sampler_(siginv_) {}

  GaussianConjSampler *GCS::clone_to_new_host(Model *host) const {
    return new GaussianConjSampler(
        dynamic_cast<GaussianModel *>(host),
        mu_->clone(),
        siginv_->clone(),
        rng());
  }

  double GCS::logpri() const {
    return sigsq_sampler_.log_prior(mod_->sigsq()) + mu_->logp(mod_->mu());
  }

  double GCS::mu() const { return mu_->mu(); }
  double GCS::kappa() const { return mu_->kappa(); }
  double GCS::df() const { return 2 * siginv_->alpha(); }
  double GCS::ss() const { return 2 * siginv_->beta(); }

  void GCS::draw() {
    // sufficient statistics
    double n = mod_->suf()->n();
    double ybar = mod_->ybar();
    double v = mod_->sample_var();

    // prior parameters
    double kappa = this->kappa();
    double mu0 = this->mu();

    double mu_hat = (n * mod_->ybar() + kappa * mu0) / (n + kappa);

    double ss = (n - 1) * v + n * kappa * square(ybar - mu0) / (n + kappa);
    double sigsq = sigsq_sampler_.draw(rng(), n, ss);

    v = sigsq / (n + kappa);
    double mu = rnorm_mt(rng(), mu_hat, sqrt(v));
    mod_->set_params(mu, sigsq);
  }

  void GCS::find_posterior_mode(double) {
    double n = mod_->suf()->n();
    double ybar = mod_->ybar();

    double k = this->kappa();
    double mu0 = this->mu();
    double ss = this->ss();
    double df = this->df();

    double DF = df + n;
    double mu_hat = (n * ybar + k * mu0) / (n + k);

    double SS = ss + (n - 1) * mod_->sample_var();
    SS += k * square(mu0 - mu_hat) + n * square(ybar - mu_hat);

    mod_->set_params(mu_hat, SS / (DF - 1));
  }

}  // namespace BOOM
