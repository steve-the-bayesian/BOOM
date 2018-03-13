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

#include "Models/TimeSeries/PosteriorSamplers/ArPosteriorSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"
#include "distributions/trun_gamma.hpp"

namespace BOOM {

  ArPosteriorSampler::ArPosteriorSampler(
      ArModel *model, const Ptr<GammaModelBase> &siginv_prior, RNG &seeding_rng)
      : HierarchicalPosteriorSampler(seeding_rng),
        model_(model),
        siginv_prior_(siginv_prior),
        max_number_of_regression_proposals_(3),
        sigsq_sampler_(siginv_prior) {}

  void ArPosteriorSampler::draw() {
    if (model_) {
      draw_model_parameters(*model_);
    }
  }

  void ArPosteriorSampler::draw_model_parameters(Model &model) {
    ArModel *ar_model = dynamic_cast<ArModel *>(&model);
    if (!ar_model) {
      report_error(
          "ArPosteriorSampler can only draw_model_parameters for "
          "objects of type ArModel.");
    }
    draw_model_parameters(*ar_model);
  }

  void ArPosteriorSampler::draw_model_parameters(ArModel &model) {
    draw_phi(model);
    draw_sigma(model);
  }

  double ArPosteriorSampler::log_prior_density(const Model &model) const {
    const ArModel *ar_model = dynamic_cast<const ArModel *>(&model);
    if (!ar_model) {
      report_error(
          "ArPosteriorSampler can only evaluate log_prior_density "
          "for ArModel objects.");
    }
    return log_prior_density(*ar_model);
  }

  double ArPosteriorSampler::log_prior_density(const ArModel &model) const {
    bool ok = model.check_stationary(model.phi());
    if (!ok) return negative_infinity();
    return sigsq_sampler_.log_prior(model.sigsq());
  }

  double ArPosteriorSampler::logpri() const {
    return model_ ? log_prior_density(*model_) : negative_infinity();
  }

  // Draws sigma given phi and observed data.
  void ArPosteriorSampler::draw_sigma(ArModel &model,
                                      double sigma_guess_scale_factor) {
    // ss = y - xb  y - xb
    //     = yty - 2 bt xty + bt xtx b
    const Vector &phi(model.phi());
    const Vector &xty(model.suf()->xty());
    const SpdMatrix &xtx(model.suf()->xtx());
    double ss = xtx.Mdist(phi) - 2 * phi.dot(xty) + model.suf()->yty();
    double df = model.suf()->n();
    double sigsq = sigsq_sampler_.draw(rng(), df, ss, sigma_guess_scale_factor);
    model.set_sigsq(sigsq);
  }

  void ArPosteriorSampler::draw_phi(ArModel &model) {
    const SpdMatrix &xtx(model.suf()->xtx());
    const Vector &xty(model.suf()->xty());
    Vector phi_hat = xtx.solve(xty);
    bool ok = false;
    int attempts = 0;
    while (!ok && ++attempts <= max_number_of_regression_proposals_) {
      Vector phi = rmvn_ivar(phi_hat, xtx / model.sigsq());
      ok = ArModel::check_stationary(phi);
      if (ok) model.set_phi(phi);
    }
    if (!ok) {
      draw_phi_univariate(model);
    }
  }

  void ArPosteriorSampler::draw_phi_univariate(ArModel &model) {
    int dim = model.phi().size();
    Vector phi = model.phi();
    if (!model.check_stationary(phi)) {
      report_error(
          "ArPosteriorSampler::draw_phi_univariate was called with an "
          "illegal initial value of phi.  That should never happen.");
    }
    const SpdMatrix &xtx(model.suf()->xtx());
    const Vector &xty(model.suf()->xty());

    for (int i = 0; i < dim; ++i) {
      double initial_phi = phi[i];
      double lo = -1;
      double hi = 1;

      // y - xb  y - xb
      //   bt xtx b  - 2 bt xty + yty
      //
      //  bt xtx b
      //   = sum_i sum_j beta[i] beta[j] xtx[i, j]
      //   = beta [i]^2 xtx[i,i] + 2 * sum_{j != i} beta[i] xtx[i, j] * beta[j]
      //                         - 2 * beta[i] * xty[i];

      // mean is (xty[i] - sum_{j != i}  )
      double ivar = xtx(i, i);
      double mu = (xty[i] - (phi.dot(xtx.col(i)) - phi[i] * xtx(i, i))) / ivar;
      bool ok = false;
      while (!ok) {
        double candidate = rtrun_norm_2_mt(rng(), mu, sqrt(1.0 / ivar), lo, hi);
        phi[i] = candidate;
        if (ArModel::check_stationary(phi)) {
          ok = true;
        } else {
          if (candidate > initial_phi)
            hi = candidate;
          else
            lo = candidate;
        }
      }
    }
    model.set_phi(phi);
  }

  void ArPosteriorSampler::set_max_number_of_regression_proposals(
      int number_of_proposals) {
    // Note that negative values are okay here.  If
    // number_of_proposals is negative, it is a signal that the user
    // always wants to draw phi one coefficient at a time.
    max_number_of_regression_proposals_ = number_of_proposals;
  }

  void ArPosteriorSampler::set_sigma_upper_limit(double max_sigma) {
    sigsq_sampler_.set_sigma_max(max_sigma);
  }

}  // namespace BOOM
