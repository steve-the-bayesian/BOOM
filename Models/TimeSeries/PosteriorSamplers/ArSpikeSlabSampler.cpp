// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

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

#include "Models/TimeSeries/PosteriorSamplers/ArSpikeSlabSampler.hpp"
#include "LinAlg/SWEEP.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  ArSpikeSlabSampler::ArSpikeSlabSampler(
      ArModel *model, const Ptr<MvnBase> &slab,
      const Ptr<VariableSelectionPrior> &spike,
      const Ptr<GammaModelBase> &residual_precision_prior, bool truncate,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        slab_(slab),
        spike_(spike),
        residual_precision_prior_(residual_precision_prior),
        truncate_(truncate),
        max_number_of_regression_proposals_(100),
        spike_slab_sampler_(model_, slab_, spike_),
        sigsq_sampler_(residual_precision_prior_),
        suf_(model_->xdim()) {}

  // ---------------------------------------------------------------------------
  void ArSpikeSlabSampler::draw() {
    set_sufficient_statistics();
    spike_slab_sampler_.draw_model_indicators(rng(), suf_, model_->sigsq());
    draw_phi();
    draw_sigma_full_conditional();
  }

  // ---------------------------------------------------------------------------
  double ArSpikeSlabSampler::logpri() const {
    if (truncate_ && !model_->check_stationary(model_->phi())) {
      return negative_infinity();
    }
    return spike_slab_sampler_.logpri() +
           sigsq_sampler_.log_prior(model_->sigsq());
  }

  // ---------------------------------------------------------------------------
  void ArSpikeSlabSampler::truncate_support(bool truncate) {
    if (truncate && !truncate_) {
      Vector phi = model_->phi();
      if (!shrink_phi(phi)) {
          report_error(
              "Could not shrink AR coefficient vector to "
              "stationary region.");
      }
      model_->set_phi(phi);
    }
    truncate_ = truncate;
  }

  // ---------------------------------------------------------------------------
  void ArSpikeSlabSampler::draw_phi() {
    Vector original_phi = model_->phi();
    int attempts = 0;
    bool ok = false;
    while (!ok && attempts < max_number_of_regression_proposals_) {
      ++attempts;
      spike_slab_sampler_.draw_beta(rng(), suf_, model_->sigsq());
      if (truncate_) {
        ok = model_->check_stationary(model_->phi());
      } else {
        ok = true;
      }
    }
    if (!ok) {
      model_->set_phi(original_phi);
      try {
        draw_phi_univariate();
      } catch(...) {
        model_->set_phi(original_phi);
      }
    }
  }

  // ---------------------------------------------------------------------------
  bool ArSpikeSlabSampler::shrink_phi(Vector &phi) {
    int attempts = 0;
    int max_attempts = 20;
    while (attempts++ < max_attempts and !model_->check_stationary(phi)) {
      phi *= .95;
    }
    return attempts < max_attempts;
  }
  // ---------------------------------------------------------------------------
  void ArSpikeSlabSampler::draw_phi_univariate() {
    const Selector &inc(model_->coef().inc());
    int p = inc.nvars();
    Vector phi = model_->included_coefficients();
    if (!model_->check_stationary(model_->phi())) {
      if (!shrink_phi(phi)) {
        report_error(
            "ArSpikeSlabSampler::draw_phi_univariate was called with an "
            "illegal initial value of phi.  That should never happen.");
      }
    }
    double sigsq = model_->sigsq();

    const SpdMatrix prior_precision = inc.select(slab_->siginv());
    const SpdMatrix precision =
        inc.select(model_->suf()->xtx()) / sigsq + prior_precision;
    const Vector posterior_mean =
        precision.solve(inc.select(model_->suf()->xty()) / sigsq +
                        prior_precision * inc.select(slab_->mu()));

    for (int i = 0; i < p; ++i) {
      SweptVarianceMatrix swept_precision(precision, true);
      swept_precision.RSW(i);
      Selector conditional(p, true);
      conditional.drop(i);

      if (conditional.nvars() == 0) {
        continue;
      }

      double conditional_mean = swept_precision.conditional_mean(
          conditional.select(phi), posterior_mean)[0];
      double conditional_sd = sqrt(swept_precision.residual_variance()(0, 0));

      double initial_phi = phi[i];
      double lo = -1;
      double hi = 1;

      bool ok = false;
      // The following is mathematically guaranteed to terminate, but
      // we limit the number of attempts just to be safe.
      int max_attempts = 1000;
      int attempts = 0;
      while (!ok) {
        if (attempts++ > max_attempts) {
          report_error("Too many attempts in draw_phi_univariate.");
        }
        double candidate =
            rtrun_norm_2_mt(rng(), conditional_mean, conditional_sd, lo, hi);
        phi[i] = candidate;
        if (ArModel::check_stationary(inc.expand(phi))) {
          ok = true;
        } else {
          if (candidate > initial_phi)
            hi = candidate;
          else
            lo = candidate;
        }
      }
    }
    model_->set_phi(phi);
  }

  // ---------------------------------------------------------------------------
  void ArSpikeSlabSampler::draw_sigma_full_conditional() {
    double data_df = model_->suf()->n();
    const Selector &inc(model_->coef().inc());
    double data_ss;
    if (inc.nvars() == 0) {
      data_ss = model_->suf()->yty();
    } else {
      data_ss = model_->suf()->relative_sse(model_->coef());
    }
    double sigsq = sigsq_sampler_.draw(rng(), data_df, data_ss);
    model_->set_sigsq(sigsq);
  }

  // ---------------------------------------------------------------------------
  void ArSpikeSlabSampler::set_sufficient_statistics() {
    suf_.set_xtwx(model_->suf()->xtx());
    suf_.set_xtwy(model_->suf()->xty());
  }

}  // namespace BOOM
