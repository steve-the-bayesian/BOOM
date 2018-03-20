// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#include "Models/Glm/PosteriorSamplers/ProbitSpikeSlabSampler.hpp"
#include <random>
#include "cpputil/math_utils.hpp"
#include "cpputil/seq.hpp"
#include "distributions.hpp"
#include "stats/logit.hpp"

namespace BOOM {
  typedef ProbitSpikeSlabSampler PSSS;
  ProbitSpikeSlabSampler::ProbitSpikeSlabSampler(
      ProbitRegressionModel *model, const Ptr<MvnBase> &prior,
      const Ptr<VariableSelectionPrior> &vspri, bool check_init,
      RNG &seeding_rng)
      : ProbitRegressionSampler(model, prior, seeding_rng),
        m_(model),
        beta_prior_(prior),
        gamma_prior_(vspri),
        max_nflips_(prior->dim()),
        allow_selection_(true) {
    if (check_init) {
      if (!std::isfinite(this->logpri())) {
        ostringstream err;
        err << "ProbitSpikeSampler initialized with an a priori "
            << "illegal value" << endl
            << "the initial Selector vector was: " << endl
            << m_->coef().inc() << endl
            << *gamma_prior_ << endl;

        report_error(err.str());
      }
    }
  }

  double PSSS::logpri() const {
    const Selector &g(m_->coef().inc());
    double ans = gamma_prior_->logp(g);
    if (!std::isfinite(ans)) return ans;
    if (g.nvars() > 0) {
      ans += dmvn(m_->included_coefficients(), g.select(beta_prior_->mu()),
                  g.select(beta_prior_->siginv()), true);
    }
    return ans;
  }

  void PSSS::draw() {
    impute_latent_data();
    if (allow_selection_) draw_gamma();
    draw_beta();
  }

  void PSSS::draw_beta() {
    const Selector &g(m_->coef().inc());
    Ominv_ = g.select(beta_prior_->siginv());
    wsp_ = Ominv_ * g.select(beta_prior_->mu()) + g.select(xtz());
    Ominv_ += g.select(xtx());
    beta_ = rmvn_suf_mt(rng(), Ominv_, wsp_);
    m_->set_Beta(g.expand(beta_));
  }

  void PSSS::limit_model_selection(uint n) {
    uint k = m_->xdim();
    max_nflips_ = n <= k ? n : k;
  }
  void PSSS::suppress_model_selection() { allow_selection_ = false; }
  void PSSS::allow_model_selection() { allow_selection_ = true; }
  uint PSSS::max_nflips() const { return max_nflips_; }

  bool PSSS::keep_flip(double logp_old, double logp_new) const {
    if (!std::isfinite(logp_new)) return false;
    double pflip = logit_inv(logp_new - logp_old);
    double u = runif_mt(rng(), 0, 1);
    return u < pflip ? true : false;
  }

  void PSSS::draw_gamma() {
    Selector inc = m_->coef().inc();
    uint nv = inc.nvars_possible();
    double logp = log_model_prob(inc);
    if (!std::isfinite(logp)) {
      ostringstream err;
      err << "ProbitSpikeSlab::draw_gamma did not start with "
          << "a legal configuration." << endl
          << "Selector vector:  " << inc << endl
          << "beta: " << m_->included_coefficients() << endl;
      report_error(err.str());
    }

    // do the sampling in random order
    std::vector<uint> flips = seq<uint>(0, nv - 1);
    std::shuffle(flips.begin(), flips.end(), std::default_random_engine());

    uint hi = std::min<uint>(nv, max_nflips());
    for (uint i = 0; i < hi; ++i) {
      uint I = flips[i];
      inc.flip(I);
      double logp_new = log_model_prob(inc);
      if (keep_flip(logp, logp_new))
        logp = logp_new;
      else
        inc.flip(I);  // reject the flip, so flip back
    }
    m_->coef().set_inc(inc);
  }

  double PSSS::log_model_prob(const Selector &g) {
    double num = gamma_prior_->logp(g);
    if (num == BOOM::negative_infinity()) return num;

    Ominv_ = g.select(beta_prior_->siginv());
    num += .5 * Ominv_.logdet();
    if (num == BOOM::negative_infinity()) return num;

    Vector mu = g.select(beta_prior_->mu());
    Vector Ominv_mu = Ominv_ * mu;
    num -= .5 * mu.dot(Ominv_mu);

    bool ok = true;
    iV_tilde_ = Ominv_ + g.select(xtx());
    Matrix L = iV_tilde_.chol(ok);
    if (!ok) return BOOM::negative_infinity();
    double denom = sum(log(L.diag()));  // = .5 log |Ominv_|

    Vector S = g.select(xtz()) + Ominv_mu;
    Lsolve_inplace(L, S);
    denom -= .5 * S.normsq();  // S.normsq =  beta_tilde ^T V_tilde beta_tilde

    return num - denom;
  }

}  // namespace BOOM
