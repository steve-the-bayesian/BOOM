// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "Models/Glm/PosteriorSamplers/SpikeSlabSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/seq.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef SpikeSlabSampler SSS;
  }

  SSS::SpikeSlabSampler(GlmModel *model, const Ptr<MvnBase> &slab_prior,
                        const Ptr<VariableSelectionPrior> &spike_prior)
      : model_(model),
        slab_prior_(slab_prior),
        spike_prior_(spike_prior),
        max_flips_(-1),
        allow_model_selection_(true) {}

  // Performs one MCMC sweep along the provided set of inclusion indicators.
  void SSS::draw_inclusion_indicators(
      RNG &rng,
      Selector &inclusion_indicators,
      const WeightedRegSuf &suf,
      double sigsq) const {
    if (!allow_model_selection_) return;

    // Randomize the order in which the inclusion indicators are drawn.
    std::vector<int> indx =
        seq<int>(0, inclusion_indicators.nvars_possible() - 1);
    // I'd like to rely on std::random_shuffle for this, but I want
    // control over the random number generator.
    for (int i = indx.size() - 1; i > 0; --i) {
      int j = random_int_mt(rng, 0, i);
      if (j != i) {
        std::swap(indx[i], indx[j]);
      }
    }

    double logp = log_model_prob(inclusion_indicators, suf, sigsq);

    if (!std::isfinite(logp)) {
      spike_prior_->make_valid(inclusion_indicators);
      logp = log_model_prob(inclusion_indicators, suf, sigsq);
    }
    if (!std::isfinite(logp)) {
      ostringstream err;
      err << "SpikeSlabSampler did not start with a "
          << "legal configuration." << endl
          << "Selector vector:  " << inclusion_indicators << endl;
      if (model_) {
        err << "beta: " << model_->included_coefficients() << endl;
      }
      report_error(err.str());
    }

    uint n = inclusion_indicators.nvars_possible();
    if (max_flips_ > 0) n = std::min<int>(n, max_flips_);
    for (int i = 0; i < n; ++i) {
      logp = mcmc_one_flip(
          rng, inclusion_indicators, indx[i], logp, suf, sigsq);
    }
  }

  // Performs one MCMC sweep along the inclusion indicators for the
  // managed GlmModel.
  void SSS::draw_model_indicators(
      RNG &rng, const WeightedRegSuf &suf, double sigsq) {
    if (!allow_model_selection_) return;
    if (!model_) {
      report_error("No model was set.");
    }
    Selector inclusion_indicators = model_->coef().inc();
    draw_inclusion_indicators(rng, inclusion_indicators, suf, sigsq);
    model_->coef().set_inc(inclusion_indicators);
  }

  void SSS::draw_beta(RNG &rng, const WeightedRegSuf &suf, double sigsq) {
    if (!model_) {
      report_error("No model was set.");
    }
    Selector inclusion_indicators = model_->coef().inc();
    if (inclusion_indicators.nvars() == 0) {
      model_->drop_all();
      return;
    }
    Vector coefficients = model_->included_coefficients();
    draw_coefficients_given_inclusion(rng, coefficients, inclusion_indicators,
                                      suf, sigsq, false);
    // If model selection is turned off and some elements of beta
    // happen to be zero (because, e.inclusion_indicators., of a
    // failed MH step) we don't want the dimension of beta to change.
    model_->set_included_coefficients(coefficients);
  }

  void SSS::draw_coefficients_given_inclusion(
      RNG &rng, Vector &coefficients, const Selector &inclusion_indicators,
      const WeightedRegSuf &suf, double sigsq, bool full_set) const {
    if (inclusion_indicators.nvars() == 0) {
      if (full_set) {
        coefficients = 0.0;
      } else {
        coefficients.clear();
      }
      return;
    }
    SpdMatrix precision = inclusion_indicators.select(slab_prior_->siginv());
    Vector precision_mu =
        precision * inclusion_indicators.select(slab_prior_->mu());
    precision += inclusion_indicators.select(suf.xtx()) / sigsq;
    precision_mu += inclusion_indicators.select(suf.xty()) / sigsq;
    Vector mean = precision.solve(precision_mu);
    Vector draw = rmvn_ivar_mt(rng, mean, precision);
    if (full_set) {
      coefficients = inclusion_indicators.expand(draw);
    } else {
      coefficients = draw;
    }
  }

  double SSS::logpri() const {
    if (!model_) {
      report_error("No model was set.");
    }
    const Selector &inclusion_indicators(model_->coef().inc());
    double ans = spike_prior_->logp(inclusion_indicators);  // p(gamma)
    if (ans == BOOM::negative_infinity()) return ans;
    if (inclusion_indicators.nvars() > 0) {
      ans += dmvn(model_->included_coefficients(),
                  inclusion_indicators.select(slab_prior_->mu()),
                  inclusion_indicators.select(slab_prior_->siginv()), true);
    }
    return ans;
  }

  double SSS::log_prior(const GlmCoefs &beta) const {
    const Selector &inclusion_indicators(beta.inc());
    double ans = spike_prior_->logp(inclusion_indicators);  // p(gamma)
    if (ans == BOOM::negative_infinity()) return ans;
    if (inclusion_indicators.nvars() > 0) {
      ans += dmvn(beta.included_coefficients(),
                  inclusion_indicators.select(slab_prior_->mu()),
                  inclusion_indicators.select(slab_prior_->siginv()), true);
    }
    return ans;
  }

  void SSS::allow_model_selection(bool tf) { allow_model_selection_ = tf; }

  void SSS::limit_model_selection(int max_flips) { max_flips_ = max_flips; }

  double SSS::log_model_prob(const Selector &inclusion_indicators,
                             const WeightedRegSuf &suf, double sigsq) const {
    double numerator = spike_prior_->logp(inclusion_indicators);
    if (numerator == BOOM::negative_infinity() ||
        inclusion_indicators.nvars() == 0) {
      // If numerator == -infinity then it is in a zero support point
      // in the prior.  If inclusion_indicators.nvars()==0 then all
      // coefficients are zero because of the point mass.  The only
      // entries remaining in the likelihood are sums of squares of
      // y[i] that are independent of inclusion_indicators.  They need
      // to be omitted here because they are omitted in the non-empty
      // case below.
      return numerator;
    }
    SpdMatrix precision = inclusion_indicators.select(slab_prior_->siginv());
    numerator += .5 * precision.logdet();
    if (numerator == BOOM::negative_infinity()) return numerator;

    Vector mu = inclusion_indicators.select(slab_prior_->mu());
    Vector precision_mu = precision * mu;
    numerator -= .5 * mu.dot(precision_mu);

    bool ok = true;
    precision += inclusion_indicators.select(suf.xtx()) / sigsq;
    Matrix L = precision.chol(ok);
    if (!ok) return BOOM::negative_infinity();
    double denominator = sum(log(L.diag()));  // = .5 log |precision|
    Vector S = inclusion_indicators.select(suf.xty()) / sigsq + precision_mu;
    Lsolve_inplace(L, S);
    denominator -= .5 * S.normsq();
    // S.normsq =  beta_tilde ^T V_tilde beta_tilde
    return numerator - denominator;
  }

  double SSS::mcmc_one_flip(RNG &rng, Selector &mod, int which_var,
                            double logp_old, const WeightedRegSuf &suf,
                            double sigsq) const {
    mod.flip(which_var);
    double logp_new = log_model_prob(mod, suf, sigsq);
    double u = runif_mt(rng, 0, 1);
    if (log(u) > logp_new - logp_old) {
      mod.flip(which_var);  // reject draw
      return logp_old;
    }
    return logp_new;
  }

}  // namespace BOOM
