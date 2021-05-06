// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include "Models/Glm/PosteriorSamplers/BinomialLogitSpikeSlabSampler.hpp"
#include "LinAlg/Cholesky.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/seq.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef BinomialLogitSpikeSlabSampler BLSSS;

  BLSSS::BinomialLogitSpikeSlabSampler(BinomialLogitModel *model,
                                       const Ptr<MvnBase> &slab,
                                       const Ptr<VariableSelectionPrior> &spike,
                                       int clt_threshold,
                                       RNG &seeding_rng)
      : BinomialLogitAuxmixSampler(model, slab, clt_threshold, seeding_rng),
        model_(model),
        slab_(check_slab_dimension(slab)),
        spike_(check_spike_dimension(spike)),
        allow_model_selection_(true),
        max_flips_(-1),
        posterior_mode_found_(false),
        log_posterior_at_mode_(negative_infinity()) {}

  BLSSS * BLSSS::clone_to_new_host(Model *new_host) const {
    return new BLSSS(dynamic_cast<BinomialLogitModel *>(new_host),
                     slab_->clone(),
                     spike_->clone(),
                     clt_threshold(),
                     rng());
  }

  void BLSSS::draw() {
    impute_latent_data();
    if (allow_model_selection_) draw_model_indicators();
    draw_beta();
  }

  void BLSSS::draw_beta() {
    Selector g = model_->coef().inc();
    if (g.nvars() == 0) {
      model_->drop_all();
      return;
    }
    SpdMatrix precision = g.select(slab_->siginv());
    Vector scaled_mean = precision * g.select(slab_->mu());
    precision += g.select(suf().xtx());
    Cholesky precision_cholesky_factor(precision);
    scaled_mean += g.select(suf().xty());
    Vector posterior_mean = precision_cholesky_factor.solve(scaled_mean);
    Vector beta = rmvn_precision_upper_cholesky_mt(
        rng(), posterior_mean, precision_cholesky_factor.getLT());

    // If model selection is turned off and some elements of beta
    // happen to be zero (because, e.g., of a failed MH step) we don't
    // want the dimension of beta to change.
    model_->set_included_coefficients(beta);
  }

  double BLSSS::logpri() const {
    const Selector &g(model_->coef().inc());
    double ans = spike_->logp(g);  // p(gamma)
    if (ans == BOOM::negative_infinity()) return ans;
    if (g.nvars() > 0) {
      ans += dmvn(model_->included_coefficients(), g.select(slab_->mu()),
                  g.select(slab_->siginv()), true);
    }
    return ans;
  }

  double BLSSS::log_model_prob(const Selector &g) const {
    // borrowed from MLVS.cpp
    double num = spike_->logp(g);
    if (num == BOOM::negative_infinity() || g.nvars() == 0) {
      // If num == -infinity then it is in a zero support point in the
      // prior.  If g.nvars()==0 then all coefficients are zero
      // because of the point mass.  The only entries remaining in the
      // likelihood are sums of squares of y[i] that are independent
      // of g.  They need to be omitted here because they are omitted
      // in the non-empty case below.
      return num;
    }
    SpdMatrix ivar = g.select(slab_->siginv());
    num += .5 * ivar.logdet();
    if (num == BOOM::negative_infinity()) return num;

    Vector mu = g.select(slab_->mu());
    Vector ivar_mu = ivar * mu;
    num -= .5 * mu.dot(ivar_mu);

    bool ok = true;
    ivar += g.select(suf().xtx());
    Matrix L = ivar.chol(ok);
    if (!ok) return BOOM::negative_infinity();
    double denom = sum(log(L.diag()));  // = .5 log |ivar|
    Vector S = g.select(suf().xty()) + ivar_mu;
    Lsolve_inplace(L, S);
    denom -= .5 * S.normsq();  // S.normsq =  beta_tilde ^T V_tilde beta_tilde
    return num - denom;
  }

  void BLSSS::allow_model_selection(bool tf) { allow_model_selection_ = tf; }

  void BLSSS::limit_model_selection(int max_flips) { max_flips_ = max_flips; }

  class BinomialLogitUnNormalizedLogPosterior : public d2TargetFun {
   public:
    BinomialLogitUnNormalizedLogPosterior(BinomialLogitModel *model,
                                          MvnBase *prior)
        : model_(model), prior_(prior) {}

    double operator()(const Vector &included_coefficients, Vector &gradient,
                      Matrix &hessian, uint nd) const {
      double ans = prior_->logp_given_inclusion(
          included_coefficients, nd > 0 ? &gradient : nullptr,
          nd > 1 ? &hessian : nullptr, model_->coef().inc(), true);
      ans += model_->log_likelihood(included_coefficients,
                                    nd > 0 ? &gradient : nullptr,
                                    nd > 1 ? &hessian : nullptr, false);
      return ans;
    }
    using d2TargetFun::operator();

   private:
    BinomialLogitModel *model_;
    MvnBase *prior_;
    Selector inc_;
  };

  void BLSSS::find_posterior_mode(double epsilon) {
    posterior_mode_found_ = false;
    log_posterior_at_mode_ = negative_infinity();
    try {
      BinomialLogitUnNormalizedLogPosterior logpost(model_, slab_.get());
      Vector beta(model_->included_coefficients());
      int dim = beta.size();
      if (dim == 0) {
        return;
        // TODO: This logic prohibits an empty model.  Better to return
        // the actual value of the un-normalized posterior, which in
        // this case would just be the likelihood portion.
      } else {
        Vector gradient(dim);
        Matrix hessian(dim, dim);
        std::string error_message;
        bool ok = max_nd2_careful(
            beta, gradient, hessian, log_posterior_at_mode_, Target(logpost),
            dTarget(logpost), d2Target(logpost), epsilon, error_message);
        if (ok) {
          posterior_mode_found_ = true;
          model_->set_included_coefficients(beta);
          return;
        } else {
          log_posterior_at_mode_ = negative_infinity();
          return;
        }
      }
    } catch (...) {
      return;
    }
  }

  void BLSSS::draw_model_indicators() {
    Selector g = model_->coef().inc();
    std::vector<int> indx = seq<int>(0, g.nvars_possible() - 1);
    // I'd like to rely on std::random_shuffle for this, but I want
    // control over the random number generator.
    for (int i = 0; i < indx.size(); ++i) {
      int j = random_int_mt(rng(), 0, indx.size() - 1);
      std::swap(indx[i], indx[j]);
    }

    double logp = log_model_prob(g);

    if (!std::isfinite(logp)) {
      spike_->make_valid(g);
      logp = log_model_prob(g);
    }
    if (!std::isfinite(logp)) {
      ostringstream err;
      err << "BinomialLogitSpikeSlabSampler did not start with a "
          << "legal configuration." << endl
          << "Selector vector:  " << g << endl
          << "beta: " << model_->included_coefficients() << endl;
      report_error(err.str());
    }

    uint n = g.nvars_possible();
    if (max_flips_ > 0) n = std::min<int>(n, max_flips_);
    for (uint i = 0; i < n; ++i) {
      logp = mcmc_one_flip(g, indx[i], logp);
    }
    model_->coef().set_inc(g);
  }

  double BLSSS::mcmc_one_flip(Selector &mod, uint which_var, double logp_old) {
    mod.flip(which_var);
    double logp_new = log_model_prob(mod);
    double u = runif_mt(rng(), 0, 1);
    if (log(u) > logp_new - logp_old) {
      mod.flip(which_var);  // reject draw
      return logp_old;
    }
    return logp_new;
  }

  const Ptr<MvnBase> &BLSSS::check_slab_dimension(const Ptr<MvnBase> &slab) {
    if (slab->dim() != model_->xdim()) {
      report_error("Slab does not match model dimension.");
    }
    return slab;
  }

  const Ptr<VariableSelectionPrior> &BLSSS::check_spike_dimension(
      const Ptr<VariableSelectionPrior> &spike) {
    if (spike->potential_nvars() != model_->xdim()) {
      report_error("Spike does not match model dimension.");
    }
    return spike;
  }

}  // namespace BOOM
