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

#include "Models/Glm/PosteriorSamplers/PartRegSampler.hpp"
#include <algorithm>
#include <random>
#include "cpputil/math_utils.hpp"
#include "cpputil/seq.hpp"

namespace BOOM {

  typedef PartRegSampler PRS;

  PRS::PartRegSampler(uint Npart, const SpdMatrix &xtx, const Vector &xty,
                      double yty, const Vector &prior_mean,
                      const SpdMatrix &prior_ivar, double prior_df,
                      double prior_sigma_guess, double inc_prob)
      : rng_(seed_rng(GlobalRng::rng)),
        suf_(new NeRegSuf(xtx,
                          xty,
                          yty,
                          xtx(0, 0),                 // sample_size
                          xty(0) / xtx(0, 0),        // ybar
                          xtx.col(0) / xtx(0, 0))),  // xbar
        prior_mean_(prior_mean),
        prior_ivar_(prior_ivar),
        prior_df_(prior_df),
        prior_ss_(pow(prior_sigma_guess, 2) * prior_df),
        inc_probs_(prior_mean.size(), inc_prob),
        Nmcmc_(1) {
    indices_ = seq<uint>(0, Nvars() - 1);
    draw_initial_particles(Npart);
  }

  PRS::PartRegSampler(uint Npart, const SpdMatrix &xtx, const Vector &xty,
                      double yty, const Vector &prior_mean,
                      const SpdMatrix &prior_ivar, double prior_df,
                      double prior_sigma_guess, const Vector &inc_probs)
      : suf_(new NeRegSuf(xtx,
                          xty,
                          yty,
                          xtx(0, 0),                 // sample_size
                          xty[0] / xtx(0, 0),        // ybar
                          xtx.col(0) / xtx(0, 0))),  // xbar
        prior_mean_(prior_mean),
        prior_ivar_(prior_ivar),
        prior_df_(prior_df),
        prior_ss_(pow(prior_sigma_guess, 2) * prior_df),
        inc_probs_(inc_probs),
        Nmcmc_(1) {
    indices_ = seq<uint>(0, Nvars() - 1);
    draw_initial_particles(Npart);
  }

  void PRS::draw_initial_particles(uint N) {
    uint p = prior_mean_.size();
    models_.clear();
    models_.reserve(N);
    for (uint i = 0; i < N; ++i) {
      Selector mod(p, false);
      while (mod.nvars() == 0) {
        for (uint j = 0; j < p; ++j) {
          double u = runif_mt(rng_, 0, 1);
          if (u < inc_probs_[j]) mod.add(j);
        }
      }
      log_model_prob(mod);  // inserts mod into table.
      models_.push_back(mod);
    }
  }

  void PRS::draw_model_indicators(uint ntimes) {
    for (uint i = 0; i < ntimes; ++i) {
      resample_models();
      mcmc(Nmcmc_);
    }
  }

  uint PRS::Nvisited() const { return logpost_.size(); }

  struct mod_gt {
    bool operator()(const PRS::Mlike &lhs, const PRS::Mlike &rhs) const {
      if (lhs.second > rhs.second)
        return true;
      else if (lhs.second < rhs.second)
        return false;
      return lhs.first > rhs.first;
    }
  };

  inline std::vector<PRS::Mlike> &fix_probs(std::vector<PRS::Mlike> &ans) {
    Vector prob(ans.size());
    for (uint i = 0; i < ans.size(); ++i) prob[i] = ans[i].second;
    prob.normalize_logprob();
    for (uint i = 0; i < ans.size(); ++i) ans[i].second = prob[i];
    return ans;
  }

  std::vector<PRS::Mlike> PRS::all_models() const {
    std::vector<PRS::Mlike> ans(logpost_.begin(), logpost_.end());
    mod_gt mg;
    std::sort(ans.begin(), ans.end(), mg);
    return fix_probs(ans);
  }

  std::vector<PRS::Mlike> PRS::top_models(uint n) const {
    uint nmax = logpost_.size();
    if (n == 0 || n >= nmax) return all_models();

    std::vector<PRS::Mlike> ans(n);
    mod_gt mg;
    std::partial_sort_copy(logpost_.begin(), logpost_.end(), ans.begin(),
                           ans.end(), mg);
    return fix_probs(ans);
  }

  Vector PRS::marginal_inclusion_probs() const {
    uint N = models_.size();
    double prob = 1.0 / N;
    Vector ans(Nvars(), 0.0);
    for (uint i = 0; i < N; ++i) {
      ans += prob * models_[i].to_Vector();
    }
    return ans;
  }

  FrequencyDistribution PRS::model_sizes() const {
    uint Nmod = models_.size();
    std::vector<uint> sizes(Nmod);
    for (uint i = 0; i < Nmod; ++i) {
      sizes[i] = models_[i].nvars();
    }
    return FrequencyDistribution(sizes);
  }

  void PRS::draw_params() {
    uint Nmod = models_.size();
    betas_.resize(Nmod);
    sigsq_.resize(Nmod);

    double DF = suf_->n() + prior_df_;
    for (uint i = 0; i < Nmod; ++i) {
      Selector g = models_[i];
      SpdMatrix Ominv = g.select(prior_ivar_);
      double SS = set_reg_post_params(g, Ominv);
      double sigsq = 1.0 / rgamma(DF / 2, SS / 2);
      Vector beta = rmvn_ivar(beta_tilde_, iV_tilde_ / sigsq);
      betas_[i] = beta;
      sigsq_[i] = sigsq;
    }
  }

  void PRS::resample_models() {
    uint N = Nparticles();
    model_counts_.clear();

    std::vector<Selector> ans;
    ans.reserve(N);
    for (uint i = 0; i < N; ++i) {
      uint I = random_int(0, N - 1);
      Selector tmp(models_[I]);
      ans.push_back(tmp);
      ++model_counts_[tmp];
      double logprob = log_model_prob(tmp);
      weights_.push_back(logprob);
    }
    models_.swap(ans);
  }

  void PRS::set_number_of_mcmc_iterations(uint n) { Nmcmc_ = n; }

  void PRS::mcmc(uint niter) {
    uint N = Nparticles();
    for (uint i = 0; i < N; ++i) {
      for (uint it = 0; it < niter; ++it) {
        mcmc_all_vars(models_[i]);
      }
    }
  }

  void PRS::mcmc_one_flip(Selector &mod, uint which_var) {
    double logp_old = log_model_prob(mod);
    mod.flip(which_var);
    double logp_new = log_model_prob(mod);
    double u = runif_mt(rng_, 0, 1);
    if (log(u) > logp_new - logp_old) {
      mod.flip(which_var);  // reject draw
    }
  }

  void PRS::mcmc_all_vars(Selector &mod) {
    std::shuffle(indices_.begin(), indices_.end(),
                 std::default_random_engine());
    uint N = mod.nvars_possible();
    for (uint i = 0; i < N; ++i) {
      uint pos = indices_[i];
      mcmc_one_flip(mod, pos);
    }
  }

  void PRS::mcmc_one_var(Selector &mod) {
    uint N = mod.nvars_possible();
    uint pos = random_int(0, N - 1);
    mcmc_one_flip(mod, pos);
  }

  uint PRS::Nparticles() const { return models_.size(); }
  uint PRS::Nvars() const { return prior_mean_.size(); }
  const std::vector<Selector> &PRS::model_indicators() const { return models_; }

  double PRS::log_model_prob(const Selector &g) const {
    LogpostTable::iterator it = logpost_.find(g);
    if (it == logpost_.end()) {
      double ans = compute_log_model_prob(g);
      logpost_[g] = ans;
      return ans;
    }
    return it->second;
  }

  double PRS::logprior(const Selector &g) const {
    double ans(0);
    uint p = inc_probs_.size();
    for (uint i = 0; i < p; ++i) {
      double prob = g[i] ? inc_probs_[i] : 1 - inc_probs_[i];
      ans += log(prob);
    }
    return ans;
  }

  double PRS::empirical_prob(const Selector &g) const {
    ModelCount::const_iterator it = model_counts_.find(g);
    if (it == model_counts_.end()) return 0;
    double n = it->second;
    return n / models_.size();
  }

  double PRS::set_reg_post_params(const Selector &g,
                                  const SpdMatrix &Ominv) const {
    Vector b = g.select(prior_mean_);

    SpdMatrix xtx = suf_->xtx(g);
    Vector xty = suf_->xty(g);

    iV_tilde_ = Ominv + xtx;
    beta_tilde_ = Ominv * b + xty;
    beta_tilde_ = iV_tilde_.solve(beta_tilde_);

    double sse =
        xtx.Mdist(beta_tilde_) - 2 * beta_tilde_.dot(xty) + suf_->yty();
    double ss_beta = Ominv.Mdist(beta_tilde_, b);
    double SS = sse + ss_beta + prior_ss_;
    return SS;
  }

  double PRS::compute_log_model_prob(const Selector &g) const {
    if (g.nvars() == 0) return BOOM::negative_infinity();
    SpdMatrix Ominv = g.select(prior_ivar_);
    double SS = set_reg_post_params(g, Ominv);
    double DF = suf_->n() + prior_df_;
    double ans = logprior(g);
    ans += .5 * (Ominv.logdet() - iV_tilde_.logdet() - DF * log(SS));
    return ans;
  }
}  // namespace BOOM
