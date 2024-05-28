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
#include "Models/Glm/PosteriorSamplers/MLVS.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include <random>
#include "LinAlg/Selector.hpp"
#include "Models/Glm/ChoiceData.hpp"
#include "Models/Glm/MultinomialLogitModel.hpp"
#include "Models/Glm/PosteriorSamplers/MLVS_data_imputer.hpp"
#include "Models/MvnBase.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/seq.hpp"
#include "distributions.hpp"  // for rlexp,dnorm,rmvn
#include "stats/logit.hpp"

namespace BOOM {
  namespace {
    typedef MultinomialLogitModel MLM;
    typedef MultinomialLogit::CompleteDataSufficientStatistics MLVSS;
  }  // namespace
  using std::ostringstream;

  MLVS::MLVS(MLM *Mod, const Ptr<MvnBase> &slab,
             const Ptr<VariableSelectionPrior> &spike, uint nthreads,
             bool check_initial_condition, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(Mod),
        slab_(slab),
        spike_(spike),
        suf_(model_->beta_size(false)),
        log_sampling_probs_(model_->log_sampling_probs()),
        downsampling_(log_sampling_probs_.size() == model_->Nchoices()),
        select_(true),
        max_nflips_(model_->beta_size(false)) {
    if (check_initial_condition) {
      if (!std::isfinite(this->logpri())) {
        ostringstream err;
        err << "MLVS initialized with an a priori illegal value" << endl
            << "the initial Selector vector was: " << endl
            << model_->coef().inc() << endl
            << *spike_ << endl;
        report_error(err.str());
      }
    }
    set_number_of_workers(nthreads);
  }
  //______________________________________________________________________
  // public interface

  void MLVS::draw() {
    impute_latent_data();
    if (select_) draw_inclusion_vector();
    draw_beta();
  }

  void MLVS::clear_latent_data() { suf_.clear(); }

  Ptr<MlvsDataImputer> MLVS::create_worker(std::mutex &m) {
    return new MlvsDataImputer(suf_, m, model_, nullptr, rng());
  }

  void MLVS::assign_data_to_workers() {
    BOOM::assign_data_to_workers(model_->dat(), workers());
  }

  double MLVS::logpri() const {
    const Selector &g = model_->coef().inc();
    double ans = spike_->logp(g);
    if (ans == BOOM::negative_infinity()) return ans;
    if (g.nvars() > 0) {
      ans += dmvn(g.select(model_->beta()), g.select(slab_->mu()),
                  g.select(slab_->siginv()), true);
    }
    return ans;
  }

  //______________________________________________________________________
  // Drawing parameters

  void MLVS::draw_beta() {
    const Selector &inc(model_->coef().inc());
    uint N = inc.nvars_possible();
    Vector Beta(N, 0.);
    if (inc.nvars() > 0) {
      SpdMatrix Ominv = inc.select(slab_->siginv());
      SpdMatrix ivar = Ominv + inc.select(suf_.xtwx());
      Vector b = inc.select(suf_.xtwu()) + Ominv * inc.select(slab_->mu());
      b = ivar.solve(b);
      Vector beta = rmvn_ivar(b, ivar);
      uint n = b.size();
      for (uint i = 0; i < n; ++i) {
        uint I = inc.indx(i);
        Beta[I] = beta[i];
      }
    }
    model_->set_beta(Beta);
  }

  inline bool keep_flip(RNG &rng, double logp_old, double logp_new) {
    if (!std::isfinite(logp_new)) return false;
    double pflip = logit_inv(logp_new - logp_old);
    double u = runif_mt(rng, 0, 1);
    return u < pflip ? true : false;
  }

  void MLVS::draw_inclusion_vector() {
    Selector inc = model_->coef().inc();
    uint nv = inc.nvars_possible();
    double logp = log_model_prob(inc);
    if (!std::isfinite(logp)) {
      logp = log_model_prob(inc);
      ostringstream err;
      err << "MLVS did not start with a legal configuration." << endl
          << "Selector vector:  " << inc << endl
          << "beta: " << model_->beta() << endl;
      report_error(err.str());
    }

    std::vector<uint> flips = seq<uint>(0, nv - 1);
    std::shuffle(flips.begin(), flips.end(), std::default_random_engine());
    uint hi = std::min<uint>(nv, max_nflips());
    for (uint i = 0; i < hi; ++i) {
      uint I = flips[i];
      inc.flip(I);
      double logp_new = log_model_prob(inc);
      if (keep_flip(rng(), logp, logp_new))
        logp = logp_new;
      else
        inc.flip(I);  // reject the flip, so flip back
    }
    model_->coef().set_inc(inc);
  }

  void MLVS::suppress_model_selection() { select_ = false; }
  void MLVS::allow_model_selection() { select_ = true; }
  void MLVS::limit_model_selection(uint n) { max_nflips_ = n; }
  uint MLVS::max_nflips() const { return max_nflips_; }

  //______________________________________________________________________
  // computing probabilities

  double MLVS::log_model_prob(const Selector &g) {
    double num = spike_->logp(g);
    if (num == BOOM::negative_infinity()) return num;
    if (g.nvars() == 0) {
      num -= -.5 * suf_.weighted_sum_of_squares();
      return num;
    }

    Ominv = g.select(slab_->siginv());
    num += .5 * Ominv.logdet();
    if (num == BOOM::negative_infinity()) return num;

    Vector mu = g.select(slab_->mu());
    Vector Ominv_mu = Ominv * mu;
    num -= .5 * mu.dot(Ominv_mu);

    bool ok = true;
    iV_tilde_ = Ominv + g.select(suf_.xtwx());
    Matrix L = iV_tilde_.chol(ok);
    if (!ok) return BOOM::negative_infinity();
    double denom = sum(log(L.diag()));  // = .5 log |Ominv|

    Vector S = g.select(suf_.xtwu()) + Ominv_mu;
    Lsolve_inplace(L, S);
    denom -= .5 * S.normsq();  // S.normsq =  beta_tilde ^T V_tilde beta_tilde

    return num - denom;
  }

}  // namespace BOOM
