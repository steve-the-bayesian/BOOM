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

#include "Models/HMM/HmmFilter.hpp"
#include "Models/HMM/hmm_tools.hpp"
#include "cpputil/math_utils.hpp"

#include "Models/EmMixtureComponent.hpp"
#include "Models/MarkovModel.hpp"
#include "Models/ModelTypes.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  void hmm_recursion_error(const Matrix &p, const Vector &Marg,
                           const Matrix &Tmat, const Vector &Wsp, uint i,
                           const Ptr<Data> &);
  void hmm_recursion_error(const Matrix &P, const Vector &marg,
                           const Matrix &tmat, const Vector &wsp, uint i,
                           const Ptr<Data> &dp) {
    std::string str;
    std::ostringstream s(str);
    s << "error in HMM recursion at step " << i << ":" << endl;
    s << "marg:" << endl << marg << endl;
    s << "P: " << endl << P << endl;
    s << "hmm.cpp:  Q = " << endl << tmat << endl;
    s << "hmm.cpp: p(data|state) = " << wsp << endl;
    s << "here is the observed data that caused the error: " << endl
      << *dp << endl;
    report_error(s.str());
  }

  HmmFilter::HmmFilter(const std::vector<Ptr<MixtureComponent>> &mix,
                       const Ptr<MarkovModel> &mark)
      : models_(mix),
        P(0),
        pi(mix.size()),
        logp(mix.size()),
        logpi(mix.size()),
        one(mix.size(), 1.0),
        logQ(mix.size(), mix.size()),
        markov_(mark) {}

  uint HmmFilter::state_space_size() const { return models_.size(); }

  double HmmFilter::initialize(const Data *dp) {
    uint S = state_space_size();
    pi = markov_->pi0();
    if (dp->missing())
      logp = 0;
    else
      for (uint s = 0; s < S; ++s) logp[s] = models_[s]->pdf(dp, true);
    pi = log(pi) + logp;
    double m = max(pi);
    pi = exp(pi - m);
    double nc = sum(pi);
    double loglike = m + log(nc);
    pi /= nc;
    return loglike;
  }

  double HmmFilter::fwd(const std::vector<Ptr<Data>> &dv) {
    logQ = log(markov_->Q());
    uint n = dv.size();
    uint S = state_space_size();
    if (logp.size() != S) logp.resize(S);
    if (P.size() < n) P.resize(n);
    double loglike = initialize(dv[0].get());
    for (uint i = 1; i < n; ++i) {
      if (dv[i]->missing())
        logp = 0;
      else
        for (uint s = 0; s < S; ++s)
          logp[s] = models_[s]->pdf(dv[i].get(), true);
      loglike += fwd_1(pi, P[i], logQ, logp, one);
    }
    return loglike;
  }
  //------------------------------------------------------------

  double HmmFilter::loglike(const std::vector<Ptr<Data>> &dv) {
    logQ = log(markov_->Q());
    pi = markov_->pi0();
    uint S = pi.size();
    uint n = dv.size();
    Matrix P(logQ);
    double ans = initialize(dv[0].get());
    for (uint i = 1; i < n; ++i) {
      if (dv[i]->missing())
        logp = 0;
      else
        for (uint s = 0; s < S; ++s)
          logp[s] = models_[s]->pdf(dv[i].get(), true);
      ans += fwd_1(pi, P, logQ, logp, one);
    }
    return ans;
  }
  //------------------------------------------------------------

  void HmmFilter::bkwd_sampling_mt(const std::vector<Ptr<Data>> &data, RNG &rng) {
    uint sample_size = data.size();
    std::vector<int> imputed_state(sample_size);
    // pi was already set by fwd.
    // So the following line would  break things when sample_size=1.
    //      pi = one * P.back();
    uint s = rmulti_mt(rng, pi);
    models_[s]->add_data(data.back());
    imputed_state.back() = s;
    for (int64_t i = sample_size - 1; i >= 0; --i) {
      pi = P[i].col(s);
      pi.normalize_prob();
      uint r = rmulti_mt(rng, pi);
      if (i > 0) {
        imputed_state[i - 1] = r;
      }
      models_[r]->add_data(data[i - 1]);
      markov_->suf()->add_transition(r, s);
      s = r;
    }
    markov_->suf()->add_initial_value(s);
    imputed_state_map_[data] = imputed_state;
  }

  //------------------------------------------------------------
  void HmmFilter::bkwd_sampling(const std::vector<Ptr<Data>> &dv) {
    uint n = dv.size();
    // pi was already set by fwd, so the following line would breaks
    // things when n=1.
    //      pi = one * P.back();
    uint s = rmulti(pi);     // last obs in state s
    allocate(dv.back(), s);  // last data point allocated

    for (uint i = n - 1; i != 0; --i) {  // start with s=h[i]
      pi = P[i].col(s);                  // compute r = h[i-1]
      uint r = rmulti(pi);
      allocate(dv[i - 1], r);
      markov_->suf()->add_transition(r, s);
      s = r;
    }
    markov_->suf()->add_initial_value(s);
    // in last step of loop i = 1, so s=h[0]
  }
  //----------------------------------------------------------------------
  void HmmFilter::allocate(const Ptr<Data> &dp, uint h) {
    models_[h]->add_data(dp);
  }
  Vector HmmFilter::state_probs(const Ptr<Data> &) const {
    Vector ans;
    report_error(
        "state_probs() cannot be called with this filter.  "
        "Use an HmmSavePiFilter instead.");
    return ans;
  }
  //----------------------------------------------------------------------
  std::vector<int> HmmFilter::imputed_state(
      const std::vector<Ptr<Data>> &data) const {
    const auto it = imputed_state_map_.find(data);
    if (it == imputed_state_map_.end()) {
      return std::vector<int>(0);
    } else {
      return it->second;
    }
  }
  
  //======================================================================
  HmmSavePiFilter::HmmSavePiFilter(const std::vector<Ptr<MixtureComponent>> &mv,
                                   const Ptr<MarkovModel> &mark,
                                   std::map<Ptr<Data>, Vector> &pi_hist)
      : HmmFilter(mv, mark), pi_hist_(pi_hist) {}
  //----------------------------------------------------------------------
  void HmmSavePiFilter::allocate(const Ptr<Data> &dp, uint h) {
    models_[h]->add_data(dp);
    Vector &v(pi_hist_[dp]);
    if (v.empty()) v.resize(pi.size());
    v += pi;
  }

  Vector HmmSavePiFilter::state_probs(const Ptr<Data> &dp) const {
    std::map<Ptr<Data>, Vector>::const_iterator it = pi_hist_.find(dp);
    if (it == pi_hist_.end()) {
      ostringstream err;
      err << "could not compute state probability distribution "
          << "for data point " << *dp << endl;
      report_error(err.str());
    }
    Vector ans(it->second);
    ans.normalize_prob();
    return ans;
  }
  //======================================================================

  // Note the conversion from EmMixtureComponent to MixtureComponent in the
  // constructor call for HmmFilter.
  HmmEmFilter::HmmEmFilter(const std::vector<Ptr<EmMixtureComponent>> &mix,
                           const Ptr<MarkovModel> &mark)
      : HmmFilter(std::vector<Ptr<MixtureComponent>>(mix.begin(), mix.end()),
                  mark),
        em_models_(mix) {}
  //------------------------------------------------------------
  void HmmEmFilter::bkwd_smoothing(const std::vector<Ptr<Data>> &dv) {
    // pi was set by fwd;
    uint n = dv.size();
    uint S = state_space_size();
    for (uint i = n - 1; i != 0; --i) {
      for (uint s = 0; s < S; ++s) {
        em_models_[s]->add_mixture_data(dv[i], pi[s]);
      }
      markov_->suf()->add_transition_distribution(P[i]);
      bkwd_1(pi, P[i], logp, one);
    }
    pi = P[1] * one;
    for (uint s = 0; s < S; ++s) {
      em_models_[s]->add_mixture_data(dv[0], pi[s]);
    }
    markov_->suf()->add_initial_distribution(pi);
  }

}  // namespace BOOM
