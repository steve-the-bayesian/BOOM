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

  // Print a useful error message when an error is found in the HMM recursions.
  inline void hmm_recursion_error(const Matrix &P,
                                  const Vector &marg,
                                  const Matrix &tmat,
                                  const Vector &wsp,
                                  uint i,
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
        markov_(mark),
        joint_distributions_(0),
        marginal_distribution_(mix.size()),
        logp(mix.size()),
        logpi(mix.size()),
        one(mix.size(), 1.0),
        logQ(mix.size(), mix.size())
  {}

  uint HmmFilter::state_space_size() const { return models_.size(); }

  double HmmFilter::initialize(const Data *dp) {
    uint S = state_space_size();
    marginal_distribution_ = markov_->pi0();
    if (dp->missing()) {
      logp = 0;
    } else {
      for (uint s = 0; s < S; ++s) {
        logp[s] = models_[s]->pdf(dp, true);
      }
    }
    marginal_distribution_ = log(marginal_distribution_) + logp;
    double m = max(marginal_distribution_);
    marginal_distribution_ = exp(marginal_distribution_ - m);
    double nc = sum(marginal_distribution_);
    double loglike = m + log(nc);
    marginal_distribution_ /= nc;
    return loglike;
  }

  double HmmFilter::fwd(const std::vector<Ptr<Data>> &dv) {
    logQ = log(markov_->Q());
    uint n = dv.size();
    uint S = state_space_size();
    if (logp.size() != S) {
      logp.resize(S);
    }
    if (joint_distributions_.size() < n) {
      joint_distributions_.resize(n);
    }
    double loglike = initialize(dv[0].get());
    for (uint i = 1; i < n; ++i) {
      if (dv[i]->missing())
        logp = 0;
      else
        for (uint s = 0; s < S; ++s)
          logp[s] = models_[s]->pdf(dv[i].get(), true);
      loglike += fwd_1(marginal_distribution_,
                       joint_distributions_[i],
                       logQ,
                       logp,
                       one);
    }
    return loglike;
  }
  //------------------------------------------------------------

  double HmmFilter::loglike(const std::vector<Ptr<Data>> &dv) {
    logQ = log(markov_->Q());
    marginal_distribution_ = markov_->pi0();
    uint S = marginal_distribution_.size();
    uint n = dv.size();
    Matrix P(logQ);
    double ans = initialize(dv[0].get());
    for (uint i = 1; i < n; ++i) {
      if (dv[i]->missing())
        logp = 0;
      else
        for (uint s = 0; s < S; ++s)
          logp[s] = models_[s]->pdf(dv[i].get(), true);
      ans += fwd_1(marginal_distribution_, P, logQ, logp, one);
    }
    return ans;
  }
  //------------------------------------------------------------

  void HmmFilter::bkwd_sampling_mt(const std::vector<Ptr<Data>> &data, RNG &rng) {
    uint sample_size = data.size();
    imputed_states_.resize(sample_size);
    // pi was already set by fwd.
    // So the following line would  break things when sample_size=1.
    //      pi = one * P.back();
    uint s = rmulti_mt(rng, marginal_distribution_);
    models_[s]->add_data(data.back());
    imputed_states_.back() = s;
    for (int64_t i = sample_size - 1; i > 0; --i) {
      marginal_distribution_ = joint_distributions_[i].col(s);
      marginal_distribution_.normalize_prob();
      uint r = rmulti_mt(rng, marginal_distribution_);
      if (i > 0) {
        imputed_states_[i - 1] = r;
      }
      models_[r]->add_data(data[i - 1]);
      markov_->suf()->add_transition(r, s);
      s = r;
    }
    markov_->suf()->add_initial_value(s);
  }

  //------------------------------------------------------------
  // void HmmFilter::bkwd_sampling(const std::vector<Ptr<Data>> &dv) {
  //   uint n = dv.size();
  //   // pi was already set by fwd, so the following line would breaks
  //   // things when n=1.
  //   //      pi = one * P.back();
  //   uint s = rmulti(pi);     // last obs in state s
  //   allocate(dv.back(), s);  // last data point allocated

  //   for (uint i = n - 1; i != 0; --i) {  // start with s=h[i]
  //     pi = joint_distributions_[i].col(s);                  // compute r = h[i-1]
  //     uint r = rmulti(pi);
  //     allocate(dv[i - 1], r);
  //     markov_->suf()->add_transition(r, s);
  //     s = r;
  //   }
  //   markov_->suf()->add_initial_value(s);
  //   // in last step of loop i = 1, so s=h[0]
  // }
  //----------------------------------------------------------------------

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
        em_models_[s]->add_mixture_data(dv[i], marginal_distribution_[s]);
      }
      markov_->suf()->add_transition_distribution(joint_distributions_[i]);
      bkwd_1(marginal_distribution_, joint_distributions_[i], logp, one);
    }
    marginal_distribution_ = joint_distributions_[1] * one;
    for (uint s = 0; s < S; ++s) {
      em_models_[s]->add_mixture_data(dv[0], marginal_distribution_[s]);
    }
    markov_->suf()->add_initial_distribution(marginal_distribution_);
  }

}  // namespace BOOM
