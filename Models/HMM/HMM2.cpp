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

#include "Models/HMM/HMM2.hpp"
#include "Models/HMM/HmmDataImputer.hpp"
#include "Models/HMM/HmmFilter.hpp"

#include "Models/EmMixtureComponent.hpp"
#include "Models/MarkovModel.hpp"

#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/string_utils.hpp"

#include "distributions.hpp"

#include <cmath>
#include <future>
#include <stdexcept>

namespace BOOM {

  namespace {
    using HMM = HiddenMarkovModel;
  }

  //======================================================================

  HMM::HiddenMarkovModel(const std::vector<Ptr<MixtureComponent>> &Mix,
                         const Ptr<MarkovModel> &Mark)
      : mark_(Mark),
        mix_(Mix),
        filter_(new HmmFilter(mix_, mark_)),
        loglike_(new UnivParams(0.0)),
        logpost_(new UnivParams(0.0)) {
    ParamPolicy::set_models(mix_.begin(), mix_.end());
    ParamPolicy::add_model(mark_);
  }

  HMM::HiddenMarkovModel(const HMM &rhs)
      : Model(rhs),
        DataPolicy(rhs),
        ParamPolicy(),
        PriorPolicy(rhs),
        mark_(rhs.mark_->clone()),
        mix_(rhs.state_space_size()),
        loglike_(new UnivParams(0.0)),
        logpost_(new UnivParams(0.0)) {
    for (uint i = 0; i < state_space_size(); ++i) {
      mix_[i] = rhs.mix_[i]->clone();
    }
    ParamPolicy::set_models(mix_.begin(), mix_.end());
    ParamPolicy::add_model(mark_);
    NEW(HmmFilter, f)(mix_, mark_);
    set_filter(f);
  }

  HMM *HMM::clone() const { return new HMM(*this); }

  void HMM::randomly_assign_data() {
    clear_client_data();
    uint S = state_space_size();
    Vector prob(S, 1.0 / S);
    for (uint s = 0; s < nseries(); ++s) {
      const DataSeriesType &ts(dat(s));
      uint n = ts.size();
      for (uint i = 0; i < n; ++i) {
        uint h = rmulti(prob);
        mix_[h]->add_data(ts[i]);
      }
    }
  }

  void HMM::initialize_params() {
    randomly_assign_data();
    uint S = state_space_size();
    Matrix Q(S, S, 1.0 / S);
    set_Q(Q);
    for (uint s = 0; s < S; ++s) mix_[s]->sample_posterior();
  }

  const Vector &HMM::pi0() const { return mark_->pi0(); }
  const Matrix &HMM::Q() const { return mark_->Q(); }

  void HMM::set_pi0(const Vector &pi0) { mark_->set_pi0(pi0); }
  void HMM::set_Q(const Matrix &Q) { mark_->set_Q(Q); }
  void HMM::set_filter(const Ptr<HmmFilter> &f) { filter_ = f; }

  void HMM::fix_pi0(const Vector &Pi0) { mark_->fix_pi0(Pi0); }
  void HMM::fix_pi0_stationary() { mark_->fix_pi0_stationary(); }
  bool HMM::pi0_fixed() const { return mark_->pi0_fixed(); }

  uint HMM::state_space_size() const { return mix_.size(); }

  // Returns the likelihood of the TimeSeries<Data> pointed at by dp.
  double HMM::pdf(const Ptr<Data> &dp, bool logscale) const {
    Ptr<DataSeriesType> dat = DAT(dp);
    double ans = filter_->loglike(*dat);
    return logscale ? ans : exp(ans);
  }

  // Clears the data for the hidden Markov chain and the mixture
  // components.  The data in the HMM's data policy remain untouched.
  void HMM::clear_client_data() {
    mark_->clear_data();
    uint S = state_space_size();
    for (uint s = 0; s < S; ++s) mix_[s]->clear_data();
  }

  void HMM::clear_prob_hist() {
    for (std::map<Ptr<Data>, Vector>::iterator it = prob_hist_.begin();
         it != prob_hist_.end(); ++it) {
      it->second = 0.0;
    }
  }

  std::vector<Ptr<MixtureComponent>> HMM::mixture_components() { return mix_; }
  Ptr<MixtureComponent> HMM::mixture_component(uint s) { return mix_[s]; }

  Ptr<MarkovModel> HMM::mark() { return mark_; }

  double HMM::saved_loglike() const { return loglike_->value(); }

  double HMM::loglike() const {
    uint ns = nseries();
    double ans = 0;
    for (uint series = 0; series < ns; ++series) {
      const DataSeriesType &ts(dat(series));
      ans += filter_->loglike(ts);
    }
    return ans;
  }

  void HMM::save_state_probs() {
    NEW(HmmSavePiFilter, filter)(mix_, mark_, prob_hist_);
    set_filter(filter);
  }

  Matrix HMM::report_state_probs(const DataSeriesType &ts) const {
    int n = ts.size();
    int S = state_space_size();
    Matrix ans(n, S);
    Ptr<HmmSavePiFilter> filter(filter_.dcast<HmmSavePiFilter>());
    if (!filter) {
      report_error(
          "filter could not be cast to SavePiFilter in "
          "HMM::report_state_probs");
    }
    for (int i = 0; i < n; ++i) {
      ans.row(i) = filter->state_probs(ts[i]);
    }
    return ans;
  }

  //======================================================================

  HMM_EM::HMM_EM(const std::vector<Ptr<EmMixtureComponent>> &Mix,
                 const Ptr<MarkovModel> &Mark)
      : HiddenMarkovModel(
            std::vector<Ptr<MixtureComponent>>(Mix.begin(), Mix.end()), Mark),
        mix_(Mix),
        eps(1e-5) {
    set_filter(new HmmEmFilter(mix_, mark()));
  }

  std::vector<Ptr<MixtureComponent>> HMM_EM::tomod(
      const std::vector<Ptr<EMC>> &Mix) const {
    std::vector<Ptr<MixtureComponent>> ans(Mix.begin(), Mix.end());
    return ans;
  }

  HMM_EM::HMM_EM(const HMM_EM &rhs)
      : Model(rhs),
        HiddenMarkovModel(rhs),
        mix_(rhs.mix_),
        eps(rhs.eps) {
    for (uint i = 0; i < mix_.size(); ++i) mix_[i] = rhs.mix_[i]->clone();
    set_mixture_components(mix_.begin(), mix_.end());
    set_filter(new HmmEmFilter(mix_, mark()));
  }

  HMM_EM *HMM_EM::clone() const { return new HMM_EM(*this); }

  void HMM_EM::find_mode(bool bayes) {
    double oldloglike = Estep(bayes);
    double crit = eps + 1;
    while (crit > eps) {
      Mstep(bayes);
      double loglike = Estep(bayes);
      crit = loglike - oldloglike;
      oldloglike = loglike;
    }
  }

  void HMM_EM::mle() { find_mode(false); }

  void HMM_EM::find_posterior_mode() { find_mode(true); }

  void HMM::set_loglike(double ell) { loglike_->set(ell); }

  void HMM::set_logpost(double ell) { logpost_->set(ell); }

  void HMM_EM::set_epsilon(double Eps) { eps = Eps; }

  void HMM_EM::initialize_params() {
    randomly_assign_data();
    uint S = state_space_size();
    for (uint h = 0; h < S; ++h) mix_[h]->mle();
    Matrix Q(S, S, 1.0 / S);
    set_Q(Q);
  }

  double HMM::impute_latent_data() {
    if (nthreads() > 0) {
      return impute_latent_data_with_threads();
    }

    clear_client_data();
    double ans = 0;
    uint ns = nseries();
    for (uint series = 0; series < ns; ++series) {
      const DataSeriesType &ts(dat(series));
      ans += filter_->fwd(ts);
      filter_->bkwd_sampling(ts);
    }
    set_loglike(ans);
    set_logpost(ans + logpri());
    return ans;
  }

  double HMM_EM::Estep(bool bayes) {
    clear_client_data();
    double ans = 0;
    uint ns = nseries();
    for (uint series = 0; series < ns; ++series) {
      const DataSeriesType &ts(dat(series));
      ans += filter_->fwd(ts);
      filter_->bkwd_smoothing(ts);
    }
    if (bayes) {
      ans += mark()->logpri();
      for (uint s = 0; s < state_space_size(); ++s) ans += mix_[s]->logpri();
    }
    return ans;
  }

  void HMM_EM::Mstep(bool bayes) {
    uint S = mix_.size();
    for (uint s = 0; s < S; ++s) {
      if (bayes) {
        mix_[s]->find_posterior_mode();
      } else {
        mix_[s]->mle();
      }
    }
    if (bayes) {
      mark()->find_posterior_mode();
    } else {
      mark()->mle();
    }
  }

  void HMM::set_nthreads(uint n) {
    thread_pool_.set_number_of_threads(n);
    workers_.clear();
    for (uint i = 0; i < n; ++i) {
      NEW(HmmDataImputer, imp)(this, i, n);
      workers_.push_back(imp);
    }
  }

  uint HMM::nthreads() const { return workers_.size(); }

  namespace {
    class HmmWorkWrapper {
     public:
      explicit HmmWorkWrapper(const Ptr<HmmDataImputer> &worker)
          : worker_(worker) {}
      void operator()() { worker_->impute_data(); }

     public:
      Ptr<HmmDataImputer> worker_;
    };
  }  // namespace

  double HMM::impute_latent_data_with_threads() {
    try {
      clear_client_data();

      std::vector<std::future<void>> futures;
      for (int i = 0; i < nthreads(); ++i) {
        workers_[i]->setup(this);
        futures.emplace_back(thread_pool_.submit(HmmWorkWrapper(workers_[i])));
      }

      uint S = state_space_size();
      double loglike = 0;
      for (uint i = 0; i < nthreads(); ++i) {
        futures[i].get();
        loglike += workers_[i]->loglike();
        mark_->combine_data(*workers_[i]->mark(), true);
        for (uint s = 0; s < S; ++s) {
          mix_[s]->combine_data(*workers_[i]->models(s), true);
        }
      }
      return loglike;
    } catch (const std::exception &e) {
      report_error(e.what());
    } catch (...) {
      report_error("HMM caught unknown exception during threaded imputation.");
    }
    return 0;
  }

}  // namespace BOOM
