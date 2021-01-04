// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#include "Models/HMM/HealthStateModel.hpp"
#include "Models/HMM/hmm_tools.hpp"
#include "distributions.hpp"

namespace BOOM {

  HealthStateData::HealthStateData(const Ptr<Data> &data, int treatment)
      : value_(data),
        treatment_(treatment),
        initial_treatment_(treatment),
        final_treatment_fraction_(1.0) {}

  HealthStateData::HealthStateData(const HealthStateData &rhs)
      : Data(rhs),
        value_(rhs.value_->clone()),
        treatment_(rhs.treatment_),
        initial_treatment_(rhs.initial_treatment_),
        final_treatment_fraction_(rhs.final_treatment_fraction_) {}

  HealthStateData *HealthStateData::clone() const {
    return new HealthStateData(*this);
  }

  std::ostream &HealthStateData::display(std::ostream &out) const {
    out << *value_ << endl
        << "treatment = " << treatment_ << endl
        << "initial_treatment = " << initial_treatment_ << endl
        << "final_treatment_fraction = " << final_treatment_fraction_ << endl;
    return out;
  }

  int HealthStateData::treatment() const { return treatment_; }
  double HealthStateData::final_treatment_fraction() const {
    return final_treatment_fraction_;
  }
  int HealthStateData::initial_treatment() const { return initial_treatment_; }
  Ptr<Data> HealthStateData::shared_value() { return value_; }
  const Data *HealthStateData::value() const { return value_.get(); }
  //======================================================================
  void HealthStateModel::initialize_param_policy() {
    for (int i = 0; i < mix_.size(); ++i) ParamPolicy::add_model(mix_[i]);
    for (int i = 0; i < mark_.size(); ++i) ParamPolicy::add_model(mark_[i]);
  }

  HealthStateModel::HealthStateModel(
      const std::vector<Ptr<MixtureComponent> > &mix,
      const std::vector<Ptr<MarkovModel> > &mark)
      : mix_(mix), mark_(mark), one_(mix.size(), 1.0) {
    initialize_param_policy();
  }

  HealthStateModel::HealthStateModel(const HealthStateModel &rhs)
      : Model(rhs),
        DataPolicy(rhs),
        ParamPolicy(rhs),
        PriorPolicy(rhs),
        one_(rhs.one_) {
    for (int i = 0; i < rhs.mix_.size(); ++i) {
      Ptr<MixtureComponent> m(rhs.mix_[i]->clone());
      mix_.push_back(m);
    }

    for (int i = 0; i < rhs.mark_.size(); ++i) {
      Ptr<MarkovModel> m(rhs.mark_[i]->clone());
      mark_.push_back(m);
    }
    initialize_param_policy();
  }

  HealthStateModel *HealthStateModel::clone() const {
    return new HealthStateModel(*this);
  }

  double HealthStateModel::loglike() const {
    double ans = 0;
    for (int i = 0; i < nseries(); ++i) {
      ans += compute_loglike(dat(i));
    }
    return ans;
  }

  uint HealthStateModel::state_space_size() const { return mix_.size(); }

  uint HealthStateModel::ntreatments() const { return mark_.size(); }

  double HealthStateModel::impute_latent_data(RNG &rng) {
    double ans = 0;
    for (int i = 0; i < nseries(); ++i) {
      ans += fwd(dat(i));
      bkwd(rng, dat(i));
    }
    return ans;
  }

  double HealthStateModel::fwd(const TimeSeries<HealthStateData> &series) {
    int n = series.length();
    int S = state_space_size();
    logp_.resize(S);
    double loglike = initialize_fwd(series[0]);
    P_.resize(n);
    for (int i = 1; i < n; ++i) {
      fill_logp(series[i]);
      fill_logQ(series[i]);
      P_[i].resize(S, S);
      loglike += fwd_1(pi_, P_[i], logQ_, logp_, one_);
    }
    return loglike;
  }

  double HealthStateModel::compute_loglike(
      const TimeSeries<HealthStateData> &series) const {
    int n = series.length();
    int S = state_space_size();
    logp_.resize(S);
    double loglike = initialize_fwd(series[0]);
    Matrix P(S, S);
    for (int i = 0; i < n; ++i) {
      fill_logp(series[i], logp_);
      fill_logQ(series[i], logQ_);
      loglike += fwd_1(pi_, P, logQ_, logp_, one_);
    }
    return loglike;
  }

  // Iterates backward through a single time series of health state
  // data that has just been processed by fwd(series).  Simulates the
  // state associated with each observation, and allocates the
  // observation to the relevant mixture component.  Also allocates
  // state transitions ot the relevant hidden Markov chain.  In the
  // case of a treatment switch, a random draw of the treatment
  // mixture indicator determines the treatment group to which the
  // transition is assigned.
  void HealthStateModel::bkwd(RNG &rng,
                              const TimeSeries<HealthStateData> &series) {
    int n = series.length();
    uint s = rmulti(pi_);
    mix_[s]->add_data(series.back()->shared_value());

    for (int i = n - 1; i > 0; --i) {
      pi_ = P_[i].col(s);
      uint r = rmulti(pi_);
      mix_[r]->add_data(series[i - 1]->shared_value());
      uint which_treatment = sample_treatment(rng, series[i], r, s);
      mark_[which_treatment]->suf()->add_transition(r, s);
      s = r;
    }

    uint initial_treatment = series[0]->treatment();
    mark_[initial_treatment]->suf()->add_initial_value(s);
  }

  // Take a random draw of the treatment mixture indicator given
  // treatment transitions.
  int HealthStateModel::sample_treatment(RNG &rng,
                                         const Ptr<HealthStateData> &data,
                                         uint previous_state,
                                         uint current_state) {
    int last_treatment = data->treatment();
    double prior_last = data->final_treatment_fraction();
    if (prior_last >= 1.0) return last_treatment;

    int first_treatment = data->initial_treatment();

    double prior_first = 1 - prior_last;
    double post_first = prior_first * mark_[first_treatment]->Q()(
                                          previous_state, current_state);
    double post_last =
        prior_last * mark_[last_treatment]->Q()(previous_state, current_state);
    double first_prob = post_first / (post_first + post_last);

    if (runif_mt(rng) < first_prob) return first_treatment;
    return last_treatment;
  }

  // Fill logp with the density of each mixture component evaluated
  // at dp.
  void HealthStateModel::fill_logp(const Ptr<HealthStateData> &dp,
                                   Vector &logp) const {
    for (int s = 0; s < state_space_size(); ++s) {
      logp[s] = mix_[s]->pdf(dp->value(), true);
    }
  }

  void HealthStateModel::fill_logp(const Ptr<HealthStateData> &dp) {
    fill_logp(dp, logp_);
  }

  void HealthStateModel::fill_logQ(const Ptr<HealthStateData> &dp) {
    fill_logQ(dp, logQ_);
  }

  void HealthStateModel::fill_logQ(const Ptr<HealthStateData> &dp,
                                   Matrix &logQ) const {
    double alpha = dp->final_treatment_fraction();
    if (alpha >= 1.0) {
      logQ = log(mark_[dp->treatment()]->Q());
    } else {
      int then = dp->initial_treatment();
      int now = dp->treatment();
      logQ = log((1 - alpha) * mark_[then]->Q() + alpha * mark_[now]->Q());
    }
  }

  double HealthStateModel::initialize_fwd(const Ptr<HealthStateData> &d) const {
    fill_logp(d, logp_);
    int treatment = d->treatment();
    pi_ = log(mark_[treatment]->pi0()) + logp_;
    double m = max(pi_);
    pi_ = exp(pi_ - m);
    double nc = sum(pi_);
    double loglike = m + log(nc);
    pi_ /= nc;
    return loglike;
  }

}  // namespace BOOM
