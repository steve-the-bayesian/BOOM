// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include "Models/Hierarchical/HierarchicalZeroInflatedPoissonModel.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  ZeroInflatedPoissonData::ZeroInflatedPoissonData(
      double nzero_trials, double npos_trials, double total_number_of_events)
      : suf_(nzero_trials, npos_trials, total_number_of_events) {}

  ZeroInflatedPoissonData::ZeroInflatedPoissonData(
      const ZeroInflatedPoissonSuf &suf)
      : suf_(suf) {}

  ZeroInflatedPoissonData::ZeroInflatedPoissonData(
      const ZeroInflatedPoissonData &rhs)
      : Data(rhs), suf_(rhs.suf_) {}

  ZeroInflatedPoissonData *ZeroInflatedPoissonData::clone() const {
    return new ZeroInflatedPoissonData(*this);
  }

  std::ostream &ZeroInflatedPoissonData::display(std::ostream &out) const {
    return suf_.print(out);
  }

  const ZeroInflatedPoissonSuf &ZeroInflatedPoissonData::suf() const {
    return suf_;
  }

  typedef HierarchicalZeroInflatedPoissonModel HZIP;

  HZIP::HierarchicalZeroInflatedPoissonModel(double lambda_prior_guess,
                                             double lambda_prior_sample_size,
                                             double zero_prob_prior_guess,
                                             double zero_prob_prior_sample_size)
      : prior_for_lambda_(
            new GammaModel(lambda_prior_guess * lambda_prior_sample_size,
                           lambda_prior_sample_size)),
        prior_for_zero_probability_(new BetaModel(
            zero_prob_prior_guess * zero_prob_prior_sample_size,
            (1 - zero_prob_prior_guess) * zero_prob_prior_sample_size)) {
    initialize();
  }

  HZIP::HierarchicalZeroInflatedPoissonModel(
      const Ptr<GammaModel> &prior_for_lambda,
      const Ptr<BetaModel> &prior_for_zero_probability)
      : prior_for_lambda_(prior_for_lambda),
        prior_for_zero_probability_(prior_for_zero_probability) {
    initialize();
  }

  HZIP::HierarchicalZeroInflatedPoissonModel(
      const BOOM::Vector &trials, const BOOM::Vector &events,
      const BOOM::Vector &number_of_zeros)
      : prior_for_lambda_(new GammaModel(1.0, 1.0)),
        prior_for_zero_probability_(new BetaModel(1.0, 1.0)) {
    initialize();
    if (trials.size() != events.size() ||
        trials.size() != number_of_zeros.size()) {
      report_error(
          "The trials, events, and number_of_zeros arguments must all "
          "have the same size in the "
          "HierarchicalZeroInflatedPoissonModel constructor.");
    }

    int ngroups = trials.size();
    for (int i = 0; i < ngroups; ++i) {
      ZeroInflatedPoissonModel *model = new ZeroInflatedPoissonModel;
      model->set_sufficient_statistics(ZeroInflatedPoissonSuf(
          number_of_zeros[i], trials[i] - number_of_zeros[i], events[i]));
      add_data_level_model(model);
    }
  }

  HZIP::HierarchicalZeroInflatedPoissonModel(
      const HierarchicalZeroInflatedPoissonModel &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        PriorPolicy(rhs),
        prior_for_lambda_(rhs.prior_for_lambda_->clone()),
        prior_for_zero_probability_(rhs.prior_for_zero_probability_->clone()) {
    initialize();
    for (int i = 0; i < rhs.data_level_models_.size(); ++i) {
      add_data_level_model(rhs.data_level_models_[i]->clone());
    }
  }

  HierarchicalZeroInflatedPoissonModel *HZIP::clone() const {
    return new HierarchicalZeroInflatedPoissonModel(*this);
  }

  void HZIP::add_data_level_model(const Ptr<ZeroInflatedPoissonModel> &model) {
    ParamPolicy::add_model(model);
    data_level_models_.push_back(model);
  }

  void HZIP::clear_data() {
    data_level_models_.clear();
    ParamPolicy::clear();
    initialize();
  }

  void HZIP::clear_client_data() {
    for (int i = 0; i < data_level_models_.size(); ++i) {
      data_level_models_[i]->clear_data();
    }
  }

  void HZIP::clear_methods() {
    prior_for_lambda_->clear_methods();
    prior_for_zero_probability_->clear_methods();
    for (int i = 0; i < data_level_models_.size(); ++i) {
      data_level_models_[i]->clear_methods();
    }
    PriorPolicy::clear_methods();
  }

  void HZIP::combine_data(const Model &rhs, bool just_suf) {
    const HZIP &rhs_model(dynamic_cast<const HZIP &>(rhs));
    for (int i = 0; i < rhs_model.number_of_groups(); ++i) {
      add_data_level_model(rhs_model.data_level_models_[i]);
    }
  }

  void HZIP::add_data(const Ptr<Data> &dp) {
    NEW(ZeroInflatedPoissonModel, model)();
    model->set_sufficient_statistics(
        dp.dcast<ZeroInflatedPoissonData>()->suf());
    add_data_level_model(model);
  }

  int HZIP::number_of_groups() const { return data_level_models_.size(); }

  ZeroInflatedPoissonModel *HZIP::data_model(int which_group) {
    return data_level_models_[which_group].get();
  }

  GammaModel *HZIP::prior_for_poisson_mean() { return prior_for_lambda_.get(); }

  BetaModel *HZIP::prior_for_zero_probability() {
    return prior_for_zero_probability_.get();
  }

  double HZIP::poisson_prior_mean() const {
    return prior_for_lambda_->alpha() / prior_for_lambda_->beta();
  }

  double HZIP::poisson_prior_sample_size() const {
    return prior_for_lambda_->beta();
  }

  double HZIP::zero_probability_prior_mean() const {
    return prior_for_zero_probability_->mean();
  }

  double HZIP::zero_probability_prior_sample_size() const {
    return prior_for_zero_probability_->sample_size();
  }

  ZeroInflatedPoissonData HZIP::sim(int64_t n) const {
    const double lambda = prior_for_lambda_->sim();
    const double zero_probability = prior_for_zero_probability_->sim();
    double number_of_zero_trials = rbinom(n, zero_probability);
    double number_of_positive_trials = n - number_of_zero_trials;
    double number_of_events = rpois(number_of_positive_trials * lambda);
    return ZeroInflatedPoissonData(number_of_zero_trials,
                                   number_of_positive_trials, number_of_events);
  }

  void HZIP::initialize() {
    ParamPolicy::add_model(prior_for_lambda_);
    ParamPolicy::add_model(prior_for_zero_probability_);
  }
}  // namespace BOOM
