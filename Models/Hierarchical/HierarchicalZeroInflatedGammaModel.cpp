// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#include "Models/Hierarchical/HierarchicalZeroInflatedGammaModel.hpp"
#include "Models/PosteriorSamplers/ZeroInflatedGammaPosteriorSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef HierarchicalZeroInflatedGammaModel HZIGM;
  }
  HierarchicalZeroInflatedGammaData::HierarchicalZeroInflatedGammaData(
      int n0, int n1, double sum, double sumlog)
      : number_of_zeros_(n0),
        number_of_positives_(n1),
        sum_(sum),
        sum_of_logs_of_positives_(sumlog) {}

  HierarchicalZeroInflatedGammaData *HierarchicalZeroInflatedGammaData::clone()
      const {
    return new HierarchicalZeroInflatedGammaData(*this);
  }

  std::ostream &HierarchicalZeroInflatedGammaData::display(std::ostream &out) const {
    out << number_of_zeros_ << " " << number_of_positives_ << " " << sum_ << " "
        << sum_of_logs_of_positives_;
    return out;
  }

  int HierarchicalZeroInflatedGammaData::number_of_zeros() const {
    return number_of_zeros_;
  }

  int HierarchicalZeroInflatedGammaData::number_of_positives() const {
    return number_of_positives_;
  }

  double HierarchicalZeroInflatedGammaData::sum() const { return sum_; }

  double HierarchicalZeroInflatedGammaData::sumlog() const {
    return sum_of_logs_of_positives_;
  }

  HZIGM::HierarchicalZeroInflatedGammaModel(
      const BOOM::Vector &number_of_zeros_per_group,
      const BOOM::Vector &number_of_positives_per_group,
      const BOOM::Vector &sum_of_positive_observations_per_group,
      const BOOM::Vector &sum_of_logs_of_positive_observations,
      BOOM::RNG &seeding_rng)
      : prior_for_mean_parameters_(new GammaModel),
        prior_for_shape_parameters_(new GammaModel),
        prior_for_positive_probability_(new BetaModel) {
    int number_of_groups = number_of_zeros_per_group.size();
    if (number_of_positives_per_group.size() != number_of_groups ||
        sum_of_positive_observations_per_group.size() != number_of_groups ||
        sum_of_logs_of_positive_observations.size() != number_of_groups) {
      report_error(
          "All data arguments to the HierarchicalZeroInflatedGammaModel "
          "constructor must be the same length");
    }

    data_models_.reserve(number_of_groups);
    for (int i = 0; i < number_of_groups; ++i) {
      NEW(ZeroInflatedGammaModel, data_model)
      (number_of_zeros_per_group[i], number_of_positives_per_group[i],
       sum_of_positive_observations_per_group[i],
       sum_of_logs_of_positive_observations[i]);
      NEW(ZeroInflatedGammaPosteriorSampler, sampler)
      (data_model.get(), prior_for_positive_probability_,
       prior_for_mean_parameters_, prior_for_shape_parameters_, seeding_rng);
      data_model->set_method(sampler);
      data_models_.push_back(data_model);
    }
    setup();
  }

  HZIGM::HierarchicalZeroInflatedGammaModel(const HZIGM &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        PriorPolicy(rhs),
        prior_for_mean_parameters_(rhs.prior_for_mean_parameters_->clone()),
        prior_for_shape_parameters_(rhs.prior_for_shape_parameters_->clone()),
        prior_for_positive_probability_(
            rhs.prior_for_positive_probability_->clone()) {
    data_models_.reserve(rhs.data_models_.size());
    for (int i = 0; i < rhs.data_models_.size(); ++i) {
      data_models_.push_back(rhs.data_models_[i]->clone());
    }
    setup();
  }

  HZIGM *HZIGM::clone() const { return new HZIGM(*this); }

  void HZIGM::clear_methods() {
    prior_for_mean_parameters_->clear_methods();
    prior_for_shape_parameters_->clear_methods();
    prior_for_positive_probability_->clear_methods();
    for (int group = 0; group < number_of_groups(); ++group) {
      data_models_[group]->clear_methods();
    }
    PriorPolicy::clear_methods();
  }

  void HZIGM::clear_data() {
    prior_for_positive_probability_->clear_data();
    prior_for_mean_parameters_->clear_data();
    prior_for_shape_parameters_->clear_data();
    data_models_.clear();
    ParamPolicy::clear();
    // After clearing all the models from the ParamPolicy, call
    // setup() to add the prior distributions back.
    setup();
  }

  void HZIGM::combine_data(const Model &rhs, bool just_suf) {
    try {
      const HZIGM &that(dynamic_cast<const HZIGM &>(rhs));
      data_models_.reserve(data_models_.size() + that.data_models_.size());
      for (int i = 0; i < that.data_models_.size(); ++i) {
        data_models_.push_back(that.data_models_[i]);
        ParamPolicy::add_model(that.data_models_[i]);
      }
    } catch (const std::exception &e) {
      ostringstream err;
      err << "Could not convert rhs to HierarchicalZeroInflatedGammaModel "
          << "in combine_data()." << endl
          << e.what();
      report_error(err.str());
    } catch (...) {
      report_error(
          "Unknown exception generated in HierarchicalZeroInflatedGammaModel::"
          "::combine_data.");
    }
  }

  void HZIGM::add_data(const Ptr<Data> &dp) {
    Ptr<HierarchicalZeroInflatedGammaData> d(
        dp.dcast<HierarchicalZeroInflatedGammaData>());
    NEW(BinomialModel, binomial)(.5);
    binomial->suf()->set(d->number_of_positives(),
                         d->number_of_positives() + d->number_of_zeros());
    NEW(GammaModel, gamma)(1, 1);
    gamma->suf()->set(d->sum(), d->sumlog(), d->number_of_positives());
    NEW(ZeroInflatedGammaModel, data_model)(binomial, gamma);
    data_models_.push_back(data_model);
    ParamPolicy::add_model(data_model);
  }

  int HZIGM::number_of_groups() const { return data_models_.size(); }

  BetaModel *HZIGM::prior_for_positive_probability() {
    return prior_for_positive_probability_.get();
  }

  GammaModel *HZIGM::prior_for_mean_parameters() {
    return prior_for_mean_parameters_.get();
  }

  GammaModel *HZIGM::prior_for_shape_parameters() {
    return prior_for_shape_parameters_.get();
  }

  ZeroInflatedGammaModel *HZIGM::data_model(int i) {
    return data_models_[i].get();
  }

  double HZIGM::positive_probability_prior_mean() const {
    return prior_for_positive_probability_->mean();
  }

  double HZIGM::positive_probability_prior_sample_size() const {
    return prior_for_positive_probability_->sample_size();
  }

  double HZIGM::mean_parameter_prior_mean() const {
    return prior_for_mean_parameters_->mean();
  }

  double HZIGM::mean_parameter_prior_shape() const {
    return prior_for_mean_parameters_->alpha();
  }

  double HZIGM::shape_parameter_prior_mean() const {
    return prior_for_shape_parameters_->mean();
  }

  double HZIGM::shape_parameter_prior_shape() const {
    return prior_for_shape_parameters_->alpha();
  }

  double HZIGM::prior_mean() const {
    return positive_probability_prior_mean() * mean_parameter_prior_mean();
  }

  HierarchicalZeroInflatedGammaData HZIGM::sim(int64_t n) const {
    const double positive_probability = prior_for_positive_probability_->sim();
    const double gamma_mean = prior_for_mean_parameters_->sim();
    const double gamma_shape = prior_for_shape_parameters_->sim();
    int number_of_positives = rbinom(n, positive_probability);
    int number_of_zeros = n - number_of_positives;
    double sum = 0.0;
    double sum_of_logs_of_positives = 0.0;
    for (int i = 0; i < number_of_positives; ++i) {
      double value = rgamma(gamma_shape, gamma_shape / gamma_mean);
      sum += value;
      sum_of_logs_of_positives += log(value);
    }
    return HierarchicalZeroInflatedGammaData(
        number_of_zeros, number_of_positives, sum, sum_of_logs_of_positives);
  }

  void HZIGM::setup() {
    ParamPolicy::add_model(prior_for_mean_parameters_);
    ParamPolicy::add_model(prior_for_shape_parameters_);
    ParamPolicy::add_model(prior_for_positive_probability_);
    for (int i = 0; i < data_models_.size(); ++i) {
      ParamPolicy::add_model(data_models_[i]);
    }
  }

}  // namespace BOOM
