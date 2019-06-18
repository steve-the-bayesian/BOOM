// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2014 Steven L. Scott

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

#include "Models/Mixtures/ConditionalFiniteMixtureModel.hpp"
#include "cpputil/lse.hpp"  // lse
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    typedef ConditionalMixtureData CMD;
    typedef ConditionalFiniteMixtureModel CFMM;
  }  // namespace

  CMD::ConditionalMixtureData(
      const Ptr<Data> &data, const Ptr<VectorData> &mixture_category_predictors,
      int number_of_mixture_components, int known_mixture_component)
      : data_(data),
        mixture_category_data_(new ChoiceData(
            CategoricalData(
                known_mixture_component < 0 ? 0 : known_mixture_component,
                number_of_mixture_components),
            mixture_category_predictors, std::vector<Ptr<VectorData> >())),
        known_mixture_component_(known_mixture_component) {}

  CMD::ConditionalMixtureData(const CMD &rhs)
      : data_(rhs.data_->clone()),
        mixture_category_data_(rhs.mixture_category_data_->clone()),
        known_mixture_component_(rhs.known_mixture_component_) {}

  ConditionalMixtureData *ConditionalMixtureData::clone() const {
    return new CMD(*this);
  }

  std::ostream &CMD::display(std::ostream &out) const {
    data_->display(out) << endl;
    mixture_category_data_->display(out);
    return out;
  }

  const Data *ConditionalMixtureData::data() const { return data_.get(); }

  Ptr<Data> ConditionalMixtureData::shared_data() { return data_; }

  Ptr<ChoiceData> ConditionalMixtureData::shared_mixture_category_data() {
    return mixture_category_data_;
  }

  const ChoiceData *ConditionalMixtureData::mixture_category_data() const {
    return mixture_category_data_.get();
  }

  int ConditionalMixtureData::known_mixture_component() const {
    return known_mixture_component_;
  }

  void ConditionalMixtureData::set_mixture_component(int mixture_indicator) {
    if (known_mixture_component_ > 0 &&
        mixture_indicator != known_mixture_component_) {
      ostringstream err;
      err << "A data point knownn to come from mixture component "
          << known_mixture_component_ << " is being set to mixture component "
          << mixture_indicator << ".";
      report_error(err.str());
    }
    mixture_category_data_->set_y(mixture_indicator);
  }

  //======================================================================
  CFMM::ConditionalFiniteMixtureModel(
      const std::vector<Ptr<MixtureComponent> > &mixture_components,
      const Ptr<MultinomialLogitModel> &mixing_distribution)
      : mixture_components_(mixture_components),
        mixing_distribution_(mixing_distribution) {
    if (mixing_distribution_->Nchoices() != mixture_components.size()) {
      ostringstream err;
      err << "The number of mixture components: " << mixture_components.size()
          << " did not match the dimension of the mixing distribution: "
          << mixing_distribution_->Nchoices()
          << " in ConditionalFiniteMixtureModel constructor." << endl;
      report_error(err.str());
    }
  }

  CFMM *CFMM::clone() const { return new CFMM(*this); }

  void ConditionalFiniteMixtureModel::clear_component_data() {
    for (int s = 0; s < number_of_mixture_components(); ++s) {
      mixture_component(s)->clear_data();
    }
  }

  void ConditionalFiniteMixtureModel::clear_data() {
    clear_component_data();
    data_.clear();
    mixing_distribution_->clear_data();
  }

  void CFMM::add_data(const Ptr<Data> &abstract_data_point) {
    add_conditional_mixture_data(
        abstract_data_point.dcast<ConditionalMixtureData>());
  }

  void CFMM::add_conditional_mixture_data(
      const Ptr<ConditionalMixtureData> &data_point) {
    data_.push_back(data_point);
    mixing_distribution_->add_data(data_point->shared_mixture_category_data());
  }

  std::vector<Ptr<ConditionalMixtureData> > &CFMM::dat() { return data_; }

  const std::vector<Ptr<ConditionalMixtureData> > &CFMM::dat() const {
    return data_;
  }

  void CFMM::combine_data(const Model &abstract_other, bool) {
    const CFMM &other(dynamic_cast<const CFMM &>(abstract_other));
    for (int i = 0; i < other.data_.size(); ++i) {
      add_conditional_mixture_data(other.data_[i]);
    }
  }

  void ConditionalFiniteMixtureModel::impute_latent_data(RNG &rng) {
    clear_component_data();
    int n = data_.size();
    int S = number_of_mixture_components();
    wsp_.resize(S);
    class_membership_probabilities_.resize(n, S);
    for (int i = 0; i < n; ++i) {
      ConditionalMixtureData &data_point(*data_[i]);
      const ChoiceData &mixture_category_data(
          *(data_point.mixture_category_data()));
      if (data_point.missing()) {
        // Need to handle missing data differently than in
        // FiniteMixtureModel because no predictors are at hand to
        // give prior probabilities.
        //
        // Ignore missing data.
      } else if (data_point.known_mixture_component() > 0) {
        // This code branch deals with the case where we know which
        // mixture component produced the given data point.
        int source = data_point.known_mixture_component();
        last_loglike_ +=
            mixture_component(source)->pdf(data_point.data(), true);
        class_membership_probabilities_.row(i) = 0.0;
        class_membership_probabilities_.row(i)[source] = 1.0;
        set_mixture_component_for_observation(i, source);
      } else {
        for (int s = 0; s < S; ++s) {
          wsp_[s] =
              mixing_distribution_->predict_subject(mixture_category_data, s) +
              mixture_component(s)->pdf(data_point.data(), true);
        }
        last_loglike_ += lse(wsp_);
        wsp_.normalize_logprob();
        class_membership_probabilities_.row(i) = wsp_;
        int mixture_indicator = rmulti_mt(rng, wsp_);
        set_mixture_component_for_observation(i, mixture_indicator);
        mixture_component(mixture_indicator)
            ->add_data(data_point.shared_data());
      }
    }
  }

  int ConditionalFiniteMixtureModel::number_of_mixture_components() const {
    return mixture_components_.size();
  }

  MultinomialLogitModel *CFMM::mixing_distribution() {
    return mixing_distribution_.get();
  }

  const MultinomialLogitModel *CFMM::mixing_distribution() const {
    return mixing_distribution_.get();
  }

  MixtureComponent *CFMM::mixture_component(int s) {
    return mixture_components_[s].get();
  }

  const MixtureComponent *CFMM::mixture_component(int s) const {
    return mixture_components_[s].get();
  }

  double ConditionalFiniteMixtureModel::last_loglike() const {
    return last_loglike_;
  }

  double CFMM::pdf(const Data *dp, bool logscale) const {
    const ConditionalMixtureData *data(
        dynamic_cast<const ConditionalMixtureData *>(dp));
    double ans = logp(*data);
    return logscale ? ans : exp(ans);
  }

  double CFMM::logp(const ConditionalMixtureData &data_point) const {
    for (int s = 0; s < number_of_mixture_components(); ++s) {
      wsp_[s] = mixing_distribution_->predict_subject(
                    *(data_point.mixture_category_data()), s) +
                mixture_components_[s]->pdf(data_point.data(), true);
    }
    return lse(wsp_);
  }

  void CFMM::set_mixture_component_for_observation(int i,
                                                   int mixture_component) {
    data_[i]->set_mixture_component(mixture_component);
  }

}  // namespace BOOM
