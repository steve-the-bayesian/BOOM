/*
  Copyright (C) 2005-2022 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/Mixtures/BetaBinomialMixture.hpp"
#include "cpputil/lse.hpp"
#include "distributions.hpp"

namespace BOOM {

  AggregatedBinomialData::AggregatedBinomialData(
      int64_t trials, int64_t successes, int64_t count)
      : BinomialData(trials, successes),
        count_(count)
  {}

  AggregatedBinomialData * AggregatedBinomialData::clone() const {
    return new AggregatedBinomialData(*this);
  }

  BetaBinomialMixtureModel::BetaBinomialMixtureModel(
      const std::vector<Ptr<BetaBinomialModel>> &components,
      const Ptr<MultinomialModel> &mixing_weight_model)
      : components_(components),
        mixing_weight_model_(mixing_weight_model)
  {
    add_models_to_param_policy();
  }

  BetaBinomialMixtureModel::BetaBinomialMixtureModel(const BetaBinomialMixtureModel &rhs)
      : Model(rhs),
        LatentVariableModel(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        mixing_weight_model_(rhs.mixing_weight_model_->clone())
  {
    for (auto &el : rhs.components_) {
      components_.push_back(el->clone());
    }
    add_models_to_param_policy();
  }

  BetaBinomialMixtureModel * BetaBinomialMixtureModel::clone() const {
    return new BetaBinomialMixtureModel(*this);
  }

  void BetaBinomialMixtureModel::impute_latent_data(RNG &rng) {
    clear_component_data();
    const auto &data(dat());
    for (const Ptr<AggregatedBinomialData> &data_point : data) {
      impute_data_point(rng, data_point);
    }
  }

  void BetaBinomialMixtureModel::impute_data_point(
      RNG &rng,
      const Ptr<AggregatedBinomialData> &data_point) {

    Vector log_probs = mixing_weight_model_->logpi();
    for (size_t s = 0; s < components_.size(); ++s) {
      log_probs[s] += components_[s]->logp(data_point->trials(),
                                           data_point->successes());
    }
    Vector probs = log_probs.normalize_logprob();

    // Apportionment contains the number of observations allocated to each
    // mixture component.
    std::vector<int> apportionment(probs.size(), 0);
    rmultinom_mt(rng, data_point->count(), probs, apportionment);
    mixing_weight_model_->suf()->add_mixture_data(Vector(apportionment));
    for (int s = 0; s < apportionment.size(); ++s) {
      if (apportionment[s] > 0) {
        components_[s]->suf()->add_data(
            data_point->trials(), data_point->successes(), apportionment[s]);
      }
    }
  }

  void BetaBinomialMixtureModel::clear_component_data() {
    for (auto &component : components_) {
      component->clear_data();
    }
    mixing_weight_model_->clear_data();
  }

  double BetaBinomialMixtureModel::log_likelihood(
      const Vector &weights, const Matrix &ab) const {
    Vector log_weights = log(weights);
    int num_components = weights.size();

    double ans = 0;
    for (int i = 0; i < dat().size(); ++i) {
      auto data_point = dat()[i];
      Vector log_weighted_densities = log_weights;
      for (int s = 0; s < num_components; ++s) {
        log_weighted_densities[s] += BetaBinomialModel::logp(
            data_point->trials(),
            data_point->successes(),
            ab(s, 0),
            ab(s, 1));
      }
      ans += lse(log_weighted_densities) * data_point->count();
    }
    return ans;
  }

  void BetaBinomialMixtureModel::add_models_to_param_policy() {
    ParamPolicy::clear();
    ParamPolicy::add_model(mixing_weight_model_);
    for (const auto &el : components_) {
      ParamPolicy::add_model(el);
    }
  }

}  // namespace BOOM
