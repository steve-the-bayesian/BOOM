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
namespace BOOM {

  AggregatedBinomialData::AggregatedBinomialData(
      int64_t trials, int64_t successes, int64_t count)
      : BinomialData(trials, successes),
        count_(count)
  {}

  BetaBinomialMixtureModel::BetaBinomialMixtureModel(
      const std::vector<Ptr<BetaBinomialModel>> &components,
      const Ptr<MultinomialModel> &mixing_weight_model)
      : components_(components),
        mixing_weight_model_(mixing_weight_model)
  {}

  void BetaBinomialMixtureModel::impute_latent_data(RNG &rng) {
    clear_component_data();
    const auto &data(dat());
    for (const Ptr<AggregatedBinomialData> &data_point : data) {
      Vector log_probs = mixing_weight_model_->logpi();
      for (size_t s = 0; s < components_.size(); ++s) {
        log_probs[s] += components_[s]->logp(data_point->trials(),
                                             data_point->successes());
      }
      Vector probs = log_probs.normalize_logprob();

    }

  }

  void BetaBinomialnomialMixtureModel::clear_component_data() {
    for (auto &component : components_) {
      component->clear_data();
    }
    mixing_weight_model_->clear_data();
  }


}  // namespace BOOM
