/*
  Copyright (C) 2005-2026 Steven L. Scott

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

#include "Bandits/BinomialBandit.hpp"
#include "Bandits/bandit_functions.hpp"

#include "Models/DataTypes.hpp"
#include "Models/ParamTypes.hpp"

#include "cpputil/report_error.hpp"

namespace BOOM {

  BinomialBandit::BinomialBandit(const std::vector<Ptr<BinomialModel>> &models) 
      : models_(models)
  {
    if (models_.empty()) {
      report_error("Vector of models was empty.");
    } else if (models_.size() == 1) {
      report_error("Vector of models only had a single element.");
    }

    for (int i = 0; i < models_.size(); ++i) {
      if (!models_[i]) {
        std::ostringstream msg;
        msg << "Element " << i << " of models vector is (nullptr).";
        report_error(msg.str());
      }
    }
  }

  double BinomialBandit::value(int arm,
                               const Params *model_params,
                               const Data *user_data,
                               const RNG *rng) const {
    if (model_params) {
      const VectorParams *arm_probs(
          dynamic_cast<const VectorParams *>(model_params));
      return (*arm_probs)[arm];
    } else {
      return models_[arm]->prob();
    }
  }
  
  void BinomialBandit::observe_data(int arm, int numSuccess, int numTrials) {
    models_[arm]->suf()->batch_update(numTrials, numSuccess);
  }
  
  void BinomialBandit::update_posterior(int ndraws) {
    probability_draws_.resize(ndraws, number_of_arms());
    for (int i = 0; i < ndraws; ++i) {
      for (int j = 0; j < number_of_arms(); ++j) { 
        models_[j]->sample_posterior();
        probability_draws_(i, j) = models_[j]->prob();
      }
    }

    optimal_arm_probabilities_ = ComputeOptimalArmProbabilities(
        probability_draws_);

    value_remaining_distribution_ = BOOM::ValueRemainingDistribution(
        probability_draws_);
  }

  const Vector &BinomialBandit::optimal_arm_probabilities() const {
    if (optimal_arm_probabilities_.empty()) {
      report_error("You must call update_posterior before calling "
                   "OptimalArmProbabilities.");
    }
    return optimal_arm_probabilities_;
  }

  const Vector &BinomialBandit::value_remaining_distribution() const {
    if (value_remaining_distribution_.empty()) {
      report_error("You must call update_posterior before calling "
                   "ValueRemainingDistribution.");
    }
    return value_remaining_distribution_;
  }
  
  
}  // namespace BOOM
