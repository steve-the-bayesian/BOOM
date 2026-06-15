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

#include "Bandits/LogitBanditExternalValue.hpp"
#include "Bandits/bandit_functions.hpp"
#include "stats/logit.hpp"

namespace BOOM {
  
  double LogitBanditExternalValue::value(
      int arm,
      const MixedMultivariateData &context) const {
    double prob = logit_inv(model()->predict(
        encoder()->encode_row(arm, context)));
    const std::vector<std::string> arm_values = encoder()->arm_values(arm);
    return value_function_(prob, arm_values);
  }

  Vector LogitBanditExternalValue::optimal_arm_probabilities(
      const MixedMultivariateData &context,
      RNG &rng) const {
    Matrix predictors = arm_predictors(context);
    Matrix value_draws = logit_inv(draws().multT(predictors));
    
    // value draws is [niter x arms]
    for (int arm = 0; arm < number_of_arms(); ++arm) {
      std::vector<std::string> arm_values = encoder()->arm_values(arm);
      for (int niter = 0; niter < draws().nrow(); ++niter) {
        double prob = value_draws(niter, arm);
        value_draws(niter, arm) = value_function_(prob, arm_values);
      }
    }
    return ComputeOptimalArmProbabilities(value_draws, rng);
  }

  Vector LogitBanditExternalValue::value_remaining_distribution(
      const MixedMultivariateData &context, RNG &rng) const {
    Matrix predictors = arm_predictors(context);
    Matrix value_draws = logit_inv(draws().multT(predictors));
    for (int arm = 0; arm < number_of_arms(); ++arm) {
      std::vector<std::string> arm_values = encoder()->arm_values(arm);
      for (int niter = 0; niter < value_draws.nrow(); ++niter) {
        double prob = value_draws(niter, arm);
        value_draws(niter, arm) = value_function_(prob, arm_values);
      }
    }
    return ValueRemainingDistribution(value_draws, rng);
  }
  
}  // namespace BOOM
