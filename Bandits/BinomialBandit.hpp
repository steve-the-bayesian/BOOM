#ifndef BOOM_BANDITS_BINOMIAL_BANDIT_HPP_
#define BOOM_BANDITS_BINOMIAL_BANDIT_HPP_
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

#include "Bandits/GenericBanditBase.hpp"

#include "Models/BinomialModel.hpp"
#include "Models/ParamTypes.hpp"
#include "Models/DataTypes.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  // A classroom example multi-armed bandit for success/failure outcomes.  Each
  // arm has an independent success probability.  Data for arms are communicated
  // by counts of successes and trials.
  class BinomialBandit : public GenericBanditBase {
   public:

    BinomialBandit(const std::vector<Ptr<BinomialModel>> &models);

    int number_of_arms() const override {
      return models_.size();
    }
    
    // Args:
    //   arm:  The arm for which the success probability is desired.
    //
    // Returns:
    //   The arm probability for the requested arm.
    double value(int arm) const {
      return models_[arm]->prob();
    }
    
    // Add the indicated number of successes and trials to the indicated arm.
    //
    // Args:
    //   arm:  The index of the targeted arm.
    //   num_successes: The number of incremental successes to add to the
    //     indicated arm.
    //   num_trials: The number of incremental trials to add to the indicated
    //     arm.
    //
    // Effetcts:
    //   The model indicated by 'arm' gets the number of successes and trials
    //   added.
    void observe_data(int arm, int num_successes, int num_trials);
    
    // Take ndraws samples from the posterior distribution given all observed
    // data.
    void update_posterior(int ndraws);
    
    const Vector &optimal_arm_probabilities() const;
    const Vector &value_remaining_distribution() const;

    // The MCMC draws of the arm-level probabilities.
    const Matrix &probability_draws() const {
      return probability_draws_;
    }
    
   private:
    std::vector<Ptr<BinomialModel>> models_;

    // The following data elements are populated by a call to update_posterior.
    Matrix probability_draws_;
    Vector optimal_arm_probabilities_;
    Vector value_remaining_distribution_;
    
  };   
  
}  // namespace BOOM

#endif  // BOOM_BANDITS_BINOMIAL_BANDIT_HPP_
