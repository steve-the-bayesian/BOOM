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

  class BinomialBandit : public GenericBanditBase {
   public:

    BinomialBandit(const std::vector<Ptr<BinomialModel>> &models);

    // Return the success probability for the requested arm.
    // Args:
    //   arm:  The arm for which the success probability is desired.
    //   model_params: If used, get the arm probabilities from here.  Must be
    //     VectorParams or VectorData.
    //   user_data:  Not used.
    //   rng: Not used.
    //
    // Returns:
    //   The arm probability for the requested arm.
    double Value(int arm,
                 const Params *model_params = nullptr,
                 const Data *user_data = nullptr,
                 const RNG *rng = nullptr) const override;

    // Add the indicated number of successes and trials to the indicated arm.
    //
    // Args:
    //   arm:  The index of the targeted arm.
    //   num_successes:  The number of incremental successes to add to the indicated arm.
    //   num_trials:  The number of incremental trials to add to the indicated arm.
    //
    // Effetcts:
    //   The model indicated by 'arm' gets the number of successes and trials
    //   added.
    void ObserveData(int arm, int num_successes, int num_trials);
    
    // Take one draw from the posterior distribution given all observed data.
    void UpdatePosterior();
    
  
   private:
  
    std::vector<Ptr<BinomialModel>> models_;
  };   
  
}  // namespace BOOM

#endif  // BOOM_BANDITS_BINOMIAL_BANDIT_HPP_
