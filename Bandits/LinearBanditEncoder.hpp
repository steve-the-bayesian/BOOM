#ifndef BOOM_BANDITS_LINEAR_BANDIT_ENCODER_HPP_
#define BOOM_BANDITS_LINEAR_BANDIT_ENCODER_HPP_
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

#include "stats/Design.hpp"
#include "stats/Encoders.hpp"


namespace BOOM {

  class ExperimentEncoderAdapter {
    
    
  };

  // An ArmMap is a bijective mapping between arm definitions and factor values.
  class ArmMap {
   public:
    ArmMap(const ExperimentStructure &xp);

    int number_of_arms() const {
      return arm_values_.size();
    }

    int number_of_factors() const {
      return arm_values_[0].size();
    }

    const std::vector<std::string> &factor_names() const {
      return xp_.factor_names();
    }
    
    const std::vector<int> &integer_factor_levels(int arm) const {
      return arm_values_[arm];
    }

    std::vector<std::string> factor_level_names(int arm) const;
    
   private:
    void FillArmValues_(const ExperimentStructure &xp);
    ExperimentStructure xp_;

    // Each row of arm_values is the unique index of an arm.  Each column is a
    // factor.  Element (i, j) gives the (integer) index
    std::vector<std::vector<int>> arm_values_;
  };
  
  
  class LinearBanditEncoder {
   public:

    Vector encode(int arm, const MixedMultivariateData &context);
    
   private:
    Ptr<DatasetEncoder> context_encoder_;
    ExperimentStructure experiment_;
    
  };

  
}  // namespace BOOM

#endif  // BOOM_BANDITS_LINEAR_BANDIT_ENCODER_HPP_
