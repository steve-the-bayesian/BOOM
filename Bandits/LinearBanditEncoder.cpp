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

#include "Bandits/LinearBanditEncoder.hpp"

namespace BOOM {

  ArmMap::ArmMap(const ExperimentStructure &xp)
      : xp_(xp)
  {
    FillArmValues_(xp);
  }

  void ArmMap::FillArmValues_(const ExperimentStructure &xp) {
    Configuration arm(xp.nlevels());
    arm_values_.clear();
    while(!arm.done()) {
      arm_values_.push_back(arm.levels());
      arm.next();
    }
  }

  std::vector<std::string> ArmMap::factor_level_names(int arm) const {
    std::vector<std::string> ans;
    for (int i = 0 ; i < xp_.nfactors(); ++i) {
      int level = arm_values_[arm][i];
      ans.push_back(xp_.full_level_name(i, level, ":"));
    }
    return ans;
  }
      
  
}  // namespace BOOM
