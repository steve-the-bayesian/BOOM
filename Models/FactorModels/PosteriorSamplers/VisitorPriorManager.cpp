/*
  Copyright (C) 2005-2024 Steven L. Scott

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

#include "Models/FactorModels/PosteriorSamplers/VisitorPriorManager.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/data_checking.hpp"

namespace BOOM {

  void VisitorPriorManager::set_prior_class_probabilities(
      const std::string &visitor_id,
      const Vector &probs) {
    check_probabilities(probs, true);
    prior_class_probabilities_[visitor_id] = probs;
  }

  const Vector &VisitorPriorManager::prior_class_probabilities(
      const std::string &visitor_id) const {
    auto it = prior_class_probabilities_.find(visitor_id);
    if (it == prior_class_probabilities_.end()) {
      return default_prior_class_probabilities_;
    } else {
      return it->second;
    }
  }
  
}  // namespace BOOM
