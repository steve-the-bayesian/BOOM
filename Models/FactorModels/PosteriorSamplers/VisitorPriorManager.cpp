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

  VisitorPriorManager::VisitorPriorManager(const Vector &default_prior)
      : default_prior_class_probabilities_(default_prior),
        known_users_(default_prior.size()),
        known_user_probabilities_(default_prior.size())
  {
    for (int i = 0; i < default_prior.size(); ++i) {
      Vector prior(default_prior.size(), 0.0);
      prior[i] = 1.0;
      known_user_probabilities_[i] = prior;
    }
  }


  size_t VisitorPriorManager::number_known() const {
    size_t ans = 0;
    for (const auto &el : known_users_) {
      ans += el.size(); 
    }
    return ans;
  }
  
  void VisitorPriorManager::set_prior_class_probabilities(
      const std::string &visitor_id,
      const Vector &probs) {
    check_probabilities(probs, true);

    int category = probs.imax();
    if (probs[category] > .999) {
      known_users_[category].insert(visitor_id);
    } else {
      prior_class_probabilities_[visitor_id] = probs;
    }
  }

  const Vector &VisitorPriorManager::prior_class_probabilities(
      const std::string &visitor_id) const {
    auto it = prior_class_probabilities_.find(visitor_id);
    if (it != prior_class_probabilities_.end()) {
      return it->second;
    } else {
      for (size_t i = 0; i < known_users_.size(); ++i) {
        if (known_users_[i].find(visitor_id) != known_users_[i].end()) {
          return known_user_probabilities_[i];
        }
      }
    } 
    return default_prior_class_probabilities_;
  }
  
}  // namespace BOOM
