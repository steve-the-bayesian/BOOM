#ifndef BOOM_FACTOR_MODELS_VISITOR_PRIOR_MANAGER_HPP_
#define BOOM_FACTOR_MODELS_VISITOR_PRIOR_MANAGER_HPP_
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

#include <map>
#include <string>
#include "LinAlg/Vector.hpp"

namespace BOOM {

  // A helper class to manage the prior distribution over class membership for
  // visitors in discrete valued factor models.
  class VisitorPriorManager {
   public:

    // Args:
    //   default_prior: The discrete probability distribution to use as a prior
    //     over a visitor's class membership category when no visitor-specific
    //     prior is specified.
    VisitorPriorManager(const Vector &default_prior)
        : default_prior_class_probabilities_(default_prior)
    {}

    int number_of_classes() const {
      return default_prior_class_probabilities_.size();
    }

    // Set a user-specific prior for a single visitor.
    //
    // Args:
    //   visitor_id:  The target visitor.
    //   probs: A discrete probability distribution over the visitor's class
    //     membership category.
    void set_prior_class_probabilities(const std::string &visitor_id,
                                       const Vector &probs);

    // The prior distribution of class membership for the specified user.
    const Vector &prior_class_probabilities(
        const std::string &visitor_id) const;
    
   private:
    Vector default_prior_class_probabilities_;
    std::map<std::string, Vector> prior_class_probabilities_;
  };

  
}  // namespace BOOM

#endif  // BOOM_FACTOR_MODELS_VISITOR_PRIOR_MANAGER_HPP_
