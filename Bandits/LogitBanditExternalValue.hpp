#ifndef BOOM_BANDITS_LOGIT_BANDIT_EXTERNAL_VALUE_HPP_
#define BOOM_BANDITS_LOGIT_BANDIT_EXTERNAL_VALUE_HPP_
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

#include "Bandits/LogitBandit.hpp"

#include <functional>

namespace BOOM {

  
  // A logit bandit with value function determined by a functor passed from
  // outside the class.
  class LogitBanditExternalValue : public LogitBandit {
   public:
    // ValueFunctionType is a function taking a double (the success probability)
    // and a vector of strings (the arm levels for a given arm) and returning a
    // double.
    using ValueFunctionType = std::function<double(
        double, const std::vector<std::string> &)>;

    LogitBanditExternalValue(const Ptr<BinomialLogitModel> &model,
                             const Ptr<LinearBanditEncoder> &encoder,
                             const ValueFunctionType &value_function)
        : LogitBandit(model, encoder),
        value_function_(value_function)
      {}

    double value(int arm, const MixedMultivariateData &context) const override;

    Vector optimal_arm_probabilities(const MixedMultivariateData &context,
                                     RNG &rng = GlobalRng::rng) const override;

    Vector value_remaining_distribution(const MixedMultivariateData &context,
                                        RNG &rng = GlobalRng::rng) const override;
    
    
   private:
    ValueFunctionType value_function_;
  };
  
}  // namespace BOOM

#endif  // BOOM_BANDITS_LOGIT_BANDIT_EXTERNAL_VALUE_HPP_
