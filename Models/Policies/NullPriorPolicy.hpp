#ifndef BOOM_NULL_PRIOR_POLICY_HPP
#define BOOM_NULL_PRIOR_POLICY_HPP

/*
  Copyright (C) 2005-2018 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#include "Models/ModelTypes.hpp"

namespace BOOM {

  // A NullPriorPolicy provides no-op overrides for all the methods normally
  // handled by PriorPolicy.  The intent is to facilitate a class that wants to
  // implement its own sample_posterior() method without deferring to a
  // PosteriorSampler.
  class NullPriorPolicy : virtual public Model {
   public:
    NullPriorPolicy *clone() const override = 0;

    // Invoke each of the sampling methods that have been set, in the
    // order they were set.
    void sample_posterior() override {}
    double logpri() const override { return 0; }

    // Returns the number of sampling methods that have been set.
    int number_of_sampling_methods() const override { return 0; }

   protected:
    PosteriorSampler *sampler(int i) override { return nullptr; }
    PosteriorSampler const *const sampler(int i) const override {
      return nullptr;
    }
  };

}  // namespace BOOM

#endif  // BOOM_NULL_PRIOR_POLICY_HPP
