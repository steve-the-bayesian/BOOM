// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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

#ifndef BOOM_PRIOR_POLICY_HPP
#define BOOM_PRIOR_POLICY_HPP

#include <map>
#include "Models/ModelTypes.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  // A policy class implementing the relationship between a Model and
  // its PosteriorSampler.
  class PriorPolicy : virtual public Model {
   public:
    PriorPolicy *clone() const override = 0;

    // Invoke each of the sampling methods that have been set, in the
    // order they were set.
    void sample_posterior() override;

    // The log of the prior density function evaluated at the current
    // parameters.
    double logpri() const override;

    // Add a posterior sampler to the current set of sampling methods.
    void set_method(const Ptr<PosteriorSampler> &sampler) override;

    // Clear all posterior samplers.
    void clear_methods() override;

    // Returns the number of sampling methods that have been set.
    int number_of_sampling_methods() const override;

    // Return a specific sampler.
    PosteriorSampler *sampler(int i) override { return samplers_[i].get(); }
    PosteriorSampler const *const sampler(int i) const override {
      return samplers_[i].get();
    }

    // Return the RNG object from the first sampler.  If samplers_ is empty then
    // raise an exception.
    RNG &rng();

   private:
    std::vector<Ptr<PosteriorSampler> > samplers_;
  };

}  // namespace BOOM

#endif  // BOOM_PRIOR_POLICY_HPP
