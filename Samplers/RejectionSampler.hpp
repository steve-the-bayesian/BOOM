// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#ifndef BOOM_SAMPLERS_REJECTION_SAMPLER_HPP_
#define BOOM_SAMPLERS_REJECTION_SAMPLER_HPP_

#include <cstdint>
#include <functional>
#include "Samplers/DirectProposal.hpp"

namespace BOOM {

  // A class for doing rejection sampling, which works by sampling
  // draws from a proposal distribution "g" that is easy to simulate
  // from, and easy to evaluate.  It must be possible to scale the
  // proposal distribution so that it is larger than the target
  // distribution "f" at every point.  (I.e. C * g(x) >= f(x) for any
  // x, where C is a constant).
  //
  // The algorithm is propose x from g(x).  Accept the proposal with
  // probability f(x) / C * g(x).  Keep trying until you accept.
  class RejectionSampler {
   public:
    typedef std::function<double(const Vector &)> Target;
    RejectionSampler(const Target &log_target_density,
                     const Ptr<DirectProposal> &proposal);

    // Returns a draw from the target distribution.  If the number of
    // rejected proposal exceeds a limit set by set_rejection_limit(),
    // then the returned Vector will be empty (have size 0).  The
    // initial limit is infinite.
    Vector draw(RNG &rng);

    // The number of rejections possible before a call to draw() is
    // considered as failed.  A negative number is interpreted as an
    // infinite limit.
    void set_rejection_limit(std::int64_t limit);

    // Rejection sampling only works when f(x) < Mg(x) for all x,
    // where f is the target, and g is the proposal.  The 'offset'
    // argument here is log(M).
    void set_log_proposal_density_offset(double offset);

   private:
    Target log_target_density_;
    Ptr<DirectProposal> proposal_;
    double log_proposal_density_offset_;
    std::int64_t rejection_limit_;
  };

}  // namespace BOOM

#endif  //  BOOM_SAMPLERS_REJECTION_SAMPLER_HPP_
