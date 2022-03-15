// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#ifndef BOOM_BETA_POSTERIOR_SAMPLER_HPP_
#define BOOM_BETA_POSTERIOR_SAMPLER_HPP_

#include <stdexcept>
#include "Models/BetaModel.hpp"
#include "Models/DoubleModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Samplers/ScalarSliceSampler.hpp"

namespace BOOM {

  // BetaPosteriorSampler assumes an independent prior on a/(a+b) and
  // a+b.  If priors with support outside (0, 1) and (0, infinity) are
  // given, then support is truncated to those intervals.
  class BetaPosteriorSampler : public PosteriorSampler {
   public:
    BetaPosteriorSampler(BetaModel *model,
                         const Ptr<DoubleModel> &mean_prior,
                         const Ptr<DoubleModel> &sample_size_prior,
                         RNG &seeding_rng = GlobalRng::rng);
    BetaPosteriorSampler *clone_to_new_host(Model *new_host) const override;
    void draw() override;
    double logpri() const override;

   private:
    BetaModel *model_;
    Ptr<DoubleModel> mean_prior_;
    Ptr<DoubleModel> sample_size_prior_;
    ScalarSliceSampler mean_sampler_;
    ScalarSliceSampler sample_size_sampler_;

    // If an exception is encountered in a call to the slice sampler,
    // generate a message containing the state of the sampler.  If *e
    // is non-NULL then print the exception message as part of the
    // error message.
    std::string error_message(const char *thing_being_drawn,
                              const std::exception *e) const;
  };

}  // namespace BOOM

#endif  // BOOM_BETA_POSTERIOR_SAMPLER_HPP_
