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

#include "Models/PointProcess/PosteriorSamplers/MmppPosteriorSampler.hpp"

namespace BOOM {
  typedef MmppPosteriorSampler MMPPPS;
  MMPPPS::MmppPosteriorSampler(
      MarkovModulatedPoissonProcess *mmpp, bool initialize_latent_data,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(mmpp),
        first_time_(initialize_latent_data) {}

  // The model should call impute_latent_data one time before this
  // method is run.
  void MMPPPS::draw() {
    if (first_time_) {
      model_->impute_latent_data(rng());
      first_time_ = false;
    }
    model_->sample_complete_data_posterior();
    model_->impute_latent_data(rng());
  }

  double MMPPPS::logpri() const { return model_->logpri(); }

}  // namespace BOOM
