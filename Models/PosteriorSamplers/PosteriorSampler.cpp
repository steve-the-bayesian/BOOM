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

#include "Models/ModelTypes.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  void intrusive_ptr_add_ref(PosteriorSampler *m) { m->up_count(); }

  void intrusive_ptr_release(PosteriorSampler *m) {
    m->down_count();
    if (m->ref_count() == 0) delete m;
  }

  PosteriorSampler::PosteriorSampler(RNG &seeding_rng)
      : rng_(seed_rng(seeding_rng)) {}

  PosteriorSampler::PosteriorSampler(const PosteriorSampler &rhs)
      : RefCounted(rhs) {
    rng_.seed(seed_rng(rhs.rng()));
  }

  PosteriorSampler *PosteriorSampler::clone_to_new_host(Model *host) const {
    report_error("Concrete class needs to define clone_to_new_host.");
    return nullptr;
  }

  void PosteriorSampler::set_seed(unsigned long s) { rng_.seed(s); }

  void PosteriorSampler::find_posterior_mode(double epsilon) {
    report_error("Sampler class does not implement find_posterior_mode.");
  }

  double PosteriorSampler::log_prior_density(
      const ConstVectorView &parameters) const {
    report_error("Sampler class does not implement log_prior_density.");
    return negative_infinity();
  }

  double PosteriorSampler::increment_log_prior_gradient(
      const ConstVectorView &parameters, VectorView gradient) const {
    report_error(
        "Sampler class does not implement "
        "increment_log_prior_gradient.");
    return negative_infinity();
  }

}  // namespace BOOM
