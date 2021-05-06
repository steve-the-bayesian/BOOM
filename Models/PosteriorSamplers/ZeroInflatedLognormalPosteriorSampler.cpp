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
#include "Models/PosteriorSamplers/ZeroInflatedLognormalPosteriorSampler.hpp"
#include "Models/ZeroInflatedLognormalModel.hpp"

namespace BOOM {

  double ZeroInflatedLognormalPosteriorSampler::logpri() const {
    return model_->Gaussian_model()->logpri() +
           model_->Binomial_model()->logpri();
  }

  ZeroInflatedLognormalPosteriorSampler *
  ZeroInflatedLognormalPosteriorSampler::clone_to_new_host(
      Model *new_host) const {
    return new ZeroInflatedLognormalPosteriorSampler(
        dynamic_cast<ZeroInflatedLognormalModel *>(new_host),
        rng());
  }

  void ZeroInflatedLognormalPosteriorSampler::draw() {
    model_->Gaussian_model()->sample_posterior();
    model_->Binomial_model()->sample_posterior();
  }
}  // namespace BOOM
