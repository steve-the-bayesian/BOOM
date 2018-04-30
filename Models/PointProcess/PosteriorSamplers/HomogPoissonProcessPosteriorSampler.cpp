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

#include "Models/PointProcess/PosteriorSamplers/HomogPoissonProcessPosteriorSampler.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    typedef HomogPoissonProcessPosteriorSampler HPS;
  }

  HPS::HomogPoissonProcessPosteriorSampler(
      HomogeneousPoissonProcess *model, const Ptr<GammaModelBase> &prior,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), model_(model), prior_(prior) {}

  double HPS::logpri() const { return prior_->logp(model_->lambda()); }

  void HPS::draw() {
    double a = prior_->alpha() + model_->suf()->count();
    double b = prior_->beta() + model_->suf()->exposure();
    double lambda = rgamma_mt(rng(), a, b);
    model_->set_lambda(lambda);
  }
}  // namespace BOOM
