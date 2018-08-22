// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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
#include "Models/Glm/PosteriorSamplers/VsPriorSampler.hpp"
#include "Models/PosteriorSamplers/BetaBinomialSampler.hpp"
#include "Models/PosteriorSamplers/FixedProbBinomialSampler.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {

  typedef VsPriorSampler VSPS;

  VSPS::VsPriorSampler(StructuredVariableSelectionPrior *Vsp,
                       const Ptr<BetaModel> &Beta,
                       RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        vsp(Vsp),
        forced_in_(Vsp->potential_nvars(), false),
        forced_out_(Vsp->potential_nvars(), false) {
    uint n = vsp->potential_nvars();
    Ptr<BetaBinomialSampler> sam;
    for (uint i = 0; i < n; ++i) {
      Ptr<BinomialModel> mod(vsp->variable(i)->model());
      Ptr<BetaModel> bp(Beta->clone());
      sam = new BetaBinomialSampler(mod.get(), bp);
      mod->set_method(sam);
      sam_.push_back(sam);
    }
  }

  VSPS::VsPriorSampler(StructuredVariableSelectionPrior *Vsp,
                       const Vector &pi_guess,
                       const Vector &a_plus_b,
                       RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        vsp(Vsp),
        forced_in_(Vsp->potential_nvars(), false),
        forced_out_(Vsp->potential_nvars(), false) {
    uint n = vsp->potential_nvars();
    assert(n == pi_guess.size() && n == a_plus_b.size());
    Ptr<PosteriorSampler> sam;
    for (uint i = 0; i < n; ++i) {
      double N = a_plus_b[i];
      assert(N > 0);
      Ptr<BinomialModel> mod = vsp->variable(i)->model();
      if (std::isfinite(N)) {
        double a = N * pi_guess[i];
        double b = N * (1 - pi_guess[i]);
        NEW(BetaModel, bp)(a, b);
        sam = new BetaBinomialSampler(mod.get(), bp);
        mod->set_method(sam);
        sam_.push_back(sam);
      } else {  // N is finite
        double p = pi_guess[i];
        vsp->variable(i)->set_prob(p);
        sam = new FixedProbBinomialSampler(mod.get(), p);
        mod->set_method(sam);
        sam_.push_back(sam);
      }
    }
  }

  VSPS::VsPriorSampler(StructuredVariableSelectionPrior *Vsp,
                       const std::vector<Ptr<BetaModel>> &Beta,
                       const Selector &forced_in,
                       const Selector &forced_out,
                       RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        vsp(Vsp),
        forced_in_(forced_in),
        forced_out_(forced_out) {
    uint n = vsp->potential_nvars();
    assert(forced_in_.nvars_possible() == n);
    assert(forced_out_.nvars_possible() == n);
    Ptr<PosteriorSampler> sam;
    for (uint i = 0; i < n; ++i) {
      Ptr<BinomialModel> mod = vsp->variable(i)->model();
      if (forced_in_[i])
        sam = new FixedProbBinomialSampler(mod.get(), 1.0);
      else if (forced_out_[i])
        sam = new FixedProbBinomialSampler(mod.get(), 0.0);
      else
        sam = new BetaBinomialSampler(mod.get(), Beta[i]);
      mod->set_method(sam);
      sam_.push_back(sam);
    }
  }

  VSPS::VsPriorSampler(StructuredVariableSelectionPrior *Vsp,
                       const std::vector<Ptr<BetaModel>> &Beta,
                       RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        vsp(Vsp),
        forced_in_(Vsp->potential_nvars(), false),
        forced_out_(Vsp->potential_nvars(), false) {
    uint n = vsp->potential_nvars();
    if (Beta.size() != n) {
      ostringstream msg;
      msg << "Vector of beta priors has the wrong size in VsPriorSampler "
          << "constructor.  There are " << n << " variables but " << Beta.size()
          << " beta distributions.";
      report_error(msg.str());
    }
    Ptr<BetaBinomialSampler> sam;
    for (uint i = 0; i < n; ++i) {
      Ptr<BinomialModel> mod = vsp->variable(i)->model();
      sam = new BetaBinomialSampler(mod.get(), Beta[i]);
      mod->set_method(sam);
      sam_.push_back(sam);
    }
  }

  void VSPS::draw() {
    uint n = potential_nvars();
    for (uint i = 0; i < n; ++i) {
      Ptr<BinomialModel> mod = vsp->variable(i)->model();
      mod->sample_posterior();
    }
  }

  uint VSPS::potential_nvars() const { return vsp->potential_nvars(); }

  double VSPS::logpri() const {
    uint n = potential_nvars();
    double ans = 0;
    for (uint i = 0; i < n; ++i) ans += sam_[i]->logpri();
    return ans;
  }

}  // namespace BOOM
