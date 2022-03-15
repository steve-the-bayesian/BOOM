// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#include "Models/PosteriorSamplers/AbsorbingMarkovConjSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  typedef AbsorbingMarkovConjSampler AMCS;
  typedef MarkovConjSampler MCS;

  AMCS::AbsorbingMarkovConjSampler(MarkovModel *Mod,
                                   const Ptr<ProductDirichletModel> &Q,
                                   const Ptr<DirichletModel> &pi0,
                                   const std::vector<uint> &absorbing_states,
                                   RNG &seeding_rng)
      : MCS(Mod, Q, pi0, seeding_rng),
        mod_(Mod),
        abs_(absorbing_states, mod_->state_space_size()),
        trans_(abs_.complement()) {}

  AMCS::AbsorbingMarkovConjSampler(MarkovModel *Mod,
                                   const Ptr<ProductDirichletModel> &Q,
                                   const std::vector<uint> &absorbing_states,
                                   RNG &seeding_rng)
      : MCS(Mod, Q, seeding_rng),
        mod_(Mod),
        abs_(absorbing_states, mod_->state_space_size()),
        trans_(abs_.complement()) {}

  AMCS::AbsorbingMarkovConjSampler(MarkovModel *Mod, const Matrix &Nu,
                                   const std::vector<uint> &absorbing_states,
                                   RNG &seeding_rng)
      : MCS(Mod, Nu, seeding_rng),
        mod_(Mod),
        abs_(absorbing_states, mod_->state_space_size()),
        trans_(abs_.complement()) {}

  AMCS::AbsorbingMarkovConjSampler(MarkovModel *Mod, const Matrix &Nu,
                                   const Vector &nu,
                                   const std::vector<uint> &absorbing_states,
                                   RNG &seeding_rng)
      : MCS(Mod, Nu, nu, seeding_rng),
        mod_(Mod),
        abs_(absorbing_states, mod_->state_space_size()),
        trans_(abs_.complement()) {}

  double AMCS::logpri() const {
    uint S = mod_->state_space_size();
    Matrix Q(mod_->Q());
    Vector nu(S);
    double ans = 0;
    for (uint s = 0; s < S; ++s) {
      if (!abs_[s]) {
        ans += ddirichlet(Q.row(s), Nu().row(s), true);
        if (ans == BOOM::negative_infinity()) {
          ostringstream err;
          err << "Q(" << s << ") = " << Q.row(s) << endl
              << "Nu(" << s << ") = " << Nu().row(s) << endl
              << "ddirichlet(Q,Nu, true) = "
              << ddirichlet(Q.row(s), Nu().row(s), true) << endl;
          report_error(err.str());
        }
      }
    }

    if (mod_->pi0_fixed()) return ans;

    check_pi0();

    ans +=
        ddirichlet(trans_.select(mod_->pi0()), trans_.select(this->nu()), true);
    return ans;
  }

  AMCS *AMCS::clone_to_new_host(Model *new_host) const {
    return new AMCS(dynamic_cast<MarkovModel *>(new_host),
                    Nu(),
                    nu(),
                    abs_.included_positions(),
                    rng());
  }

  void AMCS::draw() {
    uint S = mod_->state_space_size();
    Matrix Q(mod_->Q());
    Vector nu(S);
    for (uint s = 0; s < S; ++s) {
      if (!abs_[s]) {
        nu = Nu().row(s) + mod_->suf()->trans().row(s);
        Q.row(s) = rdirichlet_mt(rng(), nu);
      } else {
        Q.row(s) = 0.0;
        Q(s, s) = 1.0;
      }
    }

    mod_->set_Q(Q);
    if (mod_->pi0_fixed()) return;
    nu = this->nu() + mod_->suf()->init();
    nu = rdirichlet_mt(rng(), nu);
    mod_->set_pi0(nu);
  }

  void AMCS::find_posterior_mode(double) {
    uint S = mod_->state_space_size();
    Matrix Q(mod_->Q());
    Vector nu(S);
    for (uint s = 0; s < S; ++s) {
      if (abs_[s]) {
        Q.row(s) = 0.0;
        Q(s, s) = 1.0;
      } else {
        nu = Nu().row(s) + mod_->suf()->trans().row(s);
        Q.row(s) = mdirichlet(nu);
      }
    }
    mod_->set_Q(Q);

    if (mod_->pi0_fixed()) return;
    check_pi0();
    nu = this->nu() + mod_->suf()->init();
    mod_->set_pi0(mdirichlet(nu));
  }

}  // namespace BOOM
