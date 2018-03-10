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
#include "Samplers/MetropolisHastings.hpp"
#include <utility>
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef MetropolisHastings MH;

  MH::MetropolisHastings(const Target &target, const Ptr<MH_Proposal> &prop,
                         RNG *rng)
      : Sampler(rng), f_(target), prop_(prop), accepted_(false) {}

  void MH::set_proposal(const Ptr<MH_Proposal> &p) { prop_ = p; }

  void MH::set_target(const Target &f) { f_ = f; }

  Vector MH::draw(const Vector &old) {
    cand_ = prop_->draw(old, &rng());
    double logp_cand = logp(cand_);
    double logp_old = logp(old);
    if (!std::isfinite(logp_cand)) {
      if (std::isfinite(logp_old)) {
        accepted_ = false;
        return old;
      } else {
        std::ostringstream err;
        err << "Argument to 'draw' resulted in a non-finite "
            << "log posterior" << std::endl
            << old;
        report_error(err.str());
      }
    } else if (!std::isfinite(logp_old)) {
      // In this case you started with an illegal value of old, but
      // got a legal value of cand, so you should accept.
      accepted_ = true;
      return cand_;
    }

    // Both log densities are finite, so it is safe to proceed.
    double num = logp_cand - logp_old;
    double denom, d1, d2;
    denom = d1 = d2 = 0.0;
    if (!prop_->sym()) {
      d1 = prop_->logf(cand_, old);
      d2 = prop_->logf(old, cand_);
      denom = d1 - d2;
    }

    double u = log(runif_mt(rng()));
    accepted_ = u < num - denom;
    return accepted_ ? cand_ : old;
  }

  bool MH::last_draw_was_accepted() const { return accepted_; }

  double MH::logp(const Vector &x) const { return f_(x); }

  typedef ScalarMetropolisHastings SMH;
  SMH::ScalarMetropolisHastings(const ScalarTarget &f,
                                const Ptr<MH_ScalarProposal> &prop, RNG *rng)
      : ScalarSampler(rng), f_(f), prop_(prop), accepted_(false) {}

  double SMH::draw(double old) {
    double cand = prop_->draw(old, &rng());
    double logp_cand = f_(cand);
    double logp_old = f_(old);
    if (!std::isfinite(logp_cand)) {
      if (std::isfinite(logp_old)) {
        accepted_ = false;
        return old;
      } else {
        std::ostringstream err;
        err << "Argument to 'draw' resulted in a non-finite "
            << "log posterior" << std::endl
            << old;
        report_error(err.str());
      }
    } else if (!std::isfinite(logp_old)) {
      // The candidate has a fininte log posterior, but the original
      // does not.
      accepted_ = true;
      return cand;
    }
    // Both log densities are finite, so it is safe to proceed.
    double num = logp_cand - logp(old);

    double denom, d1, d2;
    denom = d1 = d2 = 0;
    if (!prop_->sym()) {
      d1 = prop_->logf(cand, old);
      d2 = prop_->logf(old, cand);
      denom = d1 - d2;
    }
    double u = log(runif_mt(rng()));
    accepted_ = u < num - denom;
    return accepted_ ? cand : old;
  }

  bool SMH::last_draw_was_accepted() const { return accepted_; }

  double SMH::logp(double x) const { return f_(x); }
}  // namespace BOOM
