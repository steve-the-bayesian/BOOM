// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include "Models/Glm/PosteriorSamplers/BinomialLogitSamplerTim.hpp"
#include <functional>

namespace BOOM {

  typedef BinomialLogitSamplerTim BLST;
  BLST::BinomialLogitSamplerTim(BinomialLogitModel *m, const Ptr<MvnBase> &pri,
                                bool mode_is_stable, double nu,
                                RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        m_(m),
        pri_(pri),
        sam_([this](const Vector &beta) { return this->logp(beta); },
             [this](const Vector &beta, Vector &gradient) {
               return this->dlogp(beta, gradient);
             },
             [this](const Vector &beta, Vector &gradient, Matrix &hessian) {
               return this->d2logp(beta, gradient, hessian);
             },
             nu),
        save_modes_(mode_is_stable) {
    if (mode_is_stable) sam_.fix_mode();
  }

  void BLST::draw() {
    if (save_modes_) {
      const Selector &inc(m_->inc());
      const Mode &mode(locate_mode(inc));
      sam_.set_mode(mode.location, mode.precision);
    }
    Vector beta = sam_.draw(m_->included_coefficients());
    m_->set_included_coefficients(beta);
  }

  double BLST::logpri() const {
    return pri_->logp(m_->included_coefficients());
  }

  double BLST::Logp(const Vector &beta, Vector &g, Matrix &h, int nd) const {
    double ans = pri_->Logp(beta, g, h, nd);
    Vector *gp = nd > 0 ? &g : 0;
    Matrix *hp = nd > 1 ? &h : 0;
    ans += m_->log_likelihood(beta, gp, hp, false);
    return ans;
  }

  double BLST::logp(const Vector &beta) const {
    Vector g;
    Matrix h;
    return Logp(beta, g, h, 0);
  }

  double BLST::dlogp(const Vector &beta, Vector &g) const {
    Matrix h;
    return Logp(beta, g, h, 1);
  }

  double BLST::d2logp(const Vector &beta, Vector &g, Matrix &h) const {
    return Logp(beta, g, h, 2);
  }

  const BLST::Mode &BLST::locate_mode(const Selector &inc) {
    Mode &mode(modes_[inc]);
    if (mode.empty()) {
      bool ok = sam_.locate_mode(m_->included_coefficients());
      if (ok) {
        mode.location = sam_.mode();
        mode.precision = sam_.ivar();
      }
    }
    return mode;
  }

}  // namespace BOOM
