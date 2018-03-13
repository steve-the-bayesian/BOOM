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

#include "Models/Glm/PosteriorSamplers/CumulativeProbitSampler.hpp"
#include <functional>
#include "Samplers/ScalarSliceSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  typedef CumulativeProbitSampler CPS;
  typedef CumulativeProbitModel CPM;
  CPS::CumulativeProbitSampler(CPM *m, const Ptr<MvnBase> &prior,
                               RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        m_(m),
        beta_prior_(prior),
        suf_(m->xdim()) {}

  double CPS::logpri() const { return beta_prior_->logp(m_->Beta()); }

  void CPS::draw() {
    impute_latent_data();
    draw_beta();
    draw_delta();
  }

  void CPS::impute_latent_data() {
    suf_.clear();
    std::vector<Ptr<OrdinalRegressionData> > data(m_->dat());
    uint maxscore = m_->maxscore();
    uint n = data.size();
    for (int i = 0; i < n; ++i) {
      uint y = data[i]->y();
      const Vector &x(data[i]->x());
      double eta = m_->predict(x);
      double z = 0;
      if (y == 0) {
        z = rtrun_norm_mt(rng(), eta, 1, 0, false);
      } else if (y == maxscore) {
        // TODO:  check delta parameterization y or y+1?
        z = rtrun_norm_mt(rng(), eta, 1, m_->delta(maxscore), true);
      } else {
        // TODO:  check delta parameterization
        double lo = m_->delta(y - 1);
        double hi = m_->delta(y);
        z = rtrun_norm_2_mt(rng(), eta, 1, lo, hi);
      }
      suf_.add_mixture_data(z, x, 1.0);
    }
  }

  void CPS::draw_beta() {
    ivar_ = suf_.xtx() + beta_prior_->siginv();
    mu_ = suf_.xty() + beta_prior_->siginv() * beta_prior_->mu();
    beta_ = rmvn_suf_mt(rng(), ivar_, mu_);
    m_->set_Beta(beta_);
  }

  namespace {

    class PartialTarget : public ScalarTargetFun {
     public:
      typedef std::function<double(const Vector &)> TF;
      PartialTarget(const TF &f, uint pos, const Vector &v)
          : f_(f), pos_(pos), v_(v) {}

      double operator()(double x) const {
        v_[pos_] = x;
        return f_(v_);
      }

     private:
      const TF f_;
      uint pos_;
      mutable Vector v_;
    };

  }  // anonymous namespace

  void CPS::draw_delta() {
    delta_ = m_->delta();
    int k = delta_.size();
    for (int i = 0; i < k; ++i) {
      double lo = i == 0 ? 0 : delta_[i - 1];
      bool top = i == k - 1;
      double hi = top ? BOOM::infinity() : delta_[i + 1];
      PartialTarget f(m_->delta_log_likelihood(), i, delta_);
      ScalarSliceSampler sam(f, true);
      if (!top)
        sam.set_limits(lo, hi);
      else
        sam.set_lower_limit(lo);
      delta_[i] = sam.draw(delta_[i]);
    }
    m_->set_delta(delta_);
  }

}  // namespace BOOM
