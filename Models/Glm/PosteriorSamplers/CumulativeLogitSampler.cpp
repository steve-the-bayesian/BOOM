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

#include "Models/Glm/PosteriorSamplers/CumulativeLogitSampler.hpp"
#include <functional>
#include "Samplers/ScalarSliceSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  typedef CumulativeLogitSampler CLS;
  typedef CumulativeLogitModel CLM;
  CLS::CumulativeLogitSampler(CLM *m, const Ptr<MvnBase> &prior,
                              RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        m_(m),
        beta_prior_(prior),
        suf_(m->xdim()) {}

  double CLS::logpri() const { return beta_prior_->logp(m_->Beta()); }

  void CLS::draw() {
    impute_latent_data();
    draw_beta();
    draw_delta();
  }

  inline double rtrun_logit_mt(RNG &rng, double mean, double scale,
                               double cutoff, bool positive_support) {
    double p = plogis(cutoff, mean, scale);
    double eta = positive_support ? runif_mt(rng, p, 1) : runif_mt(rng, 0, p);
    return qlogis(eta, mean, scale);
  }

  inline double rtrun_logit_2_mt(RNG &rng, double mean, double scale, double lo,
                                 double hi) {
    double plo = plogis(lo, mean, scale);
    double phi = plogis(hi, mean, scale);
    double u = runif_mt(rng, plo, phi);
    return qlogis(u, mean, scale);
  }

  void CLS::impute_latent_data() {
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
        z = rtrun_logit_mt(rng(), eta, 1, 0, false);
      } else if (y == maxscore) {
        z = rtrun_logit_mt(rng(), eta, 1, m_->delta(maxscore - 1), true);
      } else {
        double lo = m_->delta(y - 1);
        double hi = m_->delta(y);
        z = rtrun_logit_2_mt(rng(), eta, 1, lo, hi);
      }
      double r = fabs(z - eta);
      double lambda = Logit::draw_lambda_mt(rng(), r);
      suf_.add_data(data[i]->x(), z, 1.0 / lambda);
    }
  }

  void CLS::draw_beta() {
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

  void CLS::draw_delta() {
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
