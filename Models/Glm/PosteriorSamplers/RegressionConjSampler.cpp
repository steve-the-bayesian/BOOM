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

#include "Models/Glm/PosteriorSamplers/RegressionConjSampler.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef RegressionConjSampler RCS;
  RCS::RegressionConjSampler(
      RegressionModel *model, const Ptr<MvnGivenXandSigma> &coefficient_prior,
      const Ptr<GammaModelBase> &residual_precision_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        m_(model),
        mu_(coefficient_prior),
        siginv_(residual_precision_prior),
        sigsq_sampler_(siginv_) {}

  const Vector &RCS::b0() const { return mu_->mu(); }
  double RCS::kappa() const { return mu_->prior_sample_size(); }
  double RCS::prior_df() const { return 2.0 * siginv_->alpha(); }
  double RCS::prior_ss() const { return 2.0 * siginv_->beta(); }

  void RCS::set_posterior_suf() {
    const Vector &b0(this->b0());
    double sigsq = m_->sigsq();

    SpdMatrix Ominv = mu_->siginv();  // mu_->siginv() is Ominv/sigsq
    Ominv *= sigsq;

    beta_tilde = m_->xty() + Ominv * b0;
    ivar = Ominv + m_->xtx();
    beta_tilde = ivar.solve(beta_tilde);

    SS = prior_ss() + m_->yty() + Ominv.Mdist(b0);
    SS -= ivar.Mdist(beta_tilde);
    DF = m_->suf()->n() + prior_df();
  }

  void RCS::draw() {
    set_posterior_suf();
    double sigsq = sigsq_sampler_.draw(rng(), DF - prior_df(), SS - prior_ss());
    ivar /= sigsq;
    beta_tilde = rmvn_ivar(beta_tilde, ivar);
    m_->set_Beta(beta_tilde);
    m_->set_sigsq(sigsq);
  }

  void RCS::find_posterior_mode(double) {
    set_posterior_suf();
    m_->set_Beta(beta_tilde);
    if (DF <= 2)
      m_->set_sigsq(0.0);  // mode = (alpha-1)/beta
    else
      m_->set_sigsq(SS / (DF - 2));  //   alpha = df/2  beta = ss/2
  }

  double RCS::logpri() const {
    double ans = mu_->logp(m_->Beta());
    ans += sigsq_sampler_.log_prior(m_->sigsq());
    return ans;
  }

}  // namespace BOOM
