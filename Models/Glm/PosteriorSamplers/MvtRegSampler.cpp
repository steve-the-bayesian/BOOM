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
#include "Models/Glm/PosteriorSamplers/MvtRegSampler.hpp"
#include "LinAlg/Cholesky.hpp"
#include "Models/GammaModel.hpp"
#include "Samplers/SliceSampler.hpp"
#include "TargetFun/Loglike.hpp"
#include "distributions.hpp"
//#include "TargetFun/ScalarLogpostTF.hpp"

namespace BOOM {

  typedef MvtRegSampler MVTRS;

  struct Logp_nu {
    Logp_nu(const Ptr<ScaledChisqModel> &Numod, const Ptr<DoubleModel> &Pri)
        : loglike(Numod.get()), pri(Pri) {}
    double operator()(const Vector &x) const {
      return loglike(x) + pri->logp(x[0]);
    }
    LoglikeTF loglike;
    Ptr<DoubleModel> pri;
  };

  MVTRS::MvtRegSampler(MvtRegModel *m, const Matrix &B_guess, double prior_nobs,
                       double prior_df, const SpdMatrix &Sigma_guess,
                       const Ptr<DoubleModel> &Nu_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        mod(m),
        reg_model(new MultivariateRegressionModel(mod->Beta(), mod->Sigma())),
        nu_model(new ScaledChisqModel(m->nu())),
        nu_prior(Nu_prior) {
    reg_model->set_params(mod->Beta_prm(), mod->Sigma_prm());
    reg_sampler = new MultivariateRegressionSampler(
        reg_model.get(), B_guess, prior_nobs, prior_df, Sigma_guess);
    nu_model->set_prm(mod->Nu_prm());
    Logp_nu nu_logpost(nu_model, nu_prior);
    nu_sampler = new SliceSampler(nu_logpost, true);
  }

  void MVTRS::draw() {
    clear_suf();
    impute_w();
    draw_Sigma();
    draw_Beta();
    draw_nu();
  }

  double MVTRS::logpri() const {
    double ans = nu_model->logp(mod->nu());
    ans += reg_sampler->logpri();
    return ans;
  }

  void MVTRS::clear_suf() {
    reg_model->suf()->clear();
    nu_model->suf()->clear();
  }

  void MVTRS::impute_w() {
    Ptr<MvRegSuf> rs = reg_model->suf();
    Ptr<GammaSuf> gs = nu_model->suf();

    const std::vector<Ptr<MvRegData> > &dat(mod->dat());
    uint n = dat.size();
    for (uint i = 0; i < n; ++i) {
      Ptr<MvRegData> dp = dat[i];
      double w = impute_w(dp);
      rs->update_raw_data(dp->y(), dp->x(), w);
      gs->update_raw(w);
    }
  }

  double MVTRS::impute_w(const Ptr<MvRegData> &dp) {
    const Vector &y(dp->y());
    const Vector &x(dp->x());
    yhat = mod->predict(x);
    double nu = mod->nu();
    double ss = mod->Siginv().Mdist(y, yhat);
    double w = rgamma((nu + y.size()) / 2, (nu + ss) / 2);
    return w;
  }

  void MVTRS::draw_Sigma() { reg_sampler->draw_Sigma(); }

  void MVTRS::draw_Beta() { reg_sampler->draw_Beta(); }

  void MVTRS::draw_nu() {
    Vector nu(1, mod->nu());
    nu = nu_sampler->draw(nu);
    mod->set_nu(nu[0]);
  }
}  // namespace BOOM
