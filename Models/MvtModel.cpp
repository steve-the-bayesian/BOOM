// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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

#include "Models/MvtModel.hpp"
#include <cmath>
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/ScaledChisqModel.hpp"
#include "Models/WeightedData.hpp"
#include "Models/WeightedMvnModel.hpp"
#include "TargetFun/TargetFun.hpp"
#include "distributions.hpp"
#include "numopt.hpp"

namespace BOOM {

  namespace {
    typedef MvtModel MVT;
  }

  MVT::MvtModel(uint p, double mu, double sig, double nu)
      : ParamPolicy(),
        DataPolicy(),
        PriorPolicy(),
        mvn(new WeightedMvnModel(p, mu, sig)),
        wgt(new ScaledChisqModel(nu)) {
    ParamPolicy::add_model(mvn);
    ParamPolicy::add_model(wgt);
  }

  MVT::MvtModel(const Vector &mean, const SpdMatrix &Var, double Nu)
      : ParamPolicy(),
        DataPolicy(),
        PriorPolicy(),
        mvn(new WeightedMvnModel(mean, Var)),
        wgt(new ScaledChisqModel(Nu)) {
    ParamPolicy::add_model(mvn);
    ParamPolicy::add_model(wgt);
  }

  MVT::MvtModel(const MvtModel &rhs)
      : Model(rhs),
        VectorModel(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        LatentVariableModel(rhs),
        LoglikeModel(rhs),
        LocationScaleVectorModel(rhs),
        mvn(rhs.mvn->clone()),
        wgt(rhs.wgt->clone()) {
    ParamPolicy::add_model(mvn);
    ParamPolicy::add_model(wgt);
  }

  MvtModel *MVT::clone() const { return new MvtModel(*this); }

  void MVT::initialize_params() { mle(); }

  Ptr<VectorParams> MVT::Mu_prm() { return mvn->Mu_prm(); }
  Ptr<SpdParams> MVT::Sigma_prm() { return mvn->Sigma_prm(); }
  Ptr<UnivParams> MVT::Nu_prm() { return wgt->Nu_prm(); }

  const Ptr<VectorParams> MVT::Mu_prm() const { return mvn->Mu_prm(); }
  const Ptr<SpdParams> MVT::Sigma_prm() const { return mvn->Sigma_prm(); }
  const Ptr<UnivParams> MVT::Nu_prm() const { return wgt->Nu_prm(); }

  int MVT::dim() const { return mu().size(); }
  const Vector &MVT::mu() const { return Mu_prm()->value(); }
  const SpdMatrix &MVT::Sigma() const { return Sigma_prm()->var(); }
  Matrix MVT::Sigma_chol() const { return Sigma_prm()->var_chol(); }
  const SpdMatrix &MVT::siginv() const { return Sigma_prm()->ivar(); }
  double MVT::ldsi() const { return Sigma_prm()->ldsi(); }
  double MVT::nu() const { return Nu_prm()->value(); }

  void MVT::set_mu(const Vector &mu) { Mu_prm()->set(mu); }
  void MVT::set_Sigma(const SpdMatrix &Sig) { Sigma_prm()->set_var(Sig); }
  void MVT::set_siginv(const SpdMatrix &ivar) { Sigma_prm()->set_ivar(ivar); }
  void MVT::set_nu(double nu) { Nu_prm()->set(nu); }

  double MVT::pdf(const VectorData *dp, bool logscale) const {
    return pdf(dp->value(), logscale);
  }
  double MVT::pdf(const Data *dp, bool logscale) const {
    const Vector &v(dynamic_cast<const VectorData *>(dp)->value());
    return pdf(v, logscale);
  }
  double MVT::pdf(const Vector &x, bool logscale) const {
    return dmvt(x, mu(), siginv(), nu(), ldsi(), logscale);
  }

  double MVT::logp(const Vector &x) const { return pdf(x, true); }

  void MVT::add_data(const Ptr<VectorData> &dp) {
    DataPolicy::add_data(dp);
    NEW(DoubleData, w)(1.0);
    NEW(WeightedVectorData, v)(dp, w);
    wgt->add_data(w);
    mvn->add_data(v);
  }

  void MVT::add_data(const Ptr<Data> &dp) {
    Ptr<VectorData> d = DAT(dp);
    add_data(d);
  }

  //======================================================================

  double MVT::loglike(const Vector &mu_siginv_triangle_nu) const {
    const DatasetType &dat(this->dat());
    const ConstVectorView mu(mu_siginv_triangle_nu, 0, dim());
    SpdMatrix siginv(dim());
    Vector::const_iterator it = mu_siginv_triangle_nu.cbegin() + dim();
    siginv.unvectorize(it, true);
    double ldsi = siginv.logdet();
    double nu = mu_siginv_triangle_nu.back();
    double lognu = log(nu);

    const double logpi = 1.1447298858494;
    uint n = dat.size();
    uint d = mu.size();
    double half_npd = .5 * (nu + d);

    double ans = lgamma(half_npd) - lgamma(nu / 2) - .5 * d * (lognu + logpi);
    ans += .5 * ldsi + half_npd * lognu;
    ans *= n;

    for (uint i = 0; i < n; ++i) {
      double delta = siginv.Mdist(mu, dat[i]->value());
      ans -= half_npd * log(nu + delta / nu);
    }

    return ans;
  }
  //======================================================================

  typedef WeightedVectorData WVD;

  void MVT::Impute(bool sample, RNG &rng) {
    std::vector<Ptr<WVD> > &V(mvn->dat());

    for (uint i = 0; i < V.size(); ++i) {
      Ptr<WVD> d = V[i];
      const Vector &y(d->value());
      double delta = siginv().Mdist(y, mu());
      double a = (nu() + y.length()) / 2.0;
      double b = (nu() + delta) / 2.0;
      double w = sample ? rgamma_mt(rng, a, b) : a / b;
      d->set_weight(w);
    }
    mvn->refresh_suf();
    wgt->refresh_suf();
  }
  void MVT::impute_latent_data(RNG &rng) { Impute(true, rng); }
  void MVT::Estep() { Impute(false, GlobalRng::rng); }

  //------------------------------------------------------------

  class MvtNuTF {
   public:
    explicit MvtNuTF(MvtModel *Mod) : mod(Mod) {}
    MvtNuTF *clone() const { return new MvtNuTF(*this); }
    double operator()(const Vector &Nu) const;
    double operator()(const Vector &Nu, Vector &g) const;

   private:
    double Loglike(const Vector &Nu, Vector &g, uint nd) const;
    MvtModel *mod;
  };

  double MvtNuTF::operator()(const Vector &Nu) const {
    Vector g;
    return Loglike(Nu, g, 0);
  }

  double MvtNuTF::operator()(const Vector &Nu, Vector &g) const {
    return Loglike(Nu, g, 1);
  }

  double MvtNuTF::Loglike(const Vector &Nu, Vector &g, uint nd) const {
    const std::vector<Ptr<VectorData> > &dat(mod->dat());
    double ldsi = mod->ldsi();
    const SpdMatrix &Siginv(mod->siginv());
    const Vector &mu(mod->mu());
    const double logpi = 1.1447298858494;
    double nu = Nu[0];
    double lognu = log(nu);
    uint n = dat.size();
    uint d = mu.size();
    double half_npd = .5 * (nu + d);

    double ans = lgamma(half_npd) - lgamma(nu / 2) - .5 * d * (lognu + logpi);
    ans += .5 * ldsi + half_npd * lognu;
    ans *= n;

    if (nd > 0) {
      g[0] = .5 * (digamma(half_npd) - digamma(nu / 2.0) - d / nu);
      g[0] += half_npd / nu + .5 * lognu;
      g[0] *= n;
    }

    for (uint i = 0; i < n; ++i) {
      double delta = Siginv.Mdist(mu, dat[i]->value());
      double npd = nu + delta;
      ans -= half_npd * log(npd);
      if (nd > 0) {
        g[0] -= half_npd / npd + .5 * log(npd);
      }
    }
    return ans;
  }
  //------------------------------------------------------------

  void MVT::mle() {
    const double eps = 1e-5;
    double dloglike = eps + 1;
    double loglike = this->loglike(vectorize_params());
    double old = loglike;
    Vector Nu(1, nu());
    while (dloglike > eps) {
      Estep();
      mvn->mle();
      MvtNuTF f(this);
      loglike = max_nd1(Nu, Target(f), dTarget(f));
      set_nu(Nu[0]);
      dloglike = loglike - old;
      old = loglike;
    }
  }

  double MVT::complete_data_loglike() const {
    Vector params = vectorize_params();
    params.pop_back();
    double ans = mvn->loglike(params);
    Vector nu_vector(1, nu());
    ans += wgt->loglike(nu_vector);
    return ans;
  }

  Vector MVT::sim(RNG &rng) const {
    Vector ans = rmvn_L_mt(rng, mu().zero(), Sigma_chol());
    double nu = this->nu();
    double w = rgamma(nu / 2.0, nu / 2.0);
    return mu() + ans / sqrt(w);
  }

}  // namespace BOOM
