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

#include "Models/Glm/TRegression.hpp"
#include "Models/Glm/WeightedRegressionModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/ScaledChisqModel.hpp"

#include "distributions.hpp"
#include "numopt.hpp"

#include "cpputil/math_utils.hpp"

#include <cmath>
#include <iomanip>

namespace BOOM {

  TRegressionModel::TRegressionModel(uint p)
      : ParamPolicy(new GlmCoefs(p), new UnivParams(1.0),
                    new UnivParams(30.0)) {}

  TRegressionModel::TRegressionModel(const Vector &b, double Sigma, double nu)
      : ParamPolicy(new GlmCoefs(b), new UnivParams(square(Sigma)),
                    new UnivParams(nu)) {}

  TRegressionModel::TRegressionModel(const Matrix &X, const Vector &y)
      : ParamPolicy(new GlmCoefs(X.ncol()), new UnivParams(1.0),
                    new UnivParams(30.0)) {
    if (X.nrow() != y.size()) {
      report_error("X and y are incompatible in TRegressionModel constructor.");
    }
    for (int i = 0; i < y.size(); ++i) {
      NEW(RegressionData, dp)(y[i], X.row(i));
      add_data(dp);
    }
  }

  TRegressionModel *TRegressionModel::clone() const {
    return new TRegressionModel(*this);
  }
  GlmCoefs &TRegressionModel::coef() { return prm1_ref(); }
  const GlmCoefs &TRegressionModel::coef() const { return prm1_ref(); }
  Ptr<GlmCoefs> TRegressionModel::coef_prm() { return prm1(); }
  const Ptr<GlmCoefs> TRegressionModel::coef_prm() const { return prm1(); }

  Ptr<UnivParams> TRegressionModel::Sigsq_prm() { return prm2(); }
  const Ptr<UnivParams> TRegressionModel::Sigsq_prm() const { return prm2(); }
  const double &TRegressionModel::sigsq() const { return prm2_ref().value(); }
  double TRegressionModel::sigma() const { return sqrt(sigsq()); }
  void TRegressionModel::set_sigsq(double s2) { Sigsq_prm()->set(s2); }

  Ptr<UnivParams> TRegressionModel::Nu_prm() { return prm3(); }
  const Ptr<UnivParams> TRegressionModel::Nu_prm() const { return prm3(); }
  const double &TRegressionModel::nu() const { return prm3_ref().value(); }
  void TRegressionModel::set_nu(double Nu) { Nu_prm()->set(Nu); }

  double TRegressionModel::log_likelihood(const Vector &full_beta, double sigsq,
                                          double nu) const {
    const double sigma = sqrt(sigsq);
    const std::vector<Ptr<RegressionData>> &data(dat());
    const Selector &included_coefficients(coef().inc());
    const Vector beta = included_coefficients.select(full_beta);
    double ans = 0;
    for (int i = 0; i < data.size(); ++i) {
      const Vector x = coef().inc().select(data[i]->x());
      ans += dstudent(data[i]->y(), beta.dot(x), sigma, nu, true);
    }
    return ans;
  }

  double TRegressionModel::Loglike(const Vector &beta_sigsq_nu, Vector &g,
                                   Matrix &h, uint nd) const {
    double nu = beta_sigsq_nu.back();
    double sigsq = beta_sigsq_nu[beta_sigsq_nu.size() - 2];
    double sigma = sqrt(sigsq);
    const Selector &inclusion_indicators(coef().inc());
    int beta_dim = inclusion_indicators.nvars();
    const Vector beta(ConstVectorView(beta_sigsq_nu, 0, beta_dim));
    double ans = 0;
    if (nd > 0) {
      g = 0;
      h = 0;
    }

    for (uint i = 0; i < dat().size(); ++i) {
      const Vector X = coef().inc().select((dat())[i]->x());
      const double yhat = beta.dot(X);
      const double y = (dat())[i]->y();
      ans += dstudent(y, yhat, sigma, nu, true);
      if (nd > 0) {
        double e = y - yhat;
        double esq_ns = e * e / (nu * sigsq);
        double frac = esq_ns / (1 + esq_ns);

        Vector gbeta = ((nu + 1) * frac / e) * X;

        Vector gsignu(2);
        gsignu[0] = -1 / (2 * sigsq);
        gsignu[0] *= (1 - (nu + 1) * frac);

        gsignu[1] = .5 * (digamma((nu + 1) / 2) - digamma(nu / 2) - 1.0 / nu -
                          log1p(esq_ns) + frac * (nu + 1) / nu);
        g += concat(gbeta, gsignu);
        if (nd > 1) {
          report_error(
              "second derivatives of TRegression are not yet implemented.");
          double esq = e * e;
          double sn = sigsq * nu;
          double esp = esq + sn;

          Matrix hbb = X.outer() * ((nu + 1) * ((esq - sn) / esp));
          Vector hbs = (-e * (nu + 1) * nu / pow(esp, 2)) * X;
          Vector hbn = ((e / esp) * (1 - (nu + 1) * sigsq / esp)) * X;
        }
      }
    }
    return ans;
  }

  class TrmNuTF {
   public:
    explicit TrmNuTF(TRegressionModel *Mod) : mod(Mod) {}
    TrmNuTF *clone() const { return new TrmNuTF(*this); }
    double operator()(const Vector &Nu) const;
    double operator()(const Vector &Nu, Vector &g) const;

   private:
    double Loglike(const Vector &Nu, Vector &g, uint nd) const;
    TRegressionModel *mod;
  };

  double TrmNuTF::operator()(const Vector &Nu) const {
    Vector g;
    return Loglike(Nu, g, 0);
  }
  double TrmNuTF::operator()(const Vector &Nu, Vector &g) const {
    return Loglike(Nu, g, 1);
  }

  double TrmNuTF::Loglike(const Vector &Nu, Vector &g, uint nd) const {
    const std::vector<Ptr<RegressionData>> &dat(mod->dat());
    uint n = dat.size();
    double nu = Nu[0];

    double nh = .5 * (nu + 1);
    double logsig = log(mod->sigma());
    double lognu = log(nu);
    const double logpi = 1.1447298858494;
    double ans =
        lgamma(nh) - lgamma(nu / 2) + (nh - .5) * lognu - logsig - .5 * logpi;
    ans *= n;

    if (nd > 0) {
      g[0] =
          .5 * digamma(nh) - .5 * digamma(nu / 2) + (nh - .5) / nu + .5 * lognu;
      g[0] *= n;
    }

    for (uint i = 0; i < n; ++i) {
      Ptr<RegressionData> dp = dat[i];
      double err = dp->y() - mod->predict(dp->x());
      double dsq = err * err / mod->sigsq();
      double lnpd = log(nu + dsq);
      ans -= nh * lnpd;
      if (nd > 0) g[0] -= nh / (nu + dsq) + .5 * lnpd;
    }

    return ans;
  }

  void TRegressionModel::mle() {
    const double eps = 1e-5;
    double dloglike = eps + 1;
    double loglike = this->loglike(vectorize_params());
    double old = loglike;
    Vector Nu(1, nu());
    WeightedRegSuf suf(xdim());
    while (dloglike > eps) {
      EStep(suf);
      loglike = MStep(suf);
      dloglike = loglike - old;
      old = loglike;
    }
  }

  void TRegressionModel::EStep(WeightedRegSuf &suf) const {
    suf.clear();
    double nu2 = nu() / 2.0;
    double df2 = nu2 + .5;
    double sigsq2 = sigsq() * 2.0;
    const std::vector<Ptr<RegressionData>> &data(dat());
    for (uint i = 0; i < data.size(); ++i) {
      Ptr<RegressionData> dp = data[i];
      double err = dp->y() - predict(dp->x());
      double ss2 = square(err) / sigsq2 + nu2;
      double w = df2 / ss2;
      suf.add_data(dp->x(), dp->y(), w);
    }
  }

  double TRegressionModel::MStep(const WeightedRegSuf &suf) {
    set_Beta(suf.beta_hat());
    set_sigsq(suf.SSE() / suf.n());
    Vector Nu(1, nu());
    TrmNuTF nu_log_likelihood(this);
    double loglike =
        max_nd1(Nu, Target(nu_log_likelihood), dTarget(nu_log_likelihood));
    set_nu(Nu[0]);
    return loglike;
  }

  double TRegressionModel::pdf(const Ptr<RegressionData> &dp,
                               bool logscale) const {
    double yhat = predict(dp->x());
    return dstudent(dp->y(), yhat, sigma(), nu(), logscale);
  }

  double TRegressionModel::pdf(const Ptr<Data> &dp, bool logscale) const {
    return pdf(dp.dcast<DataType>(), logscale);
  }

  Ptr<RegressionData> TRegressionModel::sim(RNG &rng) const {
    uint p = Beta().size();
    Vector x(p);
    for (uint i = 0; i < p; ++i) x[i] = rnorm_mt(rng);
    return sim(x, rng);
  }

  Ptr<RegressionData> TRegressionModel::sim(const Vector &x, RNG &rng) const {
    double nu = this->nu();
    double w = rgamma_mt(rng, nu / 2, nu / 2);
    double yhat = predict(x);
    double z = rnorm_mt(rng, 0, sigma() / sqrt(w));
    double y = yhat + z;
    NEW(RegressionData, d)(y, x);
    return d;
  }

}  // namespace BOOM
