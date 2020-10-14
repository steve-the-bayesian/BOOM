// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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
#include "Models/Glm/MultivariateRegression.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "LinAlg/Cholesky.hpp"
#include "cpputil/Constants.hpp"
#include "distributions.hpp"

namespace BOOM {

  //======================================================================

  MvRegSuf::MvRegSuf(uint xdim, uint ydim)
      : yty_(ydim), xtx_(xdim), xty_(xdim, ydim), n_(0) {}


  MvRegSuf::MvRegSuf(const Matrix &X, const Matrix &Y)
      : yty_(Y.ncol()), xtx_(X.ncol()), xty_(X.ncol(), Y.ncol()), n_(0) {
    QR qr(X);
    Matrix R = qr.getR();
    xtx_.add_inner(R);

    QR qry(Y);
    yty_.add_inner(qry.getR());

    xty_ = qr.getQ().Tmult(Y);
    xty_ = R.Tmult(xty_);
  }

  MvRegSuf::MvRegSuf(const MvRegSuf &rhs)
      : Sufstat(rhs),
        SufTraits(rhs),
        yty_(rhs.yty_),
        xtx_(rhs.xtx_),
        xty_(rhs.xty_),
        n_(rhs.n_) {}

  MvRegSuf *MvRegSuf::clone() const { return new MvRegSuf(*this); }

  void MvRegSuf::Update(const MvRegData &d) {
    const Vector &y(d.y());
    const Vector &x(d.x());
    double w = d.weight();
    update_raw_data(y, x, w);
  }

  void MvRegSuf::update_raw_data(const Vector &y, const Vector &x, double w) {
    ++n_;
    sumw_ += w;
    xtx_.add_outer(x, w);
    xty_.add_outer(x, y, w);
    yty_.add_outer(y, w);
  }

  void MvRegSuf::clear_y_keep_x() {
    sumw_ = 0;
    xty_ = 0;
    yty_ = 0;
  }

  void MvRegSuf::update_y_not_x(const Vector &y, const Vector &x, double w) {
    sumw_ += w;
    xty_.add_outer(x, y, w);
    yty_.add_outer(y, w);
  }

  Matrix MvRegSuf::beta_hat() const { return xtx_.solve(xty_); }
  Matrix MvRegSuf::conditional_beta_hat(const SelectorMatrix &included) const {
    Matrix ans(xdim(), ydim());
    std::map<Selector, Cholesky> chol_map;
    for (int i = 0; i < ydim(); ++i) {
      const Selector &inc(included.col(i));
      auto it = chol_map.find(inc);
      if (it == chol_map.end()) {
        chol_map[it->first] = Cholesky(inc.select(xtx()));
        it = chol_map.find(inc);
      }
      ans.col(i) = inc.expand(it->second.solve(inc.select(xty_.col(i))));
    }
    return ans;
  }

  SpdMatrix MvRegSuf::SSE(const Matrix &B) const {
    SpdMatrix ans = yty();
    ans.add_inner2(B, xty(), -1);
    ans += sandwich(B.transpose(), xtx());
    return ans;
  }

  void MvRegSuf::clear() {
    yty_ = 0;
    xtx_ = 0;
    xty_ = 0;
    n_ = 0;
  }

  const SpdMatrix &MvRegSuf::yty() const { return yty_; }
  const SpdMatrix &MvRegSuf::xtx() const { return xtx_; }
  const Matrix &MvRegSuf::xty() const { return xty_; }
  double MvRegSuf::n() const { return n_; }
  double MvRegSuf::sumw() const { return sumw_; }

  void MvRegSuf::combine(const Ptr<MvRegSuf> &sp) {
    Ptr<MvRegSuf> s(sp.dcast<MvRegSuf>());
    xty_ += s->xty_;
    xtx_ += s->xtx_;
    yty_ += s->yty_;
    sumw_ += s->sumw_;
    n_ += s->n_;
  }

  void MvRegSuf::combine(const MvRegSuf &sp) {
    const MvRegSuf &s(dynamic_cast<const MvRegSuf &>(sp));
    xty_ += s.xty_;
    xtx_ += s.xtx_;
    yty_ += s.yty_;
    sumw_ += s.sumw_;
    n_ += s.n_;
  }

  Vector MvRegSuf::vectorize(bool minimal) const {
    Vector ans = yty_.vectorize(minimal);
    ans.concat(xtx_.vectorize(minimal));
    Vector tmp(xty_.begin(), xty_.end());
    ans.concat(tmp);
    ans.push_back(sumw_);
    ans.push_back(n_);
    return ans;
  }

  MvRegSuf *MvRegSuf::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  Vector::const_iterator MvRegSuf::unvectorize(Vector::const_iterator &v,
                                         bool minimal) {
    yty_.unvectorize(v, minimal);
    xtx_.unvectorize(v, minimal);
    uint xdim = xtx_.nrow();
    uint ydim = yty_.nrow();
    Matrix tmp(v, v + xdim * ydim, xdim, ydim);
    v += xdim * ydim;
    sumw_ = *v;
    ++v;
    n_ = *v;
    ++v;
    return v;
  }

  Vector::const_iterator MvRegSuf::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  std::ostream &MvRegSuf::print(std::ostream &out) const {
    out << "yty_ = " << yty_ << endl
        << "xty_ = " << xty_ << endl
        << "xtx_ = " << endl
        << xtx_;
    return out;
  }

  //======================================================================

  namespace {
    using MvReg = MultivariateRegressionModel;
  }

  MvReg::MultivariateRegressionModel(uint xdim, uint ydim)
      : ParamPolicy(new MatrixGlmCoefs(xdim, ydim), new SpdParams(ydim)),
        DataPolicy(new MvRegSuf(xdim, ydim)),
        PriorPolicy(),
        LoglikeModel() {}

  MvReg::MultivariateRegressionModel(const Matrix &X, const Matrix &Y)
      : ParamPolicy(),
        DataPolicy(new MvRegSuf(X, Y)),
        PriorPolicy(),
        LoglikeModel() {
    uint nx = X.ncol();
    uint ny = Y.ncol();
    set_params(new MatrixGlmCoefs(nx, ny), new SpdParams(ny));
    mle();
  }

  MvReg::MultivariateRegressionModel(const Matrix &B, const SpdMatrix &V)
      : ParamPolicy(new MatrixGlmCoefs(B), new SpdParams(V)),
        DataPolicy(new MvRegSuf(B.nrow(), B.ncol())),
        PriorPolicy(),
        LoglikeModel() {}

  MvReg *MvReg::clone() const { return new MvReg(*this); }

  uint MvReg::xdim() const { return Beta().nrow(); }
  uint MvReg::ydim() const { return Beta().ncol(); }

  const Matrix &MvReg::Beta() const { return Beta_prm()->value(); }
  const SpdMatrix &MvReg::Sigma() const { return Sigma_prm()->var(); }
  const SpdMatrix &MvReg::Siginv() const { return Sigma_prm()->ivar(); }
  Matrix MvReg::residual_precision_cholesky() const {
    return Sigma_prm()->ivar_chol();
  }
  double MvReg::ldsi() const { return Sigma_prm()->ldsi(); }

  Ptr<MatrixGlmCoefs> MvReg::Beta_prm() { return prm1(); }
  const Ptr<MatrixGlmCoefs> MvReg::Beta_prm() const { return prm1(); }
  Ptr<SpdParams> MvReg::Sigma_prm() { return prm2(); }
  const Ptr<SpdParams> MvReg::Sigma_prm() const { return prm2(); }

  void MvReg::set_Beta(const Matrix &B) {
    if (B.nrow() != xdim()) {
      report_error("Matrix passed to set_Beta has the wrong number of rows.");
    }
    if (B.ncol() != ydim()) {
      report_error("Matrix passed to set_Beta has the wrong number "
                   "of columns.");
    }
    Beta_prm()->set(B);
  }

  void MvReg::set_Sigma(const SpdMatrix &V) {
    if (V.nrow() != ydim()) {
      report_error("Wrong size variance matrix passed to set_Sigma.");
    }
    Sigma_prm()->set_var(V);
  }

  void MvReg::set_Siginv(const SpdMatrix &iV) {
    if (iV.nrow() != ydim()) {
      report_error("Wrong size precision matrix passed to set_Siginv.");
    }
    Sigma_prm()->set_ivar(iV);
  }

  void MvReg::mle() {
    set_Beta(suf()->beta_hat());
    set_Sigma(suf()->SSE(Beta()) / suf()->n());
  }

  double MvReg::log_likelihood(const Matrix &Beta,
                               const SpdMatrix &Sigma) const {
    Cholesky Sigma_cholesky(Sigma);
    double qform = trace(suf()->SSE(Beta) * Sigma_cholesky.inv());
    double ldsi = -1 * Sigma_cholesky.logdet();
    double n = suf()->n();
    double normalizing_constant = -.5 * (n * ydim()) * Constants::log_2pi;
    return normalizing_constant + .5 * n * ldsi - .5 * qform;
  }

  // The likelihood is \prod root(2pi)^-d |siginv|^{n/2} exp{-1/2 * trace(qform)}
  double MvReg::log_likelihood_ivar(const Matrix &Beta,
                                    const SpdMatrix &Siginv) const {
    double qform = trace(suf()->SSE(Beta) * Siginv);
    double n = suf()->n();
    double normalizing_constant = -.5 * (n * ydim()) * Constants::log_2pi;
    return normalizing_constant + .5 * n * Siginv.logdet() - .5 * qform;
  }

  double MvReg::loglike(const Vector &beta_siginv) const {
    Matrix Beta(xdim(), ydim());
    Vector::const_iterator it = beta_siginv.cbegin();
    std::copy(it, it + Beta.size(), Beta.begin());
    it += Beta.size();
    SpdMatrix siginv(ydim());
    siginv.unvectorize(it, true);
    return log_likelihood_ivar(Beta, siginv);
  }

  double MvReg::log_likelihood() const {
    return log_likelihood_ivar(Beta(), Siginv());
  }

  double MvReg::pdf(const Ptr<Data> &dptr, bool logscale) const {
    Ptr<MvRegData> dp = DAT(dptr);
    Vector mu = predict(dp->x());
    return dmvn(dp->y(), mu, Siginv(), ldsi(), logscale);
  }

  Vector MvReg::predict(const Vector &x) const { return x * Beta(); }

  MvRegData *MvReg::sim(RNG &rng) const {
    Vector x = simulate_fake_x(rng);
    return sim(x, rng);
  }

  MvRegData *MvReg::sim(const Vector &x, RNG &rng) const {
    Vector mu = predict(x);
    Vector y = rmvn_mt(rng, mu, Sigma());
    return new MvRegData(y, x);
  }

  Vector MvReg::simulate_fake_x(RNG &rng) const {
    uint p = xdim();
    Vector x(p, 1.0);
    for (uint i = 1; i < p; ++i) x[i] = rnorm_mt(rng);
    return x;
  }

}  // namespace BOOM
