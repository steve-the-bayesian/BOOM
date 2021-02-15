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
#include "Models/Glm/MvtRegModel.hpp"
#include "LinAlg/QR.hpp"
#include "cpputil/nyi.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef MvtRegModel MVTR;

  namespace {
    double default_df(30.0);
  }

  MVTR::MvtRegModel(uint xdim, uint ydim)
      : ParamPolicy(new MatrixGlmCoefs(xdim, ydim),
                    new SpdParams(ydim),
                    new UnivParams(default_df)) {}

  MVTR::MvtRegModel(const Matrix &X, const Matrix &Y, bool add_intercept)
      : ParamPolicy(new MatrixGlmCoefs(X.ncol() + add_intercept, Y.ncol()),
                    new SpdParams(Y.ncol()),
                    new UnivParams(default_df)) {
    Matrix XX(add_intercept ? cbind(1.0, X) : X);
    QR qr(XX);
    Matrix Beta(qr.solve(qr.QtY(Y)));
    Matrix resid = Y - XX * Beta;
    uint n = XX.nrow();
    SpdMatrix Sig = resid.transpose() * resid / n;

    set_Beta(Beta);
    set_Sigma(Sig);

    for (uint i = 0; i < n; ++i) {
      Vector y = Y.row(i);
      Vector x = XX.row(i);
      NEW(MvRegData, dp)(y, x);
      DataPolicy::add_data(dp);
    }
  }

  MVTR::MvtRegModel(const Matrix &B, const SpdMatrix &Sigma, double nu)
      : ParamPolicy(new MatrixGlmCoefs(B),
                    new SpdParams(Sigma),
                    new UnivParams(nu)) {}

  MVTR::MvtRegModel(const MvtRegModel &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        LoglikeModel(rhs) {}

  MVTR *MVTR::clone() const { return new MVTR(*this); }

  uint MVTR::xdim() const { return Beta().nrow(); }
  uint MVTR::ydim() const { return Beta().ncol(); }

  const Matrix &MVTR::Beta() const { return Beta_prm()->value(); }
  const SpdMatrix &MVTR::Sigma() const { return Sigma_prm()->var(); }
  const SpdMatrix &MVTR::Siginv() const { return Sigma_prm()->ivar(); }
  double MVTR::ldsi() const { return Sigma_prm()->ldsi(); }
  double MVTR::nu() const { return Nu_prm()->value(); }

  Ptr<MatrixGlmCoefs> MVTR::Beta_prm() { return ParamPolicy::prm1(); }
  Ptr<SpdParams> MVTR::Sigma_prm() { return ParamPolicy::prm2(); }
  Ptr<UnivParams> MVTR::Nu_prm() { return ParamPolicy::prm3(); }
  const Ptr<MatrixGlmCoefs> MVTR::Beta_prm() const {return ParamPolicy::prm1();}
  const Ptr<SpdParams> MVTR::Sigma_prm() const { return ParamPolicy::prm2(); }
  const Ptr<UnivParams> MVTR::Nu_prm() const { return ParamPolicy::prm3(); }

  void MVTR::set_Beta(const Matrix &B) { Beta_prm()->set(B); }
  void MVTR::set_Sigma(const SpdMatrix &V) { Sigma_prm()->set_var(V); }
  void MVTR::set_Siginv(const SpdMatrix &iV) { Sigma_prm()->set_ivar(iV); }
  void MVTR::set_nu(double new_nu) { Nu_prm()->set(new_nu); }

  void MVTR::mle() {  // ECME
    nyi("MvtRegModel::mle");
  }

  double MVTR::loglike(const Vector &beta_columns_siginv_triangle_nu) const {
    Matrix Beta(xdim(), ydim());
    SpdMatrix siginv(ydim());
    Vector::const_iterator it = beta_columns_siginv_triangle_nu.cbegin();
    std::copy(it, it + Beta.size(), Beta.begin());
    it += Beta.size();
    siginv.unvectorize(it, true);
    double ldsi = siginv.logdet();
    double nu = beta_columns_siginv_triangle_nu.back();

    const DatasetType &d(dat());
    uint n = d.size();
    double ans = 0;
    for (uint i = 0; i < n; ++i) {
      Vector mu = d[i]->x() * Beta;
      ans += dmvt(d[i]->y(), mu, siginv, nu, ldsi, true);
    }
    return ans;
  }

  double MVTR::pdf(const Ptr<Data> &dp, bool logscale) const {
    Ptr<DataType> d = DAT(dp);
    const Vector &y(d->y());
    const Vector &X(d->x());
    double ans = dmvt(y, X * Beta(), Siginv(), nu(), ldsi(), true);
    return logscale ? ans : exp(ans);
  }

  Vector MVTR::predict(const Vector &x) const { return x * Beta(); }

  MvRegData *MVTR::sim(RNG &rng) const {
    Vector x = simulate_fake_x(rng);
    return this->sim(x, rng);
  }

  MvRegData *MVTR::sim(const Vector &x, RNG &rng) const {
    Vector Y = rmvt_mt(rng, predict(x), Sigma(), nu());
    return new MvRegData(Y, x);
  }

  Vector MVTR::simulate_fake_x(RNG &rng) const {
    uint p = xdim();
    Vector x(p);
    x[0] = 1.0;
    for (uint i = 0; i < p; ++i) x[i] = rnorm_mt(rng);
    return x;
  }
}  // namespace BOOM
