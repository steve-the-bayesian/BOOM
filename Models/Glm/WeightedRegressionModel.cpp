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

#include "Models/Glm/WeightedRegressionModel.hpp"
#include "distributions.hpp"

#include <cmath>
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {
  typedef WeightedRegressionData WRD;

  //============================================================
  typedef WeightedRegSuf WRS;

  WRS::WeightedRegSuf(int p) {
    setup_mat(p);
    clear();
  }

  WRS::WeightedRegSuf(const Matrix &X, const Vector &y, const Vector &w) {
    Matrix tmpx = add_intercept(X);
    uint p = tmpx.nrow();
    setup_mat(p);
    if (w.empty()) {
      recompute(tmpx, y, Vector(y.size(), 1.0));
    } else {
      recompute(tmpx, y, w);
    }
  }

  WRS::WeightedRegSuf(const std::vector<Ptr<WeightedRegressionData>> &data) {
    uint xdim = data.front()->xdim();
    setup_mat(xdim);
    recompute(data);
  }

  WRS *WRS::clone() const { return new WRS(*this); }

  std::ostream &WRS::print(std::ostream &out) const {
    out << "xtwx_   = " << endl
        << xtx() << endl
        << "xtwy_   = " << xtwy_ << endl
        << "n_      = " << n_ << endl
        << "yt_w_y_ = " << yt_w_y_ << endl
        << "sumw_   = " << sumw_ << endl
        << "sumlogw_= " << sumlogw_ << endl;
    return out;
  }

  void WRS::combine(const Ptr<WRS> &s) {
    xtwx_ += s->xtwx_;
    xtwy_ += s->xtwy_;
    n_ += s->n_;
    yt_w_y_ += s->yt_w_y_;
    sumw_ += s->sumw_;
    sumlogw_ += s->sumlogw_;
    sym_ = sym_ && s->sym_;
  }

  void WRS::combine(const WRS &s) {
    xtwx_ += s.xtwx_;
    xtwy_ += s.xtwy_;
    n_ += s.n_;
    yt_w_y_ += s.yt_w_y_;
    sumw_ += s.sumw_;
    sumlogw_ += s.sumlogw_;
    sym_ = sym_ && s.sym_;
  }

  WeightedRegSuf *WRS::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  Vector WRS::vectorize(bool minimal) const {
    Vector ans = xtwx_.vectorize(minimal);
    ans.concat(xtwy_);
    ans.push_back(n_);
    ans.push_back(yt_w_y_);
    ans.push_back(sumw_);
    ans.push_back(sumlogw_);
    return ans;
  }

  Vector::const_iterator WRS::unvectorize(Vector::const_iterator &v, bool) {
    xtwx_.unvectorize(v);
    uint dim = xtwy_.size();
    xtwy_.assign(v, v + dim);
    v += dim;
    n_ = *v;
    ++v;
    yt_w_y_ = *v;
    ++v;
    sumw_ = *v;
    ++v;
    sumlogw_ = *v;
    ++v;
    return v;
  }

  Vector::const_iterator WRS::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  //------------------------------------------------------------
  void WRS::setup_mat(uint p) {
    xtwx_ = SpdMatrix(p, 0.0);
    xtwy_ = Vector(p, 0.0);
    sym_ = false;
  }

  void WRS::recompute(const Matrix &X, const Vector &y, const Vector &w) {
    uint n = w.size();
    assert(y.size() == n && X.nrow() == n);
    clear();
    for (uint i = 0; i < n; ++i) add_data(X.row(i), y[i], w[i]);
  }

  void WRS::recompute(const std::vector<Ptr<WeightedRegressionData>> &data) {
    clear();
    for (uint i = 0; i < data.size(); ++i) update(data[i]);
  }

  //------------------------------------------------------------
  void WRS::set_xtwx(const SpdMatrix &xtwx) { xtwx_ = xtwx; }

  void WRS::set_xtwy(const Vector &xtwy) { xtwy_ = xtwy; }

  void WRS::reset(const SpdMatrix &xtwx, const Vector &xtwy, double ytwy,
                  double sample_size, double sum_weights, double sum_log_weights) {
    xtwx_ = xtwx;
    xtwy_ = xtwy;
    n_ = sample_size;
    yt_w_y_ = ytwy;
    sumw_ = sum_weights;
    sumlogw_ = sum_log_weights;
    sym_ = true;
  }
  
  //------------------------------------------------------------

  void WRS::add_data(const Vector &x, double y, double w) {
    ++n_;
    yt_w_y_ += w * y * y;
    sumw_ += w;
    sumlogw_ += log(w);
    xtwx_.add_outer(x, w, false);
    xtwy_.axpy(x, w * y);
    sym_ = false;
  }

  void WRS::clear() {
    xtwx_ = 0.0;
    xtwy_ = 0.0;
    sumw_ = yt_w_y_ = n_ = sumlogw_ = 0.0;
    sym_ = false;
  }

  void WRS::Update(const WRD &d) { add_data(d.x(), d.y(), d.weight()); }

  //------------------------------------------------------------
  uint WRS::size() const { return xtwx_.nrow(); }
  double WRS::yty() const { return yt_w_y_; }
  Vector WRS::xty() const { return xtwy_; }
  SpdMatrix WRS::xtx() const {
    if (!sym_) make_symmetric();
    return xtwx_;
  }
  void WRS::make_symmetric() const {
    xtwx_.reflect();
    sym_ = true;
  }

  Vector WRS::xty(const Selector &inc) const { return inc.select(xtwy_); }
  SpdMatrix WRS::xtx(const Selector &inc) const { return inc.select(xtx()); }

  Vector WRS::beta_hat() const { return xtx().solve(xtwy_); }

  double WRS::weighted_sum_of_squared_errors(const Vector &beta) const {
    return xtx().Mdist(beta) - 2 * beta.dot(xty()) + yty();
  }

  double WRS::SSE() const {
    SpdMatrix ivar = xtx().inv();
    return yty() - ivar.Mdist(xty());
  }

  double WRS::SST() const { return yty() / sumw() - pow(ybar(), 2); }
  double WRS::n() const { return n_; }
  double WRS::sumw() const { return sumw_;}
  double WRS::sumlogw() const { return sumlogw_; }
  double WRS::ybar() const { return xtwy_[0] / sumw(); }

  //----------------------------------------------------------------------

  typedef WeightedRegressionModel WRM;

  WRM::WeightedRegressionModel(uint p)
      : ParamPolicy(new GlmCoefs(p), new UnivParams(1.0)),
        DataPolicy(new WRS(p)) {}

  WRM::WeightedRegressionModel(const Vector &b, double Sigma)
      : ParamPolicy(new GlmCoefs(b), new UnivParams(pow(Sigma, 2))),
        DataPolicy(new WRS(b.size())) {}

  namespace {
    std::vector<Ptr<WRD> > make_data(const Matrix &X, const Vector &y,
                                     const Vector &w) {
      std::vector<Ptr<WRD> > ans;
      for (uint i = 0; i < X.nrow(); ++i) {
        ans.push_back(new WeightedRegressionData(y[i], X.row(i), w[i]));
      }
      return ans;
    }
  }  // namespace

  WRM::WeightedRegressionModel(const Matrix &X, const Vector &y)
      : ParamPolicy(new GlmCoefs(X.ncol()), new UnivParams(1.0)),
        DataPolicy(new WRS(X.ncol()), make_data(X, y, Vector(y.size(), 1.0))) {
    mle();
  }

  WRM::WeightedRegressionModel(const Matrix &X, const Vector &y,
                               const Vector &w)
      : ParamPolicy(new GlmCoefs(X.ncol()), new UnivParams(1.0)),
        DataPolicy(new WRS(X.ncol()), make_data(X, y, w)) {
    mle();
  }

  WRM::WeightedRegressionModel(const DatasetType &d, bool all)
      : ParamPolicy(new GlmCoefs(d[0]->xdim(), all), new UnivParams(1.0)),
        DataPolicy(new WRS(d[0]->xdim()), d) {
    mle();
  }

  WRM::WeightedRegressionModel(const WeightedRegressionModel &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        GlmModel(rhs),
        NumOptModel(rhs) {}

  WeightedRegressionModel *WRM::clone() const { return new WRM(*this); }

  double WRM::pdf(const Ptr<Data> &dp, bool logscale) const {
    Ptr<DataType> d = dp.dcast<DataType>();
    return pdf(d, logscale);
  }

  double WRM::pdf(const Ptr<WeightedRegressionData> &dp, bool logscale) const {
    double mu = predict(dp->x());
    double sigsq = this->sigsq();
    double w = dp->weight();
    return dnorm(dp->y(), mu, sqrt(sigsq / w), logscale);
  }

  GlmCoefs &WRM::coef() { return ParamPolicy::prm1_ref(); }
  const GlmCoefs &WRM::coef() const { return ParamPolicy::prm1_ref(); }
  Ptr<GlmCoefs> WRM::coef_prm() { return ParamPolicy::prm1(); }
  const Ptr<GlmCoefs> WRM::coef_prm() const { return ParamPolicy::prm1(); }

  void WRM::set_sigsq(double s2) { ParamPolicy::prm2_ref().set(s2); }

  Ptr<UnivParams> WRM::Sigsq_prm() { return ParamPolicy::prm2(); }
  const Ptr<UnivParams> WRM::Sigsq_prm() const { return ParamPolicy::prm2(); }
  const double &WRM::sigsq() const { return ParamPolicy::prm2_ref().value(); }
  double WRM::sigma() const { return sqrt(sigsq()); }

  void WRM::mle() {
    SpdMatrix xtx(suf()->xtx(coef().inc()));
    Vector xty(suf()->xty(coef().inc()));
    Vector b = xtx.solve(xty);
    set_included_coefficients(b);

    double SSE = suf()->yty() - 2 * b.dot(xty) + xtx.Mdist(b);
    double n = suf()->n();
    set_sigsq(SSE / n);
  }

  double WRM::Loglike(const Vector &beta_sigsq, Vector &g, Matrix &h,
                      uint nd) const {
    const double log2pi = 1.8378770664093453;
    const Selector &inclusion_indicators(coef().inc());
    const int beta_dim(inclusion_indicators.nvars());
    const Vector beta(ConstVectorView(beta_sigsq, 0, beta_dim));
    const double sigsq = beta_sigsq.back();

    if (sigsq <= 0) {
      g = 0;
      g.back() = -sigsq;
      h = h.Id();
      return BOOM::negative_infinity();
    }

    SpdMatrix xtwx(suf()->xtx(inclusion_indicators));
    Vector xtwy(suf()->xty(inclusion_indicators));
    double ytwy = suf()->yty();
    double n = suf()->n();
    double sumlogw = suf()->sumlogw();

    double SS = xtwx.Mdist(beta) - 2 * beta.dot(xtwy) + ytwy;
    double ans = -.5 * (n * log2pi + n * log(sigsq) - sumlogw + SS / sigsq);

    if (nd > 0) {
      double siginv = 1.0 / sigsq;
      Vector gb = xtwx * beta;
      gb -= xtwy;
      gb *= -siginv;

      double isig4 = siginv * siginv;
      double gs2 = -n / 2 * siginv + SS / 2 * isig4;
      g = concat(gb, gs2);

      if (nd > 1) {
        Matrix hb = -siginv * xtwx;
        double hs2 = n / 2 * isig4 - SS * isig4 * siginv;
        h = block_diagonal(hb, Matrix(1, 1, hs2));
      }
    }

    return ans;
  }

}  // namespace BOOM
