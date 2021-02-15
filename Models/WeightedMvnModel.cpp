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

#include "Models/WeightedMvnModel.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"

#include <cmath>
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef WeightedMvnSuf WMS;
  typedef WeightedVectorData WVD;

  WMS::WeightedMvnSuf(uint p)
      : sum_(p, 0.0), sumsq_(p, 0.0), n_(0), sumw_(0), sumlogw_(0) {}

  WMS::WeightedMvnSuf(const WMS &rhs)
      : Sufstat(rhs),
        SufstatDetails<WeightedVectorData>(rhs),
        sum_(rhs.sum_),
        sumsq_(rhs.sumsq_),
        n_(rhs.n_),
        sumw_(rhs.sumw_),
        sumlogw_(rhs.sumlogw_) {}

  WMS *WMS::clone() const { return new WMS(*this); }

  void WMS::clear() {
    sumsq_ = 0;
    sum_ = 0;
    n_ = sumw_ = sumlogw_ = 0;
  }

  void WMS::Update(const WVD &rhs) {
    double w = rhs.weight();
    const Vector &x(rhs.value());
    sum_.axpy(x, w);
    sumsq_.add_outer(x, w);
    ++n_;
    sumw_ += w;
    sumlogw_ += log(w);
  }

  const Vector &WMS::sum() const { return sum_; }
  const SpdMatrix &WMS::sumsq() const { return sumsq_; }
  double WMS::n() const { return n_; }
  double WMS::sumw() const { return sumw_; }
  double WMS::sumlogw() const { return sumlogw_; }

  Vector WMS::ybar() const {
    if (sumw() == 0) return sum().zero();
    return sum() / sumw();
  }

  SpdMatrix WMS::var_hat() const {
    if (sumw() == 0) return sumsq() * 0.0;
    return center_sumsq() / sumw();
  }

  SpdMatrix WMS::center_sumsq(const Vector &mu) const {
    SpdMatrix ans = sumsq();    // sum wyy^T
    ans.add_outer(mu, sumw());  // wyyT + w.mu.muT

    ans -= as_symmetric(mu.outer(sum_, 2));
    return ans;
  }

  SpdMatrix WMS::center_sumsq() const { return center_sumsq(ybar()); }

  void WMS::combine(const Ptr<WMS> &s) {
    sum_ += s->sum_;
    sumsq_ += s->sumsq_;
    n_ += s->n_;
    sumw_ += s->sumw_;
    sumlogw_ += s->sumlogw_;
  }

  void WMS::combine(const WMS &s) {
    sum_ += s.sum_;
    sumsq_ += s.sumsq_;
    n_ += s.n_;
    sumw_ += s.sumw_;
    sumlogw_ += s.sumlogw_;
  }

  WeightedMvnSuf *WMS::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  Vector WMS::vectorize(bool minimal) const {
    Vector ans = sum_;
    ans.concat(sumsq_.vectorize(minimal));
    ans.push_back(n_);
    ans.push_back(sumw_);
    ans.push_back(sumlogw_);
    return ans;
  }

  Vector::const_iterator WMS::unvectorize(Vector::const_iterator &v,
                                          bool minimal) {
    uint dim = sum_.size();
    sum_.assign(v, v + dim);
    v += dim;
    sumsq_.unvectorize(v, minimal);
    n_ = *v;
    ++v;
    sumw_ = *v;
    ++v;
    sumlogw_ = *v;
    ++v;
    return v;
  }

  Vector::const_iterator WMS::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  std::ostream &WMS::print(std::ostream &out) const {
    return out << "sum_ = " << sum_ << endl
               << "n_ = " << n_ << endl
               << "sumw_ = " << sumw_ << endl
               << "sumlogw_ = " << sumlogw_ << endl
               << "sumsq_ = " << endl
               << sumsq_;
  }

  //======================================================================

  WeightedMvnModel::WeightedMvnModel(uint p, double mu, double sigma)
      : ParamPolicy(new VectorParams(Vector(p, mu)),
                    new SpdParams(Id(p) * (sigma * sigma))),
        DataPolicy(new WMS(p)),
        PriorPolicy() {}

  WeightedMvnModel::WeightedMvnModel(const Vector &mean, const SpdMatrix &Var)
      : ParamPolicy(new VectorParams(mean), new SpdParams(Var)),
        DataPolicy(new WMS(mean.size())),
        PriorPolicy() {}

  WeightedMvnModel::WeightedMvnModel(const WeightedMvnModel &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        LoglikeModel(rhs) {}

  WeightedMvnModel *WeightedMvnModel::clone() const {
    return new WeightedMvnModel(*this);
  }

  Ptr<VectorParams> WeightedMvnModel::Mu_prm() { return ParamPolicy::prm1(); }
  const Ptr<VectorParams> WeightedMvnModel::Mu_prm() const {
    return ParamPolicy::prm1();
  }

  Ptr<SpdParams> WeightedMvnModel::Sigma_prm() { return ParamPolicy::prm2(); }
  const Ptr<SpdParams> WeightedMvnModel::Sigma_prm() const {
    return ParamPolicy::prm2();
  }

  const Vector &WeightedMvnModel::mu() const { return Mu_prm()->value(); }
  const SpdMatrix &WeightedMvnModel::Sigma() const {
    return Sigma_prm()->var();
  }
  const SpdMatrix &WeightedMvnModel::siginv() const {
    return Sigma_prm()->ivar();
  }
  double WeightedMvnModel::ldsi() const { return Sigma_prm()->ldsi(); }

  void WeightedMvnModel::set_mu(const Vector &v) { Mu_prm()->set(v); }
  void WeightedMvnModel::set_Sigma(const SpdMatrix &s) {
    Sigma_prm()->set_var(s);
  }
  void WeightedMvnModel::set_siginv(const SpdMatrix &ivar) {
    Sigma_prm()->set_ivar(ivar);
  }

  void WeightedMvnModel::mle() {
    set_mu(suf()->ybar());
    set_Sigma(suf()->var_hat());
  }

  double WeightedMvnModel::loglike(const Vector &mu_siginv_triangle) const {
    const double log2pi = 1.83787706641;
    const ConstVectorView mu(mu_siginv_triangle, 0, dim());
    SpdMatrix siginv(dim());
    Vector::const_iterator it = mu_siginv_triangle.begin() + dim();
    siginv.unvectorize(it, true);
    double ldsi = siginv.logdet();

    double sumlogw = suf()->sumlogw();
    double n = suf()->n();

    double ans = n * .5 * (log2pi + ldsi) + dim() * .5 * sumlogw;
    ans -= -.5 * traceAB(siginv, suf()->center_sumsq(mu));
    return ans;
  }

  double WeightedMvnModel::pdf(const Ptr<WeightedVectorData> &dp,
                               bool logscale) const {
    double w = dp->weight();
    const Vector &y(dp->value());
    uint p = mu().size();
    double wldsi = p * log(w) + ldsi();
    return dmvn(y, mu(), w * siginv(), wldsi, logscale);
  }

  double WeightedMvnModel::pdf(const Ptr<Data> &dp, bool logscale) const {
    return pdf(DAT(dp), logscale);
  }

}  // namespace BOOM
