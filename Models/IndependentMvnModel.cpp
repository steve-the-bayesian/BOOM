// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include "Models/IndependentMvnModel.hpp"
#include <limits>
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    inline double square(double x) { return x * x; }
  }  // namespace

  IndependentMvnSuf::IndependentMvnSuf(int dim) : suf_(dim) {}

  IndependentMvnSuf *IndependentMvnSuf::clone() const {
    return new IndependentMvnSuf(*this);
  }

  void IndependentMvnSuf::clear() {
    for (auto &s : suf_) s.clear();
  }

  void IndependentMvnSuf::resize(int dim) {
    suf_.resize(dim);
    clear();
  }

  void IndependentMvnSuf::Update(const VectorData &d) { update_raw(d.value()); }

  void IndependentMvnSuf::update_single_dimension(double y, int position) {
    suf_[position].update_raw(y);
  }
  
  void IndependentMvnSuf::update_raw(const Vector &y) {
    for (int i = 0; i < y.size(); ++i) {
      suf_[i].update_raw(y[i]);
    }
  }

  void IndependentMvnSuf::add_mixture_data(const Vector &v, double prob) {
    for (int i = 0; i < v.size(); ++i) {
      suf_[i].add_mixture_data(v[i], prob);
    }
  }

  void IndependentMvnSuf::update_expected_value(
      double sample_size, const Vector &expected_sum,
      const Vector &expected_sum_of_squares) {
    for (int i = 0; i < expected_sum.size(); ++i) {
      suf_[i].update_expected_value(
          sample_size, expected_sum[i], expected_sum_of_squares[i]);
    }
  }

  double IndependentMvnSuf::sum(int i) const { return suf_[i].sum();}
  double IndependentMvnSuf::sumsq(int i) const { return suf_[i].sumsq();}

  double IndependentMvnSuf::centered_sumsq(int i, double mu) const {
    return sumsq(i) - 2 * mu * sum(i) + suf_[i].n() * square(mu);
  }

  double IndependentMvnSuf::n(int i) const {
    return suf_[i].n();
  }

  double IndependentMvnSuf::ybar(int i) const {
    double ni = n(i);
    if (ni < 1e-7) {
      return 0;
    }
    return sum(i) / ni;
  }

  double IndependentMvnSuf::sample_var(int i) const {
    double ni = n(i);
    if (ni - 1 < std::numeric_limits<double>::epsilon()) {
      return 0;
    }
    double ybari = ybar(i);
    double ss = sumsq(i) - ni * ybari * ybari;
    return ss / (ni - 1);
  }

  IndependentMvnSuf *IndependentMvnSuf::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  void IndependentMvnSuf::combine(const Ptr<IndependentMvnSuf> &s) {
    return this->combine(*s);
  }

  void IndependentMvnSuf::combine(const IndependentMvnSuf &s) {
    for (int i = 0; i < suf_.size(); ++i) {
      suf_[i].combine(s.suf_[i]);
    }
  }

  Vector IndependentMvnSuf::vectorize(bool) const {
    Vector ans;
    ans.reserve(3 * suf_.size());
    for (int i = 0; i < suf_.size(); ++i) {
      ans.concat(suf_[i].vectorize());
    }
    return (ans);
  }

  Vector::const_iterator IndependentMvnSuf::unvectorize(
      Vector::const_iterator &v, bool) {
    for (int i = 0; i < suf_.size(); ++i) {
      v = suf_[i].unvectorize(v);
    }
    return v;
  }

  Vector::const_iterator IndependentMvnSuf::unvectorize(
      const Vector &v, bool minimal) {
    Vector::const_iterator vi = v.begin();
    return unvectorize(vi, minimal);
  }

  std::ostream &IndependentMvnSuf::print(std::ostream &out) const {
    Matrix tmp(suf_.size(), 3);
    for (int i = 0; i < suf_.size(); ++i) {
      tmp(i, 0) = n(i);
      tmp(i, 1) = sum(i);
      tmp(i, 2) = sumsq(i);
    }
    out << tmp;
    return out;
  }

  //======================================================================

  IndependentMvnBase::IndependentMvnBase(int dim)
      : DataPolicy(new IndependentMvnSuf(dim)),
        sigma_scratch_(dim, 0.0),
        g_(dim),
        h_(dim, dim)
  {}
  
  void IndependentMvnBase::add_mixture_data(const Ptr<Data> &dp, double weight) {
    suf()->add_mixture_data(DAT(dp)->value(), weight);
  }

  double IndependentMvnBase::Logp(const Vector &x, Vector &g, Matrix &h,
                                  uint nderivs) const {
    int d = x.size();
    double qform = 0;
    double ldsi = 0;
    if (nderivs > 0) {
      g = 0;
      if (nderivs > 1) {
        h = 0;
      }
    }

    for (int i = 0; i < d; ++i) {
      double sigsq = this->sigsq(i);
      double delta = x[i] - mu(i);
      qform += square(delta) / sigsq;
      ldsi -= log(sigsq);
      if (nderivs > 0) {
        g[i] = -delta / sigsq;
        if (nderivs > 1) {
          h(i, i) = -1.0 / sigsq;
        }
      }
    }
    const double log2pi = 1.83787706641;
    return 0.5 * (ldsi - qform - d * log2pi);
  }

  DiagonalMatrix IndependentMvnBase::diagonal_variance() const {
    return DiagonalMatrix(sigsq());
  }
  
  const SpdMatrix &IndependentMvnBase::Sigma() const {
    sigma_scratch_.set_diag(sigsq());
    return sigma_scratch_;
  }

  const SpdMatrix &IndependentMvnBase::siginv() const {
    sigma_scratch_.set_diag(1.0 / sigsq());
    return sigma_scratch_;
  }

  double IndependentMvnBase::ldsi() const {
    double ans = 0;
    const Vector &sigsq(this->sigsq());
    for (int i = 0; i < length(mu()); ++i) {
      ans -= log(sigsq[i]);
    }
    return ans;
  }

  Vector IndependentMvnBase::sim(RNG &rng) const {
    Vector ans(mu());
    for (int i = 0; i < ans.size(); ++i) {
      ans += rnorm_mt(rng, 0, sigma(i));
    }
    return ans;
  }

  double IndependentMvnBase::pdf(const Data *dp, bool logscale) const {
    double ans = Logp(DAT(dp)->value(), g_, h_, 0);
    return logscale ? ans : exp(ans);
  }
  
  //======================================================================
  IndependentMvnModel::IndependentMvnModel(int dim)
      : IndependentMvnBase(dim),
        ParamPolicy(new VectorParams(dim, 0.0), new VectorParams(dim, 1.0))
  {}

  IndependentMvnModel::IndependentMvnModel(const Vector &mean,
                                           const Vector &variance)
      : IndependentMvnBase(mean.size()),
        ParamPolicy(new VectorParams(mean), new VectorParams(variance))
  {
    if (mean.size() != variance.size()) {
      report_error(
          "The mean and the variance must be equal-sized "
          "vectors in IndependentMvnModel constructor");
    }
  }

  IndependentMvnModel::IndependentMvnModel(const IndependentMvnModel &rhs)
      : Model(rhs),
        IndependentMvnBase(rhs),
        ParamPolicy(rhs),
        PriorPolicy(rhs)
  {}

  IndependentMvnModel *IndependentMvnModel::clone() const {
    return new IndependentMvnModel(*this);
  }

  void IndependentMvnModel::mle() {
    const auto &sufstat(*suf());
    for (int i = 0; i < dim(); ++i) {
      set_mu_element(sufstat.ybar(i), i);
      double ni = sufstat.n(i);
      set_sigsq_element((ni - 1) * sufstat.sample_var(i) / ni, i);
    }
  }
  
  void IndependentMvnModel::set_mu(const Vector &mu) { Mu_prm()->set(mu); }

  void IndependentMvnModel::set_mu_element(double value, int position) {
    Mu_prm()->set_element(value, position);
  }

  void IndependentMvnModel::set_sigsq(const Vector &sigsq) {
    Sigsq_prm()->set(sigsq);
  }

  void IndependentMvnModel::set_sigsq_element(double sigsq, int position) {
    Sigsq_prm()->set_element(sigsq, position);
  }

  //===========================================================================
  ZeroMeanIndependentMvnModel::ZeroMeanIndependentMvnModel(int dim)
      : IndependentMvnBase(dim),
        ParamPolicy(new VectorParams(dim, 1.0)),
        zero_(dim, 0.0)
  {}

  ZeroMeanIndependentMvnModel::ZeroMeanIndependentMvnModel(
      const Vector &variance)
      : IndependentMvnBase(variance.size()),
        ParamPolicy(new VectorParams(variance)),
        zero_(variance.size(), 0.0)
  {}

  ZeroMeanIndependentMvnModel::ZeroMeanIndependentMvnModel(
      const ZeroMeanIndependentMvnModel &rhs)
      : IndependentMvnBase(rhs),
        zero_(rhs.dim(), 0.0),
        sigma_scratch_(rhs.dim()),
        g_(rhs.dim()),
        h_(rhs.dim(), rhs.dim())
  {}

  ZeroMeanIndependentMvnModel *ZeroMeanIndependentMvnModel::clone() const {
    return new ZeroMeanIndependentMvnModel(*this);
  }

  void ZeroMeanIndependentMvnModel::mle() {
    const auto &sufstat(*suf());
    for (int i = 0; i < dim(); ++i) {
      double sample_size = sufstat.n(i);
      if (sample_size > 0) {
        set_sigsq_element(sufstat.sumsq(i) / sample_size, i);
      }
    }
  }
  
}  // namespace BOOM
