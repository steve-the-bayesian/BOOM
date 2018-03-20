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

  IndependentMvnSuf::IndependentMvnSuf(int dim)
      : sum_(dim), sumsq_(dim), n_(0) {}

  IndependentMvnSuf *IndependentMvnSuf::clone() const {
    return new IndependentMvnSuf(*this);
  }

  void IndependentMvnSuf::clear() {
    sum_ = 0;
    sumsq_ = 0;
    n_ = 0;
  }

  void IndependentMvnSuf::resize(int dim) {
    sum_.resize(dim);
    sumsq_.resize(dim);
    clear();
  }

  void IndependentMvnSuf::Update(const VectorData &d) { update_raw(d.value()); }

  void IndependentMvnSuf::update_raw(const Vector &y) {
    n_ += 1;
    sum_ += y;
    for (int i = 0; i < y.size(); ++i) {
      sumsq_[i] += square(y[i]);
    }
  }

  void IndependentMvnSuf::add_mixture_data(const Vector &v, double prob) {
    n_ += prob;
    sum_.axpy(v, prob);
    for (int i = 0; i < v.size(); ++i) {
      sumsq_[i] += prob * square(v[i]);
    }
  }

  void IndependentMvnSuf::update_expected_value(
      double sample_size, const Vector &expected_sum,
      const Vector &expected_sum_of_squares) {
    n_ += sample_size;
    sum_ += expected_sum;
    sumsq_ += expected_sum_of_squares;
  }

  double IndependentMvnSuf::sum(int i) const { return sum_[i]; }
  double IndependentMvnSuf::sumsq(int i) const { return sumsq_[i]; }

  double IndependentMvnSuf::centered_sumsq(int i, double mu) const {
    return sumsq_[i] - 2 * mu * sum_[i] + n_ * square(mu);
  }

  double IndependentMvnSuf::n() const { return n_; }

  double IndependentMvnSuf::ybar(int i) const {
    double ni = n_;
    if (ni < 1e-7) {
      return 0;
    }
    return sum_[i] / ni;
  }

  double IndependentMvnSuf::sample_var(int i) const {
    double ni = n_;
    if (ni - 1 < std::numeric_limits<double>::epsilon()) {
      return 0;
    }
    double ybari = ybar(i);
    double ss = sumsq_[i] - ni * ybari * ybari;
    return ss / (ni - 1);
  }

  IndependentMvnSuf *IndependentMvnSuf::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  void IndependentMvnSuf::combine(const Ptr<IndependentMvnSuf> &s) {
    return this->combine(*s);
  }

  void IndependentMvnSuf::combine(const IndependentMvnSuf &s) {
    n_ += s.n_;
    sum_ += s.sum_;
    sumsq_ += s.sumsq_;
  }

  Vector IndependentMvnSuf::vectorize(bool) const {
    Vector ans(1, n_);
    ans.reserve(1 + 2 * sum_.size());
    ans.concat(sum_);
    ans.concat(sumsq_);
    return (ans);
  }

  Vector::const_iterator IndependentMvnSuf::unvectorize(
      Vector::const_iterator &v, bool) {
    int dim = sum_.size();
    n_ = *v;
    v += 1;
    sum_.assign(v, v + dim);
    v += dim;
    sumsq_.assign(v, v + dim);
    v += dim;
    return v;
  }

  Vector::const_iterator IndependentMvnSuf::unvectorize(const Vector &v,
                                                        bool minimal) {
    Vector::const_iterator vi = v.begin();
    return unvectorize(vi, minimal);
  }

  ostream &IndependentMvnSuf::print(ostream &out) const {
    Matrix tmp(sum_.size(), 2);
    tmp.col(0) = sum_;
    tmp.col(1) = sumsq_;
    out << n_ << std::endl << tmp;
    return out;
  }

  //======================================================================
  IndependentMvnModel::IndependentMvnModel(int dim)
      : ParamPolicy(new VectorParams(dim, 0.0), new VectorParams(dim, 1.0)),
        DataPolicy(new IndependentMvnSuf(dim)),
        sigma_scratch_(dim),
        g_(dim),
        h_(dim, dim) {}

  IndependentMvnModel::IndependentMvnModel(const Vector &mean,
                                           const Vector &variance)
      : ParamPolicy(new VectorParams(mean), new VectorParams(variance)),
        DataPolicy(new IndependentMvnSuf(mean.size())),
        sigma_scratch_(mean.size()),
        g_(mean.size()),
        h_(mean.size(), mean.size()) {
    if (mean.size() != variance.size()) {
      report_error(
          "The mean and the variance must be equal-sized "
          "vectors in IndependentMvnModel constructor");
    }
  }

  IndependentMvnModel::IndependentMvnModel(const IndependentMvnModel &rhs)
      : Model(rhs),
        MvnBase(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        sigma_scratch_(rhs.dim()),
        g_(rhs.dim()),
        h_(rhs.dim(), rhs.dim()) {}

  IndependentMvnModel *IndependentMvnModel::clone() const {
    return new IndependentMvnModel(*this);
  }

  double IndependentMvnModel::Logp(const Vector &x, Vector &g, Matrix &h,
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

  const Vector &IndependentMvnModel::mu() const { return Mu_ref().value(); }

  const SpdMatrix &IndependentMvnModel::Sigma() const {
    sigma_scratch_.set_diag(sigsq());
    return sigma_scratch_;
  }

  const SpdMatrix &IndependentMvnModel::siginv() const {
    sigma_scratch_.set_diag(1.0 / sigsq());
    return sigma_scratch_;
  }

  double IndependentMvnModel::ldsi() const {
    double ans = 0;
    const Vector &sigsq(this->sigsq());
    for (int i = 0; i < length(mu()); ++i) {
      ans -= log(sigsq[i]);
    }
    return ans;
  }

  Vector IndependentMvnModel::sim(RNG &rng) const {
    Vector ans(mu());
    for (int i = 0; i < ans.size(); ++i) {
      ans += rnorm_mt(rng, 0, sigma(i));
    }
    return ans;
  }

  Ptr<VectorParams> IndependentMvnModel::Mu_prm() { return prm1(); }
  const Ptr<VectorParams> IndependentMvnModel::Mu_prm() const { return prm1(); }
  const VectorParams &IndependentMvnModel::Mu_ref() const { return prm1_ref(); }

  Ptr<VectorParams> IndependentMvnModel::Sigsq_prm() { return prm2(); }
  const Ptr<VectorParams> IndependentMvnModel::Sigsq_prm() const {
    return prm2();
  }
  const VectorParams &IndependentMvnModel::Sigsq_ref() const {
    return prm2_ref();
  }

  const Vector &IndependentMvnModel::sigsq() const {
    return Sigsq_ref().value();
  }

  double IndependentMvnModel::mu(int i) const { return mu()[i]; }

  double IndependentMvnModel::sigsq(int i) const { return sigsq()[i]; }

  double IndependentMvnModel::sigma(int i) const { return sqrt(sigsq(i)); }

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

  double IndependentMvnModel::pdf(const Data *dp, bool logscale) const {
    double ans = Logp(DAT(dp)->value(), g_, h_, 0);
    return logscale ? ans : exp(ans);
  }
}  // namespace BOOM
