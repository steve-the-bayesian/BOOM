// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2008 Steven L. Scott

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
#include "Models/GaussianModelBase.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "distributions.hpp"
#include "cpputil/Constants.hpp"

namespace BOOM {

  namespace {
    using GS = BOOM::GaussianSuf;
  }

  GS::GaussianSuf(double Sum, double Sumsq, double N)
      : sum_(Sum), sumsq_(Sumsq), n_(N) {}

  GS::GaussianSuf(const GS &rhs)
      : Sufstat(rhs),
        SufstatDetails<DataType>(rhs),
        sum_(rhs.sum_),
        sumsq_(rhs.sumsq_),
        n_(rhs.n_) {}

  GS *GS::clone() const { return new GS(*this); }

  void GS::Update(const DoubleData &X) {
    const double &x = X.value();
    update_raw(x);
  }

  void GS::update_raw(double y) {
    n_ += 1;
    sum_ += y;
    sumsq_ += y * y;
  }

  void GS::update_expected_value(double expected_sample_size,
                                 double expected_sum,
                                 double expected_sum_of_squares) {
    n_ += expected_sample_size;
    sum_ += expected_sum;
    sumsq_ += expected_sum_of_squares;
  }

  void GS::remove(double y) {
    n_ -= 1;
    sum_ -= y;
    sumsq_ -= y * y;
  }

  void GS::add_mixture_data(double y, double prob) {
    n_ += prob;
    prob *= y;
    sum_ += prob;
    prob *= y;
    sumsq_ += prob;
  }

  double GS::sum() const { return sum_; }
  double GS::sumsq() const { return sumsq_; }
  double GS::centered_sumsq(double mu) const {
    return sumsq_ - (2.0 * sum_ * mu) + n_ * mu * mu;
  }
  double GS::n() const { return n_; }
  double GS::ybar() const {
    if (n_ > 0) {
      return sum() / n();
    }
    return 0.0;
  }

  double GS::sample_var() const {
    if (n_ <= 1) {
      return 0;
    }
    double ss = sumsq() - n() * pow(ybar(), 2);
    return ss / (n_ - 1);
  }

  void GS::clear() { sum_ = sumsq_ = n_ = 0; }

  void GS::combine(const Ptr<GS> &s) {
    n_ += s->n_;
    sum_ += s->sum_;
    sumsq_ += s->sumsq_;
  }

  void GS::combine(const GS &rhs) {
    n_ += rhs.n_;
    sum_ += rhs.sum_;
    sumsq_ += rhs.sumsq_;
  }

  GaussianSuf *GS::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  Vector GS::vectorize(bool) const {
    Vector ans(3);
    ans[0] = n_;
    ans[1] = sum_;
    ans[2] = sumsq_;
    return ans;
  }

  Vector::const_iterator GS::unvectorize(Vector::const_iterator &v, bool) {
    n_ = *v;
    ++v;
    sum_ = *v;
    ++v;
    sumsq_ = *v;
    ++v;
    return v;
  }

  Vector::const_iterator GS::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  std::ostream &GS::print(std::ostream &out) const {
    return out << n_ << " " << sum_ << " " << sumsq_;
  }

  //======================================================================
  GaussianModelBase::GaussianModelBase() : DataPolicy(new GaussianSuf()) {}

  GaussianModelBase::GaussianModelBase(const std::vector<double> &y)
      : DataPolicy(new GaussianSuf()) {
    DataPolicy::set_data_raw(y.begin(), y.end());
  }

  double GaussianModelBase::sigma() const { return sqrt(sigsq()); }

  double GaussianModelBase::pdf(const Ptr<Data> &dp, bool logscale) const {
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double GaussianModelBase::pdf(const Data *dp, bool logscale) const {
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double GaussianModelBase::Logp(double x, double &g, double &h,
                                 uint nd) const {
    double m = mu();
    double ans = dnorm(x, m, sigma(), 1);
    if (nd > 0) {
      g = -(x - m) / sigsq();
    }
    if (nd > 1) {
      h = -1.0 / sigsq();
    }
    return ans;
  }

  double GaussianModelBase::Logp(const Vector &x, Vector &g, Matrix &h,
                                 uint nd) const {
    double X = x[0];
    double G(0), H(0);
    double ans = Logp(X, G, H, nd);
    if (nd > 0) {
      g[0] = G;
    }
    if (nd > 1) {
      h(0, 0) = H;
    }
    return ans;
  }

  double GaussianModelBase::ybar() const { return suf()->ybar(); }
  double GaussianModelBase::sample_var() const { return suf()->sample_var(); }

  void GaussianModelBase::add_mixture_data(const Ptr<Data> &dp, double prob) {
    double y = DAT(dp)->value();
    suf()->add_mixture_data(y, prob);
  }

  double GaussianModelBase::sim(RNG &rng) const {
    return rnorm_mt(rng, mu(), sigma());
  }

  void GaussianModelBase::add_data_raw(double x) {
    NEW(DoubleData, dp)(x);
    this->add_data(dp);
  }

  void GaussianModelBase::remove_data(const Ptr<Data> &dp) {
    Ptr<DoubleData> data_point = DAT(dp);
    DataPolicy::remove_data(dp);
    suf()->remove(data_point->value());
  }

  std::set<Ptr<Data>> GaussianModelBase::abstract_data_set() const {
    return std::set<Ptr<Data>>(dat().begin(), dat().end());
  }

  // The log likelihood conditional on sigma, but integrating out mu.
  double GaussianModelBase::log_integrated_likelihood(
      const GaussianSuf &suf, double mu0, double tausq, double sigsq) {
    double posterior_variance = 1.0 / (1.0 / tausq + suf.n() / sigsq);
    double posterior_mean = 
        posterior_variance * (suf.sum() / sigsq + mu0 / tausq);
    double sumsq =
        suf.centered_sumsq(suf.ybar()) / sigsq
        + suf.n() * square(suf.ybar()) / sigsq
        + square(mu0) / tausq
        - square(posterior_mean) / posterior_variance;

    double ans =
        -suf.n() * Constants::log_root_2pi
        -.5 * suf.n() * log(sigsq)
        + .5 * log(posterior_variance / tausq)
        - .5 * sumsq;
    return ans;
  }

  // The log likelihood after integrating out both sigma and mu.
  double GaussianModelBase::log_integrated_likelihood(
      const GaussianSuf &suf, double mu0, double kappa,
      double df, double ss) {
    double n = suf.n();
    double DF = df + n;
    double posterior_mean = (n * suf.ybar() + kappa * mu0) / (n + kappa);
    double SS =
        ss
        + suf.centered_sumsq(suf.ybar())
        + n * square(suf.ybar() - posterior_mean)
        + kappa * square(mu0 - posterior_mean);

    return -.5 * n * Constants::log_2pi
        + .5 * log(kappa / (n + kappa))
        + lgamma(DF / 2)
        - lgamma(df / 2)
        + (.5 * df) * log(ss / 2)
        - (.5 * DF) * log(SS / 2);
  }

  double GaussianModelBase::log_likelihood(const GaussianSuf &suf, double mu, double sigsq) {
    double n = suf.n();
    return -.5 * n * Constants::log_2pi
        - .5 * n * log(sigsq)
        - .5 * (n - 1) * suf.sample_var() / sigsq
        - .5 * n * square(suf.ybar() - mu) / sigsq;
  }
  
}  // namespace BOOM
