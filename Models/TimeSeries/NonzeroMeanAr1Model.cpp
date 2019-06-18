// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#include "Models/TimeSeries/NonzeroMeanAr1Model.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "stats/moments.hpp"

namespace BOOM {

  Ar1Suf::Ar1Suf() { clear(); }

  Ar1Suf *Ar1Suf::clone() const { return new Ar1Suf(*this); }
  void Ar1Suf::clear() {
    sumsq_ = 0;
    sum_ = 0;
    cross_ = 0;
    n_ = 0;
    first_value_ = 0;
    last_value_ = 0;
  }

  void Ar1Suf::update_raw(double y) {
    if (n_ == 0) {
      first_value_ = y;
    } else {
      cross_ += y * last_value_;
    }
    ++n_;
    sum_ += y;
    sumsq_ += y * y;
    last_value_ = y;
  }

  void Ar1Suf::Update(const DoubleData &d) { update_raw(d.value()); }

  Ar1Suf *Ar1Suf::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  void Ar1Suf::combine(const Ar1Suf &rhs) {
    report_error("combine method for Ar1Suf is ambiguous");
  }

  void Ar1Suf::combine(const Ptr<Ar1Suf> &rhs) { this->combine(*rhs); }

  Vector Ar1Suf::vectorize(bool) const {
    Vector ans(6);
    ans[0] = first_value_;
    ans[1] = n_;
    ans[2] = sum_;
    ans[3] = cross_;
    ans[4] = sumsq_;
    ans[5] = last_value_;
    return ans;
  }

  Vector::const_iterator Ar1Suf::unvectorize(Vector::const_iterator &v, bool) {
    first_value_ = *v;
    ++v;
    n_ = *v;
    ++v;
    sum_ = *v;
    ++v;
    cross_ = *v;
    ++v;
    sumsq_ = *v;
    ++v;
    last_value_ = *v;
    ++v;
    return v;
  }

  Vector::const_iterator Ar1Suf::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return this->unvectorize(it, minimal);
  }

  std::ostream &Ar1Suf::print(std::ostream &out) const {
    out << "first_value_ = " << first_value_ << endl
        << "sum_         = " << sum_ << endl
        << "n_           = " << n_ << endl
        << "cross_       = " << cross_ << endl
        << "sumsq_       = " << sumsq_ << endl
        << "last_value_  = " << last_value_ << endl;
    return out;
  }

  double Ar1Suf::n() const { return n_; }
  double Ar1Suf::model_sumsq(double mu, double phi) const {
    double ss = pow(first_value_ - mu, 2);
    ss += sumsq_excluding_first() - 2 * phi * cross_ -
          2 * (1 - phi) * mu * sum_excluding_first() + phi * phi * lag_sumsq() +
          2 * phi * (1 - phi) * mu * lag_sum() +
          (n_ - 1) * pow(mu * (1 - phi), 2);
    return ss;
  }
  double Ar1Suf::centered_lag_sumsq(double mu) const {
    return lag_sumsq() - 2 * lag_sum() * mu + (n_ - 1) * mu * mu;
  }
  double Ar1Suf::lag_sumsq() const { return sumsq_ - pow(last_value_, 2); }
  double Ar1Suf::lag_sum() const { return sum_ - last_value_; }
  double Ar1Suf::sum_excluding_first() const { return sum_ - first_value_; }
  double Ar1Suf::sumsq_excluding_first() const {
    return sumsq_ - pow(first_value_, 2);
  }
  double Ar1Suf::cross() const { return cross_; }
  double Ar1Suf::centered_cross(double mu) const {
    return cross_ - mu * (sum_excluding_first() + lag_sum()) +
           (n_ - 1) * mu * mu;
  }
  double Ar1Suf::first_value() const { return first_value_; }
  double Ar1Suf::last_value() const { return last_value_; }
  //======================================================================
  NonzeroMeanAr1Model::NonzeroMeanAr1Model(double mu, double phi, double sigma)
      : ParamPolicy(new UnivParams(mu), new UnivParams(phi),
                    new UnivParams(sigma * sigma)),
        DataPolicy(new Ar1Suf) {}

  NonzeroMeanAr1Model::NonzeroMeanAr1Model(const Vector &y)
      : ParamPolicy(new UnivParams(mean(y)), new UnivParams(0),
                    new UnivParams(1.0)),
        DataPolicy(new Ar1Suf) {
    for (int i = 0; i < y.size(); ++i) {
      NEW(DoubleData, dp)(y[i]);
      add_data(dp);
    }
    mle();
  }

  NonzeroMeanAr1Model::NonzeroMeanAr1Model(const NonzeroMeanAr1Model &rhs)
      : Model(rhs),
        MLE_Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs) {}

  NonzeroMeanAr1Model *NonzeroMeanAr1Model::clone() const {
    return new NonzeroMeanAr1Model(*this);
  }

  void NonzeroMeanAr1Model::mle() {
    SpdMatrix xtx(2);
    xtx(0, 0) = suf()->n() - 1;
    xtx(0, 1) = suf()->lag_sum();
    xtx(1, 0) = xtx(0, 1);
    xtx(1, 1) = suf()->lag_sumsq();

    Vector xty(2);
    xty[0] = suf()->sum_excluding_first();
    xty[1] = suf()->cross();

    Vector beta = xtx.solve(xty);
    double phi = beta[1];
    double mu = beta[0] / (1 - phi);
    set_mu(mu);
    set_phi(phi);

    double sumsq = suf()->model_sumsq(mu, phi);
    double sigsq = sumsq / (suf()->n() - 1);
    set_sigsq(sigsq);
  }

  double NonzeroMeanAr1Model::pdf(const Ptr<Data> &dp, bool logscale) const {
    double y = DAT(dp)->value();
    if (suf()->n() == 0) return dnorm(y, mu(), sigma(), logscale);
    double last = suf()->last_value();
    return dnorm(y, mu() + phi() * (last - mu()), sigma(), logscale);
  }

  double NonzeroMeanAr1Model::sigma() const {
    return sqrt(Sigsq_prm()->value());
  }
  double NonzeroMeanAr1Model::sigsq() const { return Sigsq_prm()->value(); }
  double NonzeroMeanAr1Model::phi() const { return Phi_prm()->value(); }
  double NonzeroMeanAr1Model::mu() const { return Mu_prm()->value(); }

  void NonzeroMeanAr1Model::set_sigsq(double sigsq) { Sigsq_prm()->set(sigsq); }
  void NonzeroMeanAr1Model::set_phi(double phi) { Phi_prm()->set(phi); }
  void NonzeroMeanAr1Model::set_mu(double mean) { Mu_prm()->set(mean); }

  Ptr<UnivParams> NonzeroMeanAr1Model::Mu_prm() { return prm1(); }
  const Ptr<UnivParams> NonzeroMeanAr1Model::Mu_prm() const { return prm1(); }
  Ptr<UnivParams> NonzeroMeanAr1Model::Phi_prm() { return prm2(); }
  const Ptr<UnivParams> NonzeroMeanAr1Model::Phi_prm() const { return prm2(); }
  Ptr<UnivParams> NonzeroMeanAr1Model::Sigsq_prm() { return prm3(); }
  const Ptr<UnivParams> NonzeroMeanAr1Model::Sigsq_prm() const {
    return prm3();
  }
}  // namespace BOOM
