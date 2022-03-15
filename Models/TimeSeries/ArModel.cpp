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

#include "Models/TimeSeries/ArModel.hpp"
#include <complex>
#include <functional>
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/Polynomial.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  ArSuf::ArSuf(int number_of_lags)
      : reg_suf_(new NeRegSuf(number_of_lags)), x_(number_of_lags) {}

  ArSuf *ArSuf::clone() const { return new ArSuf(*this); }

  void ArSuf::clear() {
    lags_.clear();
    reg_suf_->clear();
  }

  void ArSuf::Update(const DoubleData &y) {
    double yvalue = y.value();
    if (lags_.size() == reg_suf_->size()) {
      x_.assign(lags_.begin(), lags_.end());
      reg_suf_->add_mixture_data(yvalue, x_, 1.0);
      lags_.push_front(yvalue);
      lags_.pop_back();
    } else if (lags_.size() < reg_suf_->size()) {
      lags_.push_front(yvalue);
    } else {
      report_error("Vector of lags is larger than the AR(p) dimension.");
    }
  }

  void ArSuf::add_mixture_data(double y, const Vector &x, double weight) {
    reg_suf_->add_mixture_data(y, x, weight);
  }

  void ArSuf::combine(const Ptr<ArSuf> &s) { reg_suf_->combine(s->reg_suf_); }

  void ArSuf::combine(const ArSuf &s) { reg_suf_->combine(*s.reg_suf_); }

  ArSuf *ArSuf::abstract_combine(Sufstat *s) {
    return abstract_combine_impl<ArSuf>(this, s);
  }

  Vector ArSuf::vectorize(bool minimal) const {
    return reg_suf_->vectorize(minimal);
  }

  Vector::const_iterator ArSuf::unvectorize(Vector::const_iterator &v,
                                            bool minimal) {
    return reg_suf_->unvectorize(v, minimal);
  }

  Vector::const_iterator ArSuf::unvectorize(const Vector &v, bool minimal) {
    return reg_suf_->unvectorize(v, minimal);
  }

  std::ostream &ArSuf::print(std::ostream &out) const {
    reg_suf_->print(out);
    out << "lags:" << endl;
    for (int i = 0; i < lags_.size(); ++i) {
      out << i + 1 << ":  " << lags_[i] << endl;
    }
    return out;
  }

  //======================================================================
  ArModel::ArModel(int number_of_lags)
      : ParamPolicy(new GlmCoefs(Vector(number_of_lags, 0.0), true),
                    new UnivParams(1.0)),
        DataPolicy(new ArSuf(number_of_lags)),
        filter_coefficients_current_(false) {
    Phi_prm()->add_observer(this, [this]() { this->observe_phi(); });
    Phi_prm()->add_all();
  }

  ArModel::ArModel(const Ptr<GlmCoefs> &autoregression_coefficients,
                   const Ptr<UnivParams> &innovation_variance)
      : ParamPolicy(autoregression_coefficients, innovation_variance),
        DataPolicy(new ArSuf(autoregression_coefficients->size())),
        filter_coefficients_current_(false) {
    bool ok = check_stationary(autoregression_coefficients->value());
    if (!ok) {
      report_error(
          "Attempt to initialize ArModel with an illegal value "
          "of the autoregression coefficients.");
    }
    Phi_prm()->add_observer(this, [this]() { this->observe_phi(); });
  }

  ArModel *ArModel::clone() const { return new ArModel(*this); }

  int ArModel::number_of_lags() const { return phi().size(); }

  double ArModel::sigma() const { return sqrt(Sigsq_prm()->value()); }

  double ArModel::sigsq() const { return Sigsq_prm()->value(); }

  const Vector &ArModel::phi() const { return Phi_prm()->value(); }

  void ArModel::set_sigma(double sigma) { Sigsq_prm()->set(sigma * sigma); }

  void ArModel::set_sigsq(double sigsq) { Sigsq_prm()->set(sigsq); }

  void ArModel::set_phi(const Vector &phi) {
    if (phi.size() == coef().nvars_possible()) {
      coef().set_Beta(phi);
    } else {
      coef().set_included_coefficients(phi);
    }
  }

  Ptr<GlmCoefs> ArModel::Phi_prm() { return prm1(); }
  const Ptr<GlmCoefs> ArModel::Phi_prm() const { return prm1(); }
  Ptr<UnivParams> ArModel::Sigsq_prm() { return prm2(); }
  const Ptr<UnivParams> ArModel::Sigsq_prm() const { return prm2(); }

  const GlmCoefs &ArModel::coef() const { return prm1_ref(); }
  GlmCoefs &ArModel::coef() { return prm1_ref(); }

  bool ArModel::check_stationary(const Vector &phi) {
    // The process is stationary if the roots of the polynomial
    //
    // 1 - phi[0]*z - ... - phi[p-1]*z^p.
    //
    // all lie outside the unit circle.  We can do that by explicitly finding
    // and checking the roots, but that's kind of expensive.  Before doing that
    // we can do a quick check to see if the coefficients are within a loose
    // bound.
    //
    // Based on Rouche's theorem:
    // http://en.wikipedia.org/wiki/Properties_of_polynomial_roots#Based_on_the_Rouch.C3.A9_theorem
    // All the roots will be at least 1 in absolute value as long as
    // sum(abs(phi)) < 1.
    if (phi.abs_norm() < 1) return true;

    // If that didn't work then we're stuck finding roots.
    // TODO(stevescott): Really we just need to check the smallest root.  If we
    // had a cheap way of finding just the smallest root then that would be more
    // efficient than finding them all.
    Vector coefficients = concat(1, -1 * phi);

    Polynomial polynomial(coefficients);
    std::vector<std::complex<double> > roots(polynomial.roots());
    for (int i = 0; i < roots.size(); ++i) {
      if (abs(roots[i]) <= 1) return false;
    }
    return true;
  }

  Vector ArModel::autocovariance(int number_of_lags) const {
    set_filter_coefficients();
    Vector ans(number_of_lags + 1);
    for (int lag = 0; lag <= number_of_lags; ++lag) {
      int n = filter_coefficients_.size() - lag;
      const ConstVectorView psi(filter_coefficients_, 0, n);
      const ConstVectorView lag_psi(filter_coefficients_, lag, n);
      ans[lag] = psi.dot(lag_psi);
    }
    return ans * sigsq();
  }

  Vector ArModel::simulate(int n, RNG &rng) const {
    int p = number_of_lags();
    Vector acf = autocovariance(p);
    SpdMatrix Sigma(p);
    Sigma.diag() = acf[0];
    for (int i = 1; i < p; ++i) {
      Sigma.subdiag(i) = acf[i];
      Sigma.superdiag(i) = acf[i];
    }
    Vector zero(p, 0.0);
    Vector y0 = rmvn(zero, Sigma);
    return simulate(n, y0, rng);
  }

  Vector ArModel::simulate(int n, const Vector &y0, RNG &rng) const {
    if (y0.size() != number_of_lags()) {
      ostringstream err;
      err << "Error in ArModel::simulate." << endl
          << "Initial state value y0 was size " << y0.size()
          << ", but the model has " << number_of_lags() << " lags." << endl;
      report_error(err.str());
    }
    const Vector &phi(this->phi());
    std::deque<double> lags(y0.rbegin(), y0.rend());
    Vector ans;
    ans.reserve(n);
    for (int i = 0; i < n; ++i) {
      double mu = 0;
      for (int lag = 0; lag < number_of_lags(); ++lag) {
        mu += phi[lag] * lags[lag];
      }
      double y = rnorm_mt(rng, mu, sigma());
      lags.push_front(y);
      lags.pop_back();
      ans.push_back(y);
    }
    return ans;
  }

  // Determine the MA filter coefficients from the AR coefficients by
  // equating coefficients in the polynomial phi(z)*psi(z) = 1.
  // phi(z) = 1 - phi_1 z - phi_2 z^2 - ... - phi_p z^p.  This implies
  // that psi[0] = 1.  The coefficient of z^n is
  //
  // psi[n] - psi[n-1]*phi_1 - psi[n-2]*phi_2 - ... = 0
  // implying
  // psi[n] = psi[n-1]*phi_1 + ...
  //
  // The preceding math is unit-offset for phi, but zero-offset for
  // psi.
  void ArModel::set_filter_coefficients() const {
    if (filter_coefficients_current_) return;
    const Vector &phi(this->phi());
    int p = phi.size();

    filter_coefficients_.resize(2);
    filter_coefficients_[0] = 1.0;
    if (phi.empty()) return;
    filter_coefficients_[1] = phi[0];
    bool done = false;

    for (int n = 2;; ++n) {
      if (n <= phi.size()) {
        ConstVectorView phi_view(phi, 0, n);
        ConstVectorView psi(filter_coefficients_, 0, n);
        double value = phi_view.dot(psi.reverse());
        filter_coefficients_.push_back(value);
      } else {
        ConstVectorView psi(filter_coefficients_, n - p, p);
        double value = phi.dot(psi.reverse());
        filter_coefficients_.push_back(value);
        ConstVectorView psi_tail(filter_coefficients_, n - p, p);
        // You're done when the last p elements of the vector are all
        // small.
        done = psi_tail.abs_norm() < 1e-6;
      }
      if (done) break;
    }
    filter_coefficients_current_ = true;
  }

}  // namespace BOOM
