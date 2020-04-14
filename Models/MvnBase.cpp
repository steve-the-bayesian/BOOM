// Copyright 2018 Google LLC. All Rights Reserved.
/*
 Copyright (C) 2007-2010 Steven L. Scott

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

#include "Models/MvnBase.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "distributions.hpp"
#include "numopt/initialize_derivatives.hpp"

namespace BOOM {

  typedef MvnBase MB;

  MvnSuf::MvnSuf(uint p)
      : ybar_(p, 0.0), sumsq_(p, 0.0), n_(0.0), sym_(false) {}

  MvnSuf::MvnSuf(double n, const Vector &ybar, const SpdMatrix &sumsq)
      : ybar_(ybar), sumsq_(sumsq), n_(n), sym_(false) {}

  MvnSuf::MvnSuf(const MvnSuf &rhs)
      : Sufstat(rhs),
        SufstatDetails<VectorData>(rhs),
        ybar_(rhs.ybar_),
        sumsq_(rhs.sumsq_),
        n_(rhs.n_),
        sym_(rhs.sym_) {}

  MvnSuf *MvnSuf::clone() const { return new MvnSuf(*this); }

  void MvnSuf::clear() {
    ybar_ = 0;
    sumsq_ = 0;
    n_ = 0;
    sym_ = false;
  }

  void MvnSuf::resize(uint p) {
    ybar_.resize(p);
    sumsq_.resize(p);
    clear();
  }

  void MvnSuf::check_dimension(const Vector &y) {
    if (ybar_.empty()) {
      resize(y.size());
    }
    if (y.size() != ybar_.size()) {
      ostringstream msg;
      msg << "attempting to update MvnSuf of dimension << " << ybar_.size()
          << " with data of dimension " << y.size() << "." << endl
          << "Value of data point is [" << y << "]";
      report_error(msg.str().c_str());
    }
  }

  void MvnSuf::update_raw(const Vector &y) {
    check_dimension(y);
    n_ += 1.0;
    // Being careful to avoid unnecessary memory allocations.
    // This computes  wsp_ = (y - ybar_) / n_;
    wsp_ = y;
    wsp_ -= ybar_;
    wsp_ /= n_;
    ybar_ += wsp_;  // new ybar
    sumsq_.add_outer(wsp_, n_ - 1, false);

    // And this sets wsp_ = (y - ybar_)
    wsp_ = y;
    wsp_ -= ybar_;
    sumsq_.add_outer(wsp_, 1, false);
    sym_ = false;
  }

  void MvnSuf::update_expected_value(double sample_size,
                                     const Vector &expected_sum,
                                     const SpdMatrix &expected_sum_of_squares) {
    n_ += sample_size;
    wsp_ = (expected_sum - ybar_) / n_;
    ybar_ += wsp_;

    sumsq_.add_outer(wsp_, n_ - sample_size, false);
    sumsq_.add_outer(expected_sum - ybar_, sample_size, false);
    sym_ = false;
  }

  void MvnSuf::remove_data(const Vector &y) {
    if (n_ <= 0.0) {
      report_error("Sufficient statistics already empty.");
    }
    ybar_ *= n_;
    ybar_ -= y;
    if (n_ > 1.0) {
      ybar_ /= (n_ - 1);
    }
    sumsq_.add_outer(y - ybar_, -(n_ - 1) / n_, false);
    n_ -= 1.0;
    sym_ = false;
  }

  void MvnSuf::Update(const VectorData &X) {
    const Vector &x(X.value());
    update_raw(x);
  }

  void MvnSuf::add_mixture_data(const Vector &y, double prob) {
    check_dimension(y);
    n_ += prob;
    wsp_ = (y - ybar_) * (prob / n_);  // old ybar_, new n_
    ybar_ += wsp_;                     // new ybar_
    sumsq_.add_outer(wsp_, n_ - prob, false);
    sumsq_.add_outer(y - ybar_, prob, false);
    sym_ = false;
  }

  Vector MvnSuf::sum() const { return ybar_ * n_; }
  SpdMatrix MvnSuf::sumsq() const {
    check_symmetry();
    SpdMatrix ans(sumsq_);
    ans.add_outer(ybar_, n_);
    return ans;
  }
  double MvnSuf::n() const { return n_; }

  void MvnSuf::check_symmetry() const {
    if (!sym_) {
      sumsq_.reflect();
      sym_ = true;
    }
  }

  const Vector &MvnSuf::ybar() const { return ybar_; }
  SpdMatrix MvnSuf::sample_var() const {
    if (n() > 1) return center_sumsq() / (n() - 1);
    return sumsq_ * 0.0;
  }

  SpdMatrix MvnSuf::var_hat() const {
    if (n() > 0) return center_sumsq() / n();
    return sumsq_ * 0.0;
  }

  SpdMatrix MvnSuf::center_sumsq(const Vector &mu) const {
    SpdMatrix ans = center_sumsq();
    ans.add_outer(ybar_ - mu, n_);
    return ans;
  }

  const SpdMatrix &MvnSuf::center_sumsq() const {
    check_symmetry();
    return sumsq_;
  }

  void MvnSuf::combine(const Ptr<MvnSuf> &s) { this->combine(*s); }

  // TODO: test this
  void MvnSuf::combine(const MvnSuf &s) {
    Vector zbar = (sum() + s.sum()) / (n() + s.n());
    sumsq_ = center_sumsq(zbar) + s.center_sumsq(zbar);
    ybar_ = zbar;
    n_ += s.n();
    sym_ = true;
  }

  MvnSuf *MvnSuf::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  Vector MvnSuf::vectorize(bool minimal) const {
    Vector ans(ybar_);
    ans.concat(sumsq_.vectorize(minimal));
    ans.push_back(n_);
    return ans;
  }

  Vector::const_iterator MvnSuf::unvectorize(Vector::const_iterator &v, bool) {
    uint dim = ybar_.size();
    ybar_.assign(v, v + dim);
    v += dim;
    sumsq_.unvectorize(v);
    n_ = *v;
    ++v;
    return v;
  }

  Vector::const_iterator MvnSuf::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  std::ostream &MvnSuf::print(std::ostream &out) const {
    out << n_ << endl << ybar_ << endl << sumsq_;
    return out;
  }

  //======================================================================

  uint MvnBase::dim() const { return mu().size(); }

  double MvnBase::Logp(const Vector &x, Vector &g, Matrix &h, uint nd) const {
    double ans = dmvn(x, mu(), siginv(), ldsi(), true);
    if (nd > 0) {
      g = -(siginv() * (x - mu()));
      if (nd > 1) h = -siginv();
    }
    return ans;
  }

  double MvnBase::logp_given_inclusion(const Vector &x_subset, Vector *gradient,
                                       Matrix *Hessian,
                                       const Selector &included,
                                       bool reset_derivatives) const {
    if (included.nvars() == 0) {
      return 0.0;
    }
    Vector mu0 = included.select(mu());
    SpdMatrix precision = included.select(siginv());
    double ans = dmvn(x_subset, mu0, precision, precision.logdet(), true);
    initialize_derivatives(gradient, Hessian, included.nvars(),
                           reset_derivatives);
    if (gradient) {
      *gradient -= precision * (x_subset - mu0);
      if (Hessian) {
        *Hessian -= precision;
      }
    }
    return ans;
  }

  double MvnBase::log_likelihood(const Vector &mu, const SpdMatrix &siginv,
                                 const MvnSuf &suf) const {
    const double log2pi = 1.83787706641;
    double n = suf.n();
    const Vector &ybar = suf.ybar();
    const SpdMatrix &sumsq = suf.center_sumsq();

    double qform = n * (siginv.Mdist(ybar, mu)) + traceAB(siginv, sumsq);
    double nc = 0.5 * n * (-dim() * log2pi + siginv.logdet());
    double ans = nc - .5 * qform;
    return ans;
  }

  Vector MvnBase::sim(RNG &rng) const { return rmvn_mt(rng, mu(), Sigma()); }

  typedef MvnBaseWithParams MBP;

  MBP::MvnBaseWithParams(uint p, double mu, double sigsq)
      : ParamPolicy(new VectorParams(p, mu), new SpdParams(p, sigsq)) {}

  // N(mu,V)... if (ivar) then V is the inverse variance.
  MBP::MvnBaseWithParams(const Vector &mean, const SpdMatrix &V, bool ivar)
      : ParamPolicy(new VectorParams(mean), new SpdParams(V, ivar)) {}

  MBP::MvnBaseWithParams(const Ptr<VectorParams> &mu,
                         const Ptr<SpdParams> &Sigma)
      : ParamPolicy(mu, Sigma) {}

  MBP::MvnBaseWithParams(const MvnBaseWithParams &rhs)
      : Model(rhs),
        VectorModel(rhs),
        MvnBase(rhs),
        ParamPolicy(rhs),
        LocationScaleVectorModel(rhs) {}

  Ptr<VectorParams> MBP::Mu_prm() { return ParamPolicy::prm1(); }
  const Ptr<VectorParams> MBP::Mu_prm() const { return ParamPolicy::prm1(); }

  Ptr<SpdParams> MBP::Sigma_prm() { return ParamPolicy::prm2(); }
  const Ptr<SpdParams> MBP::Sigma_prm() const { return ParamPolicy::prm2(); }

  const Vector &MBP::mu() const { return prm1_ref().value(); }
  const SpdMatrix &MBP::Sigma() const { return prm2_ref().var(); }
  const SpdMatrix &MBP::siginv() const { return prm2_ref().ivar(); }
  double MBP::ldsi() const { return prm2_ref().ldsi(); }
  Matrix MBP::Sigma_chol() const { return prm2_ref().var_chol(); }

  void MBP::set_mu(const Vector &v) { prm1_ref().set(v); }
  void MBP::set_Sigma(const SpdMatrix &s) { prm2_ref().set_var(s); }
  void MBP::set_siginv(const SpdMatrix &ivar) { prm2_ref().set_ivar(ivar); }

}  // namespace BOOM
