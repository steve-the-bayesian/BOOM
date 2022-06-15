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
#include "Models/IRT/PartialCreditModel.hpp"
#include "Models/CategoricalData.hpp"
#include "Models/IRT/Subject.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "cpputil/lse.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/seq.hpp"

#include <functional>
#include <stdexcept>

namespace BOOM {
  namespace IRT {
    typedef PartialCreditModel PCR;
    typedef MultinomialLogitModel MLM;

    typedef PcrBetaConstraint PBC;
    bool PBC::check(const Vector &b) const {
      uint b_sz = b.size();
      if (b_sz < 2) return false;
      uint M = b_sz - 2;

      double b0 = b.front();
      double bM = b[M];  // next to last element

      return bM == (M + 1) * b0;  // factor of 'a' cancels
    }

    Vector & PBC::impose(Vector &b) const {
      uint b_sz = b.size();
      assert(b_sz >= 2);
      uint M = b_sz - 2;
      double dM = M;
      double beta0 = b[0];
      double betaM = b[M];

      double ad0 = M == 0 ? 0.0 : beta0 + (beta0 - betaM) / dM;
      if (ad0 != 0.0) {
        b -= ad0;
        b.back() += ad0;
      }
      return b;
    }

    Vector PBC::reduce(const Vector &b) const {
      uint b_sz = b.size();
      assert(b_sz >= 2);
      Vector ans(b_sz - 1);
      std::copy(b.begin() + 1, b.end(), ans.begin());
      return ans;
    }

    Vector PBC::expand(const Vector &b) const {
      Vector ans(b.size() + 1);
      std::copy(b.begin(), b.end(), ans.begin());
      uint M = b.size() - 1;
      double dM = M;
      assert(M > 1);
      ans[0] = b[M - 1] / (dM + 1);
      return ans;
    }
    //______________________________________________________________________
    typedef PcrDConstraint PDC;

    bool PDC::check(const Vector &d) const {
      return (d[0] == 0.0 && d.sum() == 0.0);
    }

    Vector &PDC::impose(Vector &d) const {
      double d0 = d[0];
      d -= d0;
      d.back() -= d.sum();
      return d;
    }

    Vector PDC::reduce(const Vector &d_full) const {
      if (d_full.size() <= 2) return Vector(0);
      Vector ans(d_full.size() - 2);
      std::copy(d_full.begin() + 1, d_full.end() - 1, ans.begin());
      return ans;
    }

    Vector PDC::expand(const Vector &d_min) const {
      Vector ans(d_min.size() + 2);
      ans[0] = 0.0;
      std::copy(d_min.begin(), d_min.end(), ans.begin() + 1);
      ans.back() = -d_min.sum();
      return ans;
    }
    //______________________________________________________________________

    void PCR::setup() {
      set_abd_current();
      setup_X();
      setup_beta();
      setup_aliases();
      set_observers();
    }

    void PCR::setup_X() {
      X_.set_diag(1.0);
      VectorView v(X_.last_col());
      for (uint i = 0; i < v.size(); ++i) v[i] = i + 1;
    }

    void PCR::setup_beta() {
      uint M = maxscore();
      if (d0_is_fixed) beta_ = new ConstrainedVectorParams(Vector(M + 2));
      fill_beta(true);
      b_ = beta_->value();
    }

    void PCR::setup_aliases() {
      a_prm = A_prm();
      b_prm = B_prm();
      d_prm = D_prm();
    }

    void PCR::set_observers() {
      A_prm()->add_observer(this, [this]() { this->observe_a(); });
      B_prm()->add_observer(this, [this]() { this->observe_b(); });
      D_prm()->add_observer(this, [this]() { this->observe_d(); });
      beta_->add_observer(this, [this]() { this->observe_beta(); });
    }

    //______________________________________________________________________
    // constructors

    Ptr<ConstrainedVectorParams> make_d_uint(uint Maxscore, bool id_d0);
    Ptr<ConstrainedVectorParams> make_d_vec(const Vector &d, bool id_d0);

    Ptr<ConstrainedVectorParams> make_d_uint(uint Maxscore, bool id_d0) {
      Vector d(Maxscore + 1, 0.0);
      return make_d_vec(d, id_d0);
    }

    Ptr<ConstrainedVectorParams> make_d_vec(const Vector &d, bool id_d0) {
      if (id_d0) return new ConstrainedVectorParams(d, new PcrDConstraint);
      return new ConstrainedVectorParams(d, new SumConstraint(0.0));
    }

    PCR::PartialCreditModel(const std::string &Id, uint Mscore, uint which_sub,
                            uint Nscales, const std::string &Name, bool id_d0)
        : Item(Id, Mscore, which_sub, Nscales, Name),
          ParamPolicy(new UnivParams(1.0), new UnivParams(0.0),
                      make_d_uint(Mscore, id_d0)),
          PriorPolicy(),
          b_(Mscore + 2),
          eta_(Mscore + 1),
          X_(Mscore + 1, Mscore + 2),
          d0_is_fixed(true),
          which_subscale_(which_sub) {
      setup();
    }

    PCR::PartialCreditModel(const std::string &Id, uint Mscore, uint which_sub,
                            uint Nscales, double a, double b, const Vector &d,
                            const std::string &Name, bool id_d0)
        : Item(Id, Mscore, which_sub, Nscales, Name),
          ParamPolicy(new UnivParams(a), new UnivParams(b),
                      make_d_vec(d, id_d0)),
          PriorPolicy(),
          b_(Mscore + 2),
          eta_(Mscore + 1),
          X_(Mscore + 1, Mscore + 2),
          d0_is_fixed(true),
          which_subscale_(which_sub) {
      setup();
    }

    PCR::PartialCreditModel(const PCR &rhs)
        : Model(rhs),
          Item(rhs),
          ParamPolicy(rhs),
          PriorPolicy(rhs),
          b_(rhs.b_),
          eta_(rhs.eta_),
          X_(rhs.X_),
          d0_is_fixed(rhs.d0_is_fixed),
          beta_(rhs.beta_->clone()),
          beta_current(rhs.beta_current),
          a_current(rhs.a_current),
          b_current(rhs.b_current),
          d_current(rhs.d_current),
          which_subscale_(rhs.which_subscale_) {
      //      setup();
      setup_aliases();
      set_observers();
    }

    PCR *PCR::clone() const { return new PCR(*this); }

    uint PCR::which_subscale() const { return which_subscale_; }

    Ptr<UnivParams> PCR::A_prm(bool check) {
      if (check && !a_current) fill_abd();
      return ParamPolicy::prm1();
    }

    Ptr<UnivParams> PCR::B_prm(bool check) {
      if (check && !b_current) fill_abd();
      return ParamPolicy::prm2();
    }

    Ptr<ConstrainedVectorParams> PCR::D_prm(bool check) {
      if (check && !d_current) fill_abd();
      return ParamPolicy::prm3();
    }

    Ptr<ConstrainedVectorParams> PCR::Beta_prm(bool check) {
      if (check && !beta_current) fill_beta();
      return beta_;
    }

    const Ptr<UnivParams> PCR::A_prm(bool check) const {
      if (check && !a_current) fill_abd();
      return ParamPolicy::prm1();
    }

    const Ptr<UnivParams> PCR::B_prm(bool check) const {
      if (check && !b_current) fill_abd();
      return ParamPolicy::prm2();
    }

    const Ptr<ConstrainedVectorParams> PCR::D_prm(bool check) const {
      if (check && !d_current) fill_abd();
      return ParamPolicy::prm3();
    }

    const Ptr<ConstrainedVectorParams> PCR::Beta_prm(bool check) const {
      if (check && !beta_current) fill_beta();
      return beta_;
    }

    std::vector<Ptr<Params>> PCR::parameter_vector() {
      sync_params();
      return ParamPolicy::parameter_vector();
    }

    const std::vector<Ptr<Params>> PCR::parameter_vector() const {
      sync_params();
      return ParamPolicy::parameter_vector();
    }

    double PCR::a() const { return A_prm()->value(); }
    double PCR::b() const { return B_prm()->value(); }

    double PCR::d(uint m) const { return d()[m]; }
    const Vector &PCR::d() const { return D_prm()->value(); }

    void PCR::set_a(double A) {
      A_prm()->set(A);
      a_current = true;
    }
    void PCR::set_b(double B) {
      B_prm()->set(B);
      b_current = true;
    }
    void PCR::set_d(const Vector &D) {
      assert(D.size() == maxscore() + 1);  // ==nlevels
      D_prm()->set(D);
      d_current = true;
    }

    void PCR::fix_d0() { d0_is_fixed = true; }
    void PCR::free_d0() { d0_is_fixed = false; }
    bool PCR::is_d0_fixed() const { return d0_is_fixed; }

    void PCR::initialize_params() {
      set_a(1.0);
      set_b(1.0);
      Vector h = response_histogram();
      Vector b = beta();
      b[0] = 0;
      b.back() = 0.001;  // a should not be zero
      for (uint i = 1; i < h.size(); ++i) b[i] = log(h[i] / h[0]);
      set_beta(b);
    }

    void PCR::sync_params() const {
      if (beta_current) {
        if (!(a_current && b_current && d_current)) fill_abd();
      } else if (a_current && b_current && d_current) {
        if (!beta_current) fill_beta();
      } else {
        report_error("No current params in sync_params");
      }
    }

    const Vector &PCR::beta() const { return Beta_prm()->value(); }

    void PCR::set_beta(const Vector &b) {
      beta_->set(b);
      beta_current = true;
    }

    const Vector &PCR::fill_eta(const Vector &Theta) const {
      eta_ = X(Theta) * beta();
      return eta_;
    }

    const Matrix &PCR::X(const Vector &Theta) const {
      return X(Theta[which_subscale()]);
    }

    const Matrix &PCR::X(double theta) const {
      VectorView v(X_.last_col());
      v[0] = theta;
      for (uint i = 1; i < v.size(); ++i) v[i] = v[i - 1] + theta;
      return X_;
    }

    double PCR::response_prob(Response r, const Vector &Theta,
                              bool logsc) const {
      return response_prob(r->value(), Theta, logsc);
    }

    double PCR::response_prob(uint r, const Vector &Theta, bool logsc) const {
      fill_eta(Theta);
      double lognc = lse(eta_);
      double ans = eta_[r] - lognc;
      return logsc ? ans : exp(ans);
    }

    std::pair<double, double> PCR::theta_moments() const {
      double mean(0), var(0), n(0);
      for (auto &subject : subjects()) {
        increment_theta_moments(subject, mean, var, n);
      }
      if (n > 0) mean /= n;
      var -= n * mean * mean;
      if (n > 1) var /= n - 1;
      return std::make_pair(mean, var);
    }

    void PCR::increment_theta_moments(const Ptr<Subject> &s, double &m,
                                      double &v, double &n) const {
      double theta = (s->Theta())[which_subscale()];
      m += theta;
      v += theta * theta;
      n += 1.0;
    }

    std::ostream &PCR::display_item_params(std::ostream &out, bool) const {
      out << a() << " " << b() << " " << d() << " ";
      return out;
    }

    void PCR::impose_beta_constraint() { b_ = beta(); }

    void PCR::fill_beta(bool first_time) const {
      double a = first_time ? A_prm(false)->value() : this->a();
      double b = first_time ? B_prm(false)->value() : this->b();
      const Vector &d(first_time ? D_prm(false)->value() : this->d());

      uint M = maxscore();
      b_[0] = a * (d[0] - b);
      for (uint m = 1; m <= M; ++m) b_[m] = b_[m - 1] + a * (d[m] - b);
      b_.back() = a;

      beta_->set(b_);
      beta_current = true;
      set_abd_current();
    }

    void PCR::fill_abd() const {
      uint M = maxscore();
      const Vector &beta(this->beta());
      assert(beta.size() == M + 2);

      double A = beta.back();
      double MA = (M + 1) * A;
      double B = beta[M - 1] / (-MA);

      Vector D(M + 1);
      D[0] = beta[0] + B;

      double last = 0;
      for (uint m = 0; m < D.size(); ++m) {
        D[m] = (beta[m] - last) / A + B;
        last = beta[m];
      }

      a_prm->set(A);  // mutable pointers get around const-ness
      b_prm->set(B);
      d_prm->set(D);        // the 'set' operations modify beta_current
      beta_current = true;  // set it right again
      set_abd_current();
    }

    void PCR::set_abd_current() const {
      a_current = true;
      b_current = true;
      d_current = true;
    }

  }  // namespace IRT
}  // namespace BOOM
