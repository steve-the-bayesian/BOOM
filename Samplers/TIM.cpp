// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include "Samplers/TIM.hpp"
#include <functional>
#include "Samplers/MH_Proposals.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  TIM::TIM(const BOOM::Target &logf, const BOOM::dTarget &dlogf,
           const BOOM::d2Target &d2logf, double nu, RNG *rng)
      : MetropolisHastings(logf, 0, rng),
        prop_(0),
        nu_(nu),
        f_(logf),
        df_(dlogf),
        d2f_(d2logf),
        cand_(1),
        dummy_gradient_(0),
        dummy_Hessian_(0, 0),
        mode_is_fixed_(0),
        mode_has_been_found_(0) {}

  inline double TIM_empty_target(const Vector &) { return 1.0; }

  typedef std::function<double(const Vector &, Vector &, Matrix &, int)>
      FullTarget;

  TIM::TIM(const FullTarget &logf, double nu, RNG *rng)
      : MetropolisHastings(TIM_empty_target, 0, rng),
        prop_(0),
        nu_(nu),
        cand_(1),
        dummy_gradient_(0),
        dummy_Hessian_(0, 0),
        mode_is_fixed_(0),
        mode_has_been_found_(0) {
    f_ = [logf, this](const Vector &x) {
      return logf(x, this->dummy_gradient_, this->dummy_Hessian_, 0);
    };
    df_ = [logf, this](const Vector &x, Vector &gradient) -> double {
      return logf(x, gradient, this->dummy_Hessian_, 1);
    };
    d2f_ = [logf](const Vector &x, Vector &gradient,
                  Matrix &Hessian) -> double {
      return logf(x, gradient, Hessian, 2);
    };
    MetropolisHastings::set_target(f_);
  }

  Vector TIM::draw(const Vector &old) {
    check_proposal(old.size());
    if (!mode_has_been_found_ || !mode_is_fixed_) {
      bool ok = locate_mode(old);
      if (!ok) {
        report_failure(old);
      }
    }
    return MetropolisHastings::draw(old);
  }

  void TIM::report_failure(const Vector &old) {
    ostringstream err;
    Vector gradient(old.size());
    Matrix Hessian(old.size(), old.size());
    double value = d2f_(old, gradient, Hessian);
    err << "failed attempt to find mode in BOOM::TIM" << endl
        << "current parameter value is " << endl
        << old << endl
        << "target function value at this parameter is " << value << endl
        << "current gradient is " << gradient << endl
        << "hessian matrix is " << endl
        << Hessian << endl;
    report_error(err.str());
  }

  void TIM::fix_mode(bool yn) { mode_is_fixed_ = yn; }

  bool TIM::locate_mode(const Vector &old) {
    cand_ = old;
    Vector gradient = old;
    Matrix Hessian(old.size(), old.size());
    double max_value;
    std::string error_message;
    bool ok = max_nd2_careful(cand_, gradient, Hessian, max_value, f_, df_,
                              d2f_, 1e-5, error_message);

    if (!ok) {
      mode_has_been_found_ = false;
      return false;
    }
    Hessian *= -1;
    mode_has_been_found_ = true;
    check_proposal(old.size());
    prop_->set_mu(cand_);
    prop_->set_ivar(Hessian);
    return true;
  }

  void TIM::set_mode(const Vector &mode_location, const Matrix &precision) {
    prop_->set_mu(mode_location);
    prop_->set_ivar(precision);
    mode_has_been_found_ = true;
    mode_is_fixed_ = true;
  }

  const Vector &TIM::mode() const {
    if (!prop_) {
      report_error("need to call TIM::locate_mode() before calling TIM::mode");
    }
    return prop_->mode();
  }

  const SpdMatrix &TIM::ivar() const {
    if (!prop_) {
      report_error(
          "need to call TIM::locate_mode() before calling TIM::ivar()");
    }
    return prop_->ivar();
  }

  Ptr<MvtIndepProposal> TIM::create_proposal(int dim, double nu) {
    Vector mu(dim);
    SpdMatrix Sigma(dim);
    Sigma.set_diag(1.0);
    return new MvtIndepProposal(mu, Sigma, nu);
  }

  void TIM::check_proposal(int dim) {
    if (!!prop_) return;
    prop_ = create_proposal(dim, nu_);
    MetropolisHastings::set_proposal(prop_);
  }
}  // namespace BOOM
