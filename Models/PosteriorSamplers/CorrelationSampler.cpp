// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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
#include "Models/PosteriorSamplers/CorrelationSampler.hpp"
#include <limits>
#include "LinAlg/Cholesky.hpp"
#include "Models/ParamTypes.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef MvnCorrelationSampler CS;
  CS::MvnCorrelationSampler(MvnModel *model, const Ptr<CorrelationModel> &prior,
                            RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), mod_(model), pri_(prior) {}
  //----------------------------------------------------------------------
  void CS::draw() {
    const Vector &mu(mod_->mu());
    int n = mu.size();
    Sumsq_ = mod_->suf()->center_sumsq(mu);
    df_ = mod_->suf()->n();
    Vector sigma = sqrt(diag(mod_->Sigma()));
    R_ = var2cor(mod_->Sigma());
    for (int i = 0; i < n; ++i) {
      Sumsq_.row(i) /= sigma[i];
      Sumsq_.col(i) /= sigma[i];
    }
    for (int i = 1; i < n; ++i) {
      i_ = i;
      for (int j = 0; j < i; ++j) {
        j_ = j;
        draw_one();
      }
    }
    for (int i = 0; i < sigma.size(); ++i) {
      R_.row(i) *= sigma[i];
      R_.col(i) *= sigma[i];
    }
    mod_->set_Sigma(R_);
  }
  //----------------------------------------------------------------------
  CS *CS::clone_to_new_host(Model *new_host) const {
    return new CS(dynamic_cast<MvnModel *>(new_host),
                  pri_->clone(),
                  rng());
  }
  //----------------------------------------------------------------------
  double CS::logpri() const { return pri_->logp(R_); }
  //----------------------------------------------------------------------
  double CS::Rdet(double r) {
    set_r(r);
    double ans = det(R_);
    return (ans);
  }
  //----------------------------------------------------------------------
  double CS::logp(double r) {
    set_r(r);

    Cholesky L(R_);
    if (!L.is_pos_def()) {
      return BOOM::negative_infinity();
    }
    double ans = pri_->logp(R_);
    ans += -.5 * (df_ + R_.nrow() + 1) * L.logdet();
    ans += -.5 * trace(L.solve(Sumsq_));
    return (ans);
  }
  //----------------------------------------------------------------------
  void CS::set_r(double r) {
    R_(i_, j_) = r;
    R_(j_, i_) = r;
  }
  //----------------------------------------------------------------------
  // univariate slice sampling to set each element
  void CS::draw_one() {
    double oldr = R_(i_, j_);
    double logp_star = logp(R_(i_, j_));
    double u = logp_star - rexp_mt(rng(), 1);
    find_limits();
    if (lo_ >= hi_) {
      set_r(0);
      return;
    }
    //    const double eps(100*std::numeric_limits<double>::epsilon());
    const double eps(1e-6);
    check_limits(oldr, eps);
    while (1) {
      double cand = runif_mt(rng(), lo_, hi_);
      double logp_cand = logp(cand);
      if (logp_cand > u) {  // found something inside slice
        set_r(cand);
        return;
      } else {  // contract slice
        if (cand > oldr) {
          hi_ = cand;
        } else {
          lo_ = cand;
        }
      }
      if (fabs(hi_ - lo_) < eps) {
        set_r(hi_);
        return;
      }
    }
  }

  void CS::check_limits(double oldr, double eps) {
    if (oldr < lo_ - eps || oldr > hi_ + eps) {
      std::ostringstream err;
      err << "Error:  original matrix is not positive definite "
          << "in CorrelationSampler::draw." << endl
          << "lo = " << lo_ << endl
          << "hi = " << hi_ << endl
          << "R(" << i_ << ", " << j_ << ") = " << oldr << endl;
      report_error(err.str());
    }
  }

  //----------------------------------------------------------------------
  // use the method from Barnard, Meng, and McCulloch (2000,
  // Statistica Sinica) to find the upper and lower limits for which
  // R(i,j) remains positive definite
  void CS::find_limits() {
    double f1 = Rdet(1.0);
    double f0 = Rdet(0.0);
    double fn = Rdet(-1);

    double a = .5 * (f1 + fn - 2 * f0);
    double b = .5 * (f1 - fn);
    double c = f0;

    double d2 = b * b - 4 * a * c;
    if (d2 < 0) {
      lo_ = 0;
      hi_ = 0;
      return;
    }
    double d = std::sqrt(d2);
    lo_ = .5 * (-b - d) / a;
    hi_ = .5 * (-b + d) / a;
    if (hi_ < lo_) std::swap(lo_, hi_);
    if (isnan(hi_) || isnan(lo_)) {
      ostringstream err;
      err << "illegal values in CS::find_limits:" << endl
          << "f1 = " << f1 << endl
          << "f0 = " << f0 << endl
          << "fn = " << fn << endl
          << "a = " << a << endl
          << "b = " << b << endl
          << "c = " << c << endl
          << "d = " << d << endl
          << "d2 = " << d2 << endl
          << "lo = " << lo_ << endl
          << "hi = " << hi_ << endl;
    }
  }
}  // namespace BOOM
