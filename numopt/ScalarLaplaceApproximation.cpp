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
#include "numopt/ScalarLaplaceApproximation.hpp"
#include "cpputil/Constants.hpp"
#include "numopt/ScalarNewtonMax.hpp"

namespace BOOM {
  typedef ScalarLaplaceApproximation SLA;
  SLA::ScalarLaplaceApproximation(const d2ScalarTargetFun &logf,
                                  double starting_value)
      : logf_(logf) {
    find_mode(starting_value);
  }
  //----------------------------------------------------------------------
  void SLA::find_mode(double starting_value) {
    x_ = starting_value;
    double g = 0;
    y_ = scalar_newton_max(logf_, x_, g, h_);
  }
  //----------------------------------------------------------------------
  double SLA::integral() const {
    double sigma = sqrt(fabs(1 / h_));
    return Constants::root_2pi * sigma * exp(y_);
  }
  //----------------------------------------------------------------------
  double SLA::log_integral() const {
    double sigma = sqrt(fabs(1 / h_));
    return y_ + log(sigma) + Constants::log_root_2pi;
  }
  //----------------------------------------------------------------------
  // target computing the moment/cumulant generating function
  class MgfTarget : public d2ScalarTargetFun {
   public:
    MgfTarget(const d2ScalarTargetFun &logf, double s) : logf_(logf), s_(s) {}

    double operator()(double u) const override { return logf_(u) + s_ * u; }

    double operator()(double u, double &g) const override {
      double ans = logf_(u, g) + s_ * u;
      g += s_;  // derivative is with respect to u
      return ans;
    }

    double operator()(double u, double &g, double &h) const override {
      double ans = logf_(u, g, h) + s_ * u;
      g += s_;  // derivative is with respect to u
      return ans;
    }

    double operator()(double u, double &g, double &h,
                      uint nderiv) const override {
      if (nderiv > 2) {
        return (*this)(u, g, h);
      } else if (nderiv == 1) {
        return (*this)(u, g);
      } else {
        return (*this)(u);
      }
    }

   private:
    const d2ScalarTargetFun &logf_;
    double s_;
  };
  //----------------------------------------------------------------------
  // approximate cumulant generating function
  // should be called with small values of s
  double SLA::cumulant_generating_function(double s) const {
    MgfTarget logf_s(logf_, s);
    ScalarLaplaceApproximation M(logf_s, x_);  // numerator of MGF
    return M.log_integral() - log_integral();
  }
  //----------------------------------------------------------------------
  // derivative of cumulant function
  double SLA::dCGF(double s, double dx) const {
    double dy =
        cumulant_generating_function(s + dx) - cumulant_generating_function(s);
    return dy / dx;
  }
  //----------------------------------------------------------------------
  // mean is the derivative of the cumulant generating function at s=0
  double SLA::mean(double eps) const { return dCGF(0, eps); }
  //----------------------------------------------------------------------
  // variance is the second derivative of the cumulant generating function at
  // s=0
  double SLA::variance(double eps) const {
    double dx = eps;
    return (dCGF(0 + dx, eps) - dCGF(0, eps)) / dx;
  }
}  // namespace BOOM
