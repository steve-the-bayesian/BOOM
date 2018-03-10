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
#ifndef BOOM_SCALAR_LAPLACE_APPROXIMATION_
#define BOOM_SCALAR_LAPLACE_APPROXIMATION_
#include "TargetFun/TargetFun.hpp"

namespace BOOM {
  class ScalarLaplaceApproximation {
   public:
    // logf is the (potentially un-normalized) log density whose
    // normalizing constant and/or mean and variance are desired
    // logf must exist for the life of the ScalarLaplaceApproximation
    ScalarLaplaceApproximation(const d2ScalarTargetFun &logf,
                               double starting_value);
    double integral() const;      // integral of exp(logf) over real line
    double log_integral() const;  // stable version of log(integral())

    // the mean and variance can be computed based on numerical
    // derivatives of the cumulant_generating function at zero.  The
    // 'eps' argument is the step size used in the numerical
    // derivative.
    double mean(double eps = 1e-5) const;
    double variance(double eps = 1e-5) const;
    double cumulant_generating_function(double s) const;
    double dCGF(double s, double eps) const;  // derivative of cum_gen_fun
   private:
    void find_mode(double starting_value);

    double x_;  // location of the mode
    double h_;  // hessian evaluated at mode
    double y_;  // value at mode

    const d2ScalarTargetFun &logf_;
  };

}  // namespace BOOM
#endif  // BOOM_SCALAR_LAPLACE_APPROXIMATION_
