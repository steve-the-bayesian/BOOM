// Copyright 2018 Google LLC. All Rights Reserved.
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

#include <cmath>
#include <iostream>

#include "numopt.hpp"

#include <utility>
#include "LinAlg/Matrix.hpp"
#include "cpputil/report_error.hpp"
#include "numopt/Powell.hpp"

namespace BOOM {
  /*----------------------------------------------------------------------
    Maximizing functions of several variables.
    ----------------------------------------------------------------------*/
  double max_nd0(Vector &x, Target tf) {
    Negate f(std::move(tf));
    Vector wsp(x);
    int fc = 0;
    double ans =
        nelder_mead_driver(x, wsp, f, 1e-8, 1e-8, 1.0, .5, 2.0, fc, 1000);
    return ans * -1;
  }
  //======================================================================
  bool max_nd1_careful(Vector &x, double &function_value, Target f, dTarget dtf,
                       std::string &error_message, double epsilon,
                       int max_iterations, OptimizationMethod method) {
    dNegate negative_f(std::move(f), std::move(dtf));
    bool fail = false;
    int fcount = 0;
    int gcount = 0;
    int ntries = 0;
    int maxntries = 5;
    Vector original_x = x;
    switch (method) {
      case BFGS: {
        // Try to minimize negative_f
        function_value = bfgs(x, negative_f, negative_f, 200, epsilon, epsilon,
                              fcount, gcount, fail);
        if (!std::isfinite(function_value) || !x.all_finite()) {
          x = original_x;
        }
        break;
      }

      case ConjugateGradient: {
        fail = !conj_grad(x, function_value, negative_f, negative_f, epsilon,
                          epsilon, PolakRibiere, fcount, gcount, max_iterations,
                          error_message);
        if (!std::isfinite(function_value) || !x.all_finite()) {
          x = original_x;
        }
        break;
      }

      case Both: {
        // Start with conjugate gradient, but set convergence epsilon
        // larger than desired.
        fail = !conj_grad(x, function_value, negative_f, negative_f,
                          epsilon * 10, epsilon * 10, PolakRibiere, fcount,
                          gcount, max_iterations, error_message);
        if (!std::isfinite(function_value) || !x.all_finite()) {
          x = original_x;
        }
        // Polish off with bfgs near the mode.
        function_value = bfgs(x, negative_f, negative_f, 200, epsilon, epsilon,
                              fcount, gcount, fail);
        if (!std::isfinite(function_value) || !x.all_finite()) {
          x = original_x;
        }
        break;
      }
      default:
        error_message = "Unknown optimization method.";
        return false;
    }  // switch method

    while (fail && ntries < maxntries) {
      Vector g = x;
      nelder_mead_driver(x, g, Target(negative_f), 1e-5, 1e-5, 1.0, .5, 2.0,
                         fcount, 1000);
      fcount = gcount = 0;
      fail = false;
      if (method == BFGS) {
        function_value = bfgs(x, negative_f, negative_f, 200, 1e-8, 1e-8,
                              fcount, gcount, fail);
        if (!std::isfinite(function_value) || !x.all_finite()) {
          x = original_x;
        }
      } else if (method == ConjugateGradient || method == Both) {
        fail = !conj_grad(x, function_value, negative_f, negative_f, epsilon,
                          epsilon, PolakRibiere, fcount, gcount, max_iterations,
                          error_message);
        if (!std::isfinite(function_value) || !x.all_finite()) {
          x = original_x;
        }
      }
      ++ntries;
    }
    function_value *= -1;
    return !fail;
  }

  //======================================================================
  double max_nd1(Vector &x, Target target, dTarget differentiable_target,
                 double epsilon, int max_iterations,
                 OptimizationMethod method) {
    double ans;
    std::string error_message;
    max_nd1_careful(x, ans, std::move(target), std::move(differentiable_target),
                    error_message, epsilon, max_iterations, method);
    return ans;
  }

  //======================================================================
  double max_nd2(Vector &x, Vector &g, Matrix &h, Target f, dTarget df,
                 d2Target d2f, double leps) {
    double ans;
    std::string error_message;
    bool ok = max_nd2_careful(x, g, h, ans, std::move(f), std::move(df),
                              std::move(d2f), leps, error_message);
    if (!ok) {
      report_error(error_message);
    }
    return ans;
  }

  //======================================================================
  bool max_nd2_careful(Vector &x, Vector &g, Matrix &h, double &ans, Target f,
                       dTarget df, d2Target d2f, double leps,
                       std::string &error_message) {
    unsigned int ntries = 0, maxtries = 5;
    Vector original_x = x;
    d2Negate nd2f(std::move(f), std::move(df), std::move(d2f));
    /*------------ we should check that h is negative definite -----------*/
    int function_count = 0;
    bool happy = true;
    int gradient_count = 0;
    error_message = "";
    do {
      ans = newton_raphson_min(x, g, h, nd2f, function_count, leps, happy,
                               error_message);
      ++ntries;
      if (!happy) {
        // If the Newton-Raphson algorithm did not succeed, then
        // perhaps it was because of a poor starting value.  Try
        // finding the optimum using bfgs, then try Newton-Raphson
        // again.
        x = original_x;
        bool fail = false;
        double bfgs_answer = bfgs(x, nd2f, nd2f, 200, leps, leps,
                                  function_count, gradient_count, fail);
        // If bfgs succeeded and got the same answer as Newton Raphson
        // (and that answer was finite), then accept that answer.
        happy =
            (!fail) && std::isfinite(ans) && (fabs(bfgs_answer - ans) < leps);
      }
    } while (!happy && ntries < maxtries);

    if (ntries >= maxtries) {
      std::ostringstream err;
      err << "max_nd2 failed.   too many newton_raphson failures " << endl
          << "last error message was: " << endl
          << error_message;
      error_message = err.str();
      return false;
    }

    /* othewise we're okay */
    g *= -1.0;
    h *= -1.0;
    ans *= -1;
    return true;
  }

}  // namespace BOOM
