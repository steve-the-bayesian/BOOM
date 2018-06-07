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

#include <iostream>
#include <sstream>
#include <ostream>
#include "LinAlg/Matrix.hpp"  // includes Vector.hpp as well
#include "LinAlg/Vector.hpp"
#include "cpputil/math_utils.hpp"
#include "numopt.hpp"

namespace BOOM {
  using std::endl;

  namespace {
    inline bool BAD(double lcrit, double epsilon) {
      return (lcrit != lcrit)                             // nan
             || ((fabs(lcrit) > epsilon) && (lcrit < 0))  //
          ;
    }

    inline bool keep_going(double lcrit, double leps, int iteration,
                           int max_iterations, bool step_halving) {
      if (step_halving) {
        // If the last iteration used step halving we're not done.
        return true;
      } else if (iteration >= max_iterations) {
        // If the maximum number of iterations is exceeded it is time
        // to bail out.
        return false;
      } else if (lcrit > leps) {
        // If the convergence criterion exceeds epsilon then there is
        // more work to do.
        return true;
      } else {
        // Mission accomplished!
        return false;
      }
    }
  }  // namespace

  // ======================================================================*/
  // Newton-Raphson routine to MINIMIZE the target function tf.  If
  // you want to maximize tf consider calling max_nd2 instead.  theta
  // is the initial value, g and h will be returned.  This function
  // implements step_halving if an increase in the function value
  // occurs.
  //
  // Args:
  //   theta: On input, the starting value of the optimization
  //     algorithm.  On output, the minimizing value of the target
  //     function.
  //   gradient: Space to hold gradient values.  On exit this should
  //     be all 0's (or very nearly so).
  //   hessian: Space to hold the second derivative (Hessian) values.
  //     If this function was called to obtain the MLE or MAP for a
  //     statistical model then the inverse of the negative second
  //     derivative matrix is an estimate of the asymptotic variance.
  //   target:  The function to be minimized.
  //   function_count: On exit this will be set to a count of the
  //     number of times target() was called.
  //   leps: A small positive number.  When the decrease in function
  //     values is less than leps then the algorithm is considered to
  //     have converged.
  //   happy_ending: On exit this is true if the function converged
  //     without issue, and false otherwise.
  //   error_message: If happy_ending is false then an error message
  //     explaining the error will be reported here.  If happy_ending
  //     is true then this will be empty.
  //
  // Return:
  //   The value of target at the minimum.
  double newton_raphson_min(Vector &theta, Vector &gradient, Matrix &hessian,
                            const d2Target &target, int &function_count,
                            double leps, bool &happy_ending,
                            std::string &error_message) {
    function_count = 0;
    happy_ending = true;
    error_message = "";
    try {
      double loglike = 0, lcrit = 1 + leps;
      int iteration = 0, max_iterations = 30;
      int step_halving = 0, total_step_halving = 0;
      const int max_step_halving = 10, max_total_step_halving = 50;

      double oldloglike = target(theta, gradient, hessian);
      ++function_count;
      while (keep_going(lcrit, leps, iteration, max_iterations, step_halving)) {
        if (!gradient.all_finite() || !hessian.all_finite()) {
          std::ostringstream err;
          err << "The Newton-Raphson algorithm encountered values that "
              << "produced illegal derivatives.";
          error_message = err.str();
          happy_ending = false;
          return loglike;
        }

        ++iteration;
        Vector step = hessian.solve(gradient);
        theta -= step;
        double directional_derivative = gradient.dot(step);
        loglike = target(theta, gradient, hessian);
        ++function_count;
        lcrit = oldloglike - loglike;
        // Likelihood criterion should be positive if all is well.
        step_halving = 0;
        if (BAD(lcrit, leps / 2.0)) { /* step halving */
          if (std::isfinite(loglike)) {
            // Only check the directional derivative if the outcome of the
            // function evaluation was finite.  Otherwise it is likely to be the
            // case that the function bailed out early before all derivatives
            // could be computed, or else that at least one derivative element
            // is also non-finite.
            if (directional_derivative < 0) {
              // Mathematically it is impossible to have a negative directional
              // derivative, because step = -H.inv() * g so the directional
              // derivative is -g*Hinv*g, which is must be negative.  If code
              // gets here it is a sign that the target function was coded
              // incorrectly.
              if (fabs(directional_derivative) < leps) return loglike;
            }
          }
          ++total_step_halving;
          Vector oldtheta = theta + step;
          double step_scale_factor = 1.0;
          while (BAD(lcrit, leps / 2.0) &&
                 (step_halving++ <= max_step_halving)) {
            step_scale_factor /= 2.0;
            step *= step_scale_factor;  // halve step size
            theta = oldtheta - step;
            loglike = target(theta, gradient, hessian);
            ++function_count;
            lcrit = oldloglike - loglike;
          }
          if (!hessian.is_pos_def()) {
            happy_ending = false;
            std::ostringstream err;
            err << "The Hessian matrix is not positive definite in "
                << "newton_raphson_min." << endl
                << hessian << endl;
            error_message = err.str();
            return loglike;
          }
        }

        oldloglike = loglike;
        if ((step_halving > max_step_halving) ||
            (total_step_halving > max_total_step_halving)) {
          happy_ending = false;
          return loglike;
        }
      }
      return loglike;
    } catch (std::exception &e) {
      error_message =
          "Exception caught in newton_raphson_min.  "
          "Error message:\n";
      error_message += e.what();
    } catch (...) {
      error_message = "Unknown exception caught in newton_raphson_min.";
    }
    happy_ending = false;
    return infinity();
  }

}  // namespace BOOM
