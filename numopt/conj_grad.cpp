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
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

#include <stdexcept>
#include <vector>
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "numopt.hpp"

namespace BOOM {

  // Args:
  //   X: On input X contains the starting value for the algorithm.
  //     On output X is the minimizing value.
  //   Fmin:  The value of the function at the minimum;
  //   target:  A functor returning the value of the function to be optimized.
  //   dtarget: A functor that computes the gradient of the function
  //     to be optimized.
  //   abstol:  Absolute tolerance.
  //   intol:  Relative tolerance.
  //   method:  Which flavor of conjugate gradient algorithm should be used?
  //   fncount:  The number of times target was evaluated.
  //   gradient:  The number of times dtarget was evaluated.
  //   maxit:  The maximum number of iterations.
  //   error_message: Will be empty if the return value is true.
  //     Otherwise contains an error meessage explaining the type of
  //     error encountered.
  //
  // Returns:
  //   The value of target at the minimum.
  bool conj_grad(Vector &X, double &Fmin, const Target &target,
                 const dTarget &dtarget, double abstol, double intol,
                 ConjugateGradientMethod method, int &fncount, int &grcount,
                 int maxit, std::string &error_message) {
    const double reltest = 10.0;
    const double acctol = 0.0001;
    const double stepredn = 0.2;
    Vector Bvec = X;
    int n = Bvec.size();
    error_message = "";
    bool accpoint;
    int count, cycle, cyclimit;
    double f;
    double G1, G2, G3, gradproj;
    int funcount = 0, gradcount = 0, i;
    double newstep, oldstep, setstep, steplength = 1.0;
    double tol;

    if (maxit <= 0) {
      Fmin = target(Bvec);  // fminfn(n, Bvec, ex);
      fncount = grcount = 0;
      error_message = "The maximum number of iterations was negative.";
      return false;
    }

    Vector c(n);
    Vector g(n);
    Vector t(n);

    setstep = 1.7;
    cyclimit = n;
    tol = intol * n * sqrt(intol);

    f = target(Bvec);
    if (!std::isfinite(f)) {
      ostringstream err;
      err << "bad initial value: " << Bvec << " in conj_grad";
      error_message = err.str();
      return false;
    }

    Fmin = f;
    funcount = 1;
    gradcount = 0;
    do {
      t = 0.0;
      c = 0.0;

      cycle = 0;
      oldstep = 1.0;
      count = 0;
      do {
        cycle++;
        gradcount++;
        if (gradcount > maxit) {
          fncount = funcount;
          grcount = gradcount;
          error_message = "max_iter_exceeded in conj_grad";
          return false;
        }
        dtarget(Bvec, g);

        X = Bvec;
        G1 = 0;
        G2 = 0;
        switch (method) {
          case FletcherReeves: {
            G1 = g.normsq();
            G2 = c.normsq();
            break;
          }
          case PolakRibiere: {
            G1 = g.normsq() - g.dot(c);
            G2 = c.normsq();
            break;
          }
          case BealeSorenson: {
            G1 = g.normsq() - g.dot(c);
            G2 = t.dot(g) - t.dot(c);
            break;
          }
          default: {
            error_message =
                "Unknown ConjugateGradientMethod passed to "
                "conj_grad.";
            return false;
          }
        }
        c = g;

        if (G1 > tol) {
          G3 = G2 > 0.0 ? G1 / G2 : 1.0;
          t *= G3;
          t -= g;

          gradproj = t.dot(g);
          steplength = oldstep;

          accpoint = false;
          do {
            count = 0;
            for (i = 0; i < n; i++) {
              Bvec[i] = X[i] + steplength * t[i];
              if (reltest + X[i] == reltest + Bvec[i]) count++;
            }
            if (count < n) {
              f = target(Bvec);
              funcount++;
              accpoint = (std::isfinite(f) &&
                          f <= Fmin + gradproj * steplength * acctol);
              if (!accpoint) steplength *= stepredn;
            }
          } while (!(count == n || accpoint));
          if (count < n) {
            newstep = 2 * (f - Fmin - gradproj * steplength);
            if (newstep > 0) {
              newstep = -(gradproj * steplength * steplength / newstep);

              Bvec = X;
              Bvec.axpy(t, newstep);

              Fmin = f;
              f = target(Bvec);
              funcount++;
              if (f < Fmin)
                Fmin = f;
              else {
                Bvec = X;
                Bvec.axpy(t, steplength);
              }
            }
          }
        }
        oldstep = setstep * steplength;
        if (oldstep > 1.0) oldstep = 1.0;
      } while ((count != n) && (G1 > tol) && (cycle != cyclimit));

    } while ((cycle != 1) || ((count != n) && (G1 > tol) && Fmin > abstol));

    fncount = funcount;
    grcount = gradcount;
    return true;
  }
}  // namespace BOOM
