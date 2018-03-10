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
#include <cstring>
#include <stdexcept>

#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"

#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

#include "numopt.hpp"
/* Nelder-Mead */
namespace BOOM {

  inline double Myf(const Vector &x, const Target &target) {
    const double big = 1.0e+35; /*a very large number*/
    double f = target(x);
    return std::isfinite(f) ? f : big;
  }

  inline bool keep_going(double ans, double ans2, double abstol, double reltol,
                         int maxit, int fit) {
    if (fit >= maxit) return true;
    double da = ans2 - ans;
    double abar = fabs(ans2 + ans);
    if (da / abar < reltol) return false;
    if (fabs(da) < abstol && abar < abstol)
      return false;
    else
      return true;
  }

  double nelder_mead_driver(Vector &Bvec, Vector &X, const Target &target,
                            double abstol, double intol, double alpha,
                            double bet, double gamm, int &fncount, int maxit) {
    double ans, ans2;
    int restarts(0);
    int maxrestart(20);
    int fcount = 0;
    do {
      ++restarts;
      if (restarts > maxrestart) report_error("too many restarts");
      fcount = 0;
      ans = nelder_mead(Bvec, X, target, abstol, intol, alpha, bet, gamm,
                        fcount, maxit);
      Bvec = X;
      fncount += fcount;
      X = 0;
      fcount = 0;
      ans2 = nelder_mead(Bvec, X, target, abstol, intol, alpha, bet, gamm,
                         fcount, maxit);
      Bvec = X;
      fncount += fcount;
    } while (keep_going(ans, ans2, abstol, intol, maxit, fcount));
    return ans2;
  }

  double nelder_mead(Vector &Bvec, Vector &X, const Target &target,
                     double abstol, double intol, double alpha, double bet,
                     double gamm, int &fncount, int maxit) {
    int n = Bvec.size();
    int C;
    bool calcvert, shrinkfail = false;
    double convtol, f;
    int funcount = 0, H, i, j, L = 0;
    int n1 = 0;
    double oldsize;
    double size, step, temp, trystep;
    double VH, VL, VR;
    double Fmin;

    if (maxit <= 0) {
      Fmin = target(Bvec);
      fncount = 0;
      return Fmin;
    }
    Matrix P(n + 1, n + 2);
    f = target(Bvec);
    if (!std::isfinite(f)) {
      // error("Function cannot be evaluated at initial parameters");
      ostringstream err;
      err << "Error in nelder_mead:  " << endl
          << "Function cannot be evaluated at initial parameters:" << endl
          << Bvec;
      report_error(err.str());
    } else {
      funcount = 1;
      convtol = intol * (fabs(f) + intol);
      n1 = n + 1;
      C = n + 2;
      P(n1 - 1, 0) = f;
      for (i = 0; i < n; i++) P(i, 0) = Bvec[i];

      L = 1;
      size = 0.0;

      step = 0.0;
      for (i = 0; i < n; i++) {
        if (0.1 * fabs(Bvec[i]) > step) step = 0.1 * fabs(Bvec[i]);
      }
      if (step == 0.0) step = 0.1;
      for (j = 2; j <= n1; j++) {
        for (i = 0; i < n; i++) P(i, j - 1) = Bvec[i];

        trystep = step;
        while (P(j - 2, j - 1) == Bvec[j - 2]) {
          P(j - 2, j - 1) = Bvec[j - 2] + trystep;
          trystep *= 10;
        }
        size += trystep;
      }
      oldsize = size;
      calcvert = true;
      shrinkfail = false;
      do {
        if (calcvert) {
          for (j = 0; j < n1; j++) {
            if (j + 1 != L) {
              for (i = 0; i < n; i++) Bvec[i] = P(i, j);
              //              f = fminfn(n, Bvec, ex);
              f = Myf(Bvec, target);
              funcount++;
              P(n1 - 1, j) = f;
            }
          }
          calcvert = false;
        }

        VL = P(n1 - 1, L - 1);
        VH = VL;
        H = L;

        for (j = 1; j <= n1; j++) {
          if (j != L) {
            f = P(n1 - 1, j - 1);
            if (f < VL) {
              L = j;
              VL = f;
            }
            if (f > VH) {
              H = j;
              VH = f;
            }
          }
        }

        if (VH > VL + convtol && VL > abstol) {
          for (i = 0; i < n; i++) {
            temp = -P(i, H - 1);
            for (j = 0; j < n1; j++) temp += P(i, j);
            P(i, C - 1) = temp / n;
          }
          for (i = 0; i < n; i++)
            Bvec[i] = (1.0 + alpha) * P(i, C - 1) - alpha * P(i, H - 1);
          //      f = fminfn(n, Bvec, ex);
          //      if (!R_FINITE(f)) f = big;
          f = Myf(Bvec, target);
          funcount++;
          VR = f;
          if (VR < VL) {
            P(n1 - 1, C - 1) = f;
            for (i = 0; i < n; i++) {
              f = gamm * Bvec[i] + (1 - gamm) * P(i, C - 1);
              P(i, C - 1) = Bvec[i];
              Bvec[i] = f;
            }
            f = Myf(Bvec, target);
            //      f = fminfn(n, Bvec, ex);
            //      if (!R_FINITE(f)) f = big;
            funcount++;
            if (f < VR) {
              for (i = 0; i < n; i++) P(i, H - 1) = Bvec[i];
              P(n1 - 1, H - 1) = f;
            } else {
              for (i = 0; i < n; i++) P(i, H - 1) = P(i, C - 1);
              P(n1 - 1, H - 1) = VR;
            }
          } else {
            if (VR < VH) {
              for (i = 0; i < n; i++) P(i, H - 1) = Bvec[i];
              P(n1 - 1, H - 1) = VR;
            }

            for (i = 0; i < n; i++)
              Bvec[i] = (1 - bet) * P(i, H - 1) + bet * P(i, C - 1);
            f = Myf(Bvec, target);
            //      f = fminfn(n, Bvec, ex);
            //      if (!R_FINITE(f)) f = big;
            funcount++;

            if (f < P(n1 - 1, H - 1)) {
              for (i = 0; i < n; i++) P(i, H - 1) = Bvec[i];
              P(n1 - 1, H - 1) = f;
            } else {
              if (VR >= VH) {
                calcvert = true;
                size = 0.0;
                for (j = 0; j < n1; j++) {
                  if (j + 1 != L) {
                    for (i = 0; i < n; i++) {
                      P(i, j) = bet * (P(i, j) - P(i, L - 1)) + P(i, L - 1);
                      size += fabs(P(i, j) - P(i, L - 1));
                    }
                  }
                }
                if (size < oldsize) {
                  shrinkfail = false;
                  oldsize = size;
                } else {
                  shrinkfail = true;
                }
              }
            }
          }
        }

      } while (!(VH <= VL + convtol || VL <= abstol || shrinkfail ||
                 funcount > maxit));
    }

    Fmin = P(n1 - 1, L - 1);
    for (i = 0; i < n; i++) X[i] = P(i, L - 1);
    if (shrinkfail) report_error("Nelder-Mead shrink failure");
    fncount = funcount;
    return Fmin;
  }
}  // namespace BOOM
