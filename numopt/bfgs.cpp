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

#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"

#include "TargetFun/TargetFun.hpp"
#include "numopt.hpp"

#include <cmath>
#include <iomanip>
#include <vector>

namespace BOOM {

  const double stepredn = 0.2;
  const double acctol = 0.0001;
  const double reltest = 10.0;

  /*  BFGS variable-metric method, based on Pascal code
      in J.C. Nash, `Compact Numerical Methods for Computers', 2nd edition,
      converted by p2c then re-crafted by B.D. Ripley */

  double bfgs(Vector &b, const Target &target, const dTarget &dtarget,
              int maxit, double abstol, double reltol, int &fncount,
              int &grcount, bool &fail, int trace_frequency) {
    bool trace = trace_frequency > 0;
    int nREPORT = trace_frequency;
    bool accpoint;
    double Fmin;
    int count, funcount, gradcount;
    double f, gradproj;
    int i, j, ilast, iter = 0;
    double s, steplength;
    double D1, D2;
    int n;
    int n0 = b.size();

    Vector g(n0);  // gradient
    fail = false;

    if (maxit <= 0) {
      Fmin = target(b);
      fncount = grcount = 0;
      return Fmin;
    }

    //    vector<int> l(n0);
    n = n0;
    Vector t(n);
    Vector X(n);
    Vector c(n);
    Matrix B(n, n);

    f = target(b);

    if (!std::isfinite(f)) {
      std::ostringstream err;
      err << "Non-fatal warning: initial value in bfgs is not finite" << endl
          << "Initial x = " << b << endl
          << "Initial f(x) = " << f << endl;
      report_warning(err.str());
      fail = true;
      return f;
    }

    Fmin = f;
    funcount = gradcount = 1;
    dtarget(b, g);
    iter++;
    ilast = gradcount;

    do {
      if (ilast == gradcount) {
        for (i = 0; i < n; i++) {
          for (j = 0; j < i; j++) B(i, j) = 0.0;
          B(i, i) = 1.0;
        }
      }
      X = b;
      c = g;
      gradproj = 0.0;
      for (i = 0; i < n; i++) {
        s = 0.0;
        for (j = 0; j <= i; j++) s -= B(i, j) * g[j];
        for (j = i + 1; j < n; j++) s -= B(j, i) * g[j];
        t[i] = s;
        gradproj += s * g[i];
      }

      if (gradproj < 0.0) { /* search direction is downhill */
        steplength = 1.0;
        accpoint = false;
        do {
          count = 0;
          for (i = 0; i < n; i++) {
            b[i] = X[i] + steplength * t[i];
            if (reltest + X[i] == reltest + b[i]) /* no change */
              count++;
          }
          if (count < n) {
            f = target(b);
            funcount++;
            accpoint = std::isfinite(f) &&
                       (f <= Fmin + gradproj * steplength * acctol);
            if (!accpoint) {
              steplength *= stepredn;
            }
          }
        } while (!(count == n || accpoint));
        // SLS... changed the following line, which appears to assume f>0
        //        bool enough = (f > abstol) &&
        bool enough = (fabs(f - Fmin) > abstol) &&
                      fabs(f - Fmin) > reltol * (fabs(Fmin) + reltol);
        /* stop if value if small or if relative change is low */
        if (!enough) {
          count = n;
          Fmin = f;
        }
        if (count < n) { /* making progress */
          Fmin = f;
          dtarget(b, g);
          gradcount++;
          iter++;
          D1 = 0.0;
          for (i = 0; i < n; i++) {
            t[i] = steplength * t[i];
            c[i] = g[i] - c[i];
            D1 += t[i] * c[i];
          }
          if (D1 > 0) {
            D2 = 0.0;
            for (i = 0; i < n; i++) {
              s = 0.0;
              for (j = 0; j <= i; j++) s += B(i, j) * c[j];
              for (j = i + 1; j < n; j++) s += B(j, i) * c[j];
              X[i] = s;
              D2 += s * c[i];
            }
            D2 = 1.0 + D2 / D1;
            for (i = 0; i < n; i++) {
              for (j = 0; j <= i; j++)
                B(i, j) += (D2 * t[i] * t[j] - X[i] * t[j] - t[i] * X[j]) / D1;
            }
          } else { /* D1 < 0 */
            ilast = gradcount;
          }
        } else { /* no progress */
          if (ilast < gradcount) {
            count = 0;
            ilast = gradcount;
          }
        }
      } else { /* uphill search */
        count = 0;
        if (ilast == gradcount)
          count = n;
        else
          ilast = gradcount;
        /* Resets unless has just been reset */
      }
      if (trace && (iter % nREPORT == 0)) {
        std::ostringstream msg;
        msg << "iter " << std::setw(4) << iter << " value " << f << endl;
        report_message(msg);
      }
      if (iter >= maxit) break;
      if (gradcount - ilast > 2 * n) ilast = gradcount; /* periodic restart */
    } while (count != n || ilast != gradcount);
    if (trace) {
      std::ostringstream msg;
      msg << "final value " << Fmin << endl;
      if (iter < maxit)
        msg << "converged" << endl;
      else
        msg << "stopped after " << iter << "iterations" << endl;
      report_message(msg);
    }
    fail = (iter < maxit) ? false : true;
    fncount = funcount;
    grcount = gradcount;
    return Fmin;
  }

}  // namespace BOOM
