// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

// code obtained from
// http://people.sc.fsu.edu/~jburkardt/cpp_src/asa047/asa047.html

#include "numopt/NelderMead.hpp"

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>

#include "LinAlg/Vector.hpp"
#include "cpputil/report_error.hpp"
#include "numopt.hpp"

namespace NelderMeadStatlib {
  using namespace BOOM;
  void nelmin(const Target &fn, int n, Vector &start, Vector &xmin,
              double *ynewlo, double reqmin, const Vector &step,
              int konvge,  // frequency of convergence checks
              int kcount,  // max function evaluations
              int *icount, int *numres, int *ifault);
}  // namespace NelderMeadStatlib

namespace BOOM {
  NelderMeadMinimizer::NelderMeadMinimizer(const Target &f)
      : f_(f),
        precision_(1e-6),
        default_step_size_(1.0),
        frequency_of_convergence_checks_(20),
        max_number_of_evaluations_(10000),
        number_of_evaluations_(0),
        number_of_restarts_(0),
        error_code_(0) {}

  void NelderMeadMinimizer::minimize(const Vector &starting_value) {
    starting_value_ = starting_value;
    minimizing_value_ = starting_value;
    if (stepsize_.size() != starting_value.size()) {
      stepsize_.resize(starting_value.size());
      stepsize_ = default_step_size_;
    }
    n_ = starting_value_.size();

    NelderMeadStatlib::nelmin(
        f_, n_, starting_value_, minimizing_value_, &minimum_, precision_,
        stepsize_, frequency_of_convergence_checks_, max_number_of_evaluations_,
        &number_of_evaluations_, &number_of_restarts_, &error_code_);
  }

  void NelderMeadMinimizer::set_stepsize(const Vector &stepsize) {
    stepsize_ = stepsize;
  }

  void NelderMeadMinimizer::set_convergence_check_frequency(int frequency) {
    if (frequency <= 0) {
      report_error("convergence_frequency must be positive.");
    }
    frequency_of_convergence_checks_ = frequency;
  }

  void NelderMeadMinimizer::set_precision(double precision) {
    if (precision <= 0) {
      report_error("precision must be positive");
    }
    precision_ = precision;
  }

  void NelderMeadMinimizer::set_evaluation_limit(int number_of_evalutations) {
    if (number_of_evalutations <= 0) {
      report_error("number_of_evalutations must be positive");
    }
    max_number_of_evaluations_ = number_of_evalutations;
  }

  double NelderMeadMinimizer::minimum() const { return minimum_; }

  const Vector &NelderMeadMinimizer::minimizing_value() const {
    return minimizing_value_;
  }

  bool NelderMeadMinimizer::success() const { return error_code_ == 0; }

  std::string NelderMeadMinimizer::error_message() const {
    if (error_code_ == 0) {
      return "success";
    } else if (error_code_ == 1) {
      return "precision_, n_, or frequency_of_convergence_checks_ "
             "has an illegal value.";
    } else if (error_code_ == 2) {
      return "max_number_of_evaluations_ exceeded.";
    } else {
      return "Unknown error code.";
    }
  }

  int NelderMeadMinimizer::number_of_restarts() const {
    return number_of_restarts_;
  }

  int NelderMeadMinimizer::number_of_evaluations() const {
    return number_of_evaluations_;
  }

}  // namespace BOOM

namespace NelderMeadStatlib {

  using namespace BOOM;

  void nelmin(const Target &fn, int n, Vector &start, Vector &xmin,
              double *ynewlo, double reqmin, const Vector &step, int konvge,
              int kcount, int *icount, int *numres, int *ifault)
  //  Purpose:
  //
  //    NELMIN minimizes a function using the Nelder-Mead algorithm.
  //
  //  Discussion:
  //
  //    This routine seeks the minimum value of a user-specified function.
  //
  //    Simplex function minimisation procedure due to Nelder+Mead(1965),
  //    as implemented by O'Neill(1971, Appl.Statist. 20, 338-45), with
  //    subsequent comments by Chambers+Ertel(1974, 23, 250-1), Benyon(1976,
  //    25, 97) and Hill(1978, 27, 380-2)
  //
  //    The function to be minimized must be defined by a function of
  //    the form
  //
  //      function fn ( x, f )
  //      double fn
  //      double x(*)
  //
  //    and the name of this subroutine must be declared EXTERNAL in the
  //    calling routine and passed as the argument FN.
  //
  //    This routine does not include a termination test using the
  //    fitting of a quadratic surface.
  //
  //  Licensing:
  //
  //    This code is distributed under the GNU LGPL license.
  //
  //  Modified:
  //
  //    27 February 2008
  //
  //
  //    Original FORTRAN77 version by R ONeill.
  //    C++ version by John Burkardt.
  //
  //  Reference:
  //
  //    John Nelder, Roger Mead,
  //    A simplex method for function minimization,
  //    Computer Journal,
  //    Volume 7, 1965, pages 308-313.
  //
  //    R ONeill,
  //    Algorithm AS 47:
  //    Function Minimization Using a Simplex Procedure,
  //    Applied Statistics,
  //    Volume 20, Number 3, 1971, pages 338-345.
  //
  //  Parameters:
  //
  //    Input, double FN ( double x[] ), the name of the routine which evaluates
  //    the function to be minimized.
  //
  //    Input, int N, the number of variables.
  //
  //    Input/output, double START[N].  On input, a starting point
  //    for the iteration.  On output, this data may have been overwritten.
  //
  //    Output, double XMIN[N], the coordinates of the point which
  //    is estimated to minimize the function.
  //
  //    Output, double YNEWLO, the minimum value of the function.
  //
  //    Input, double REQMIN, the terminating limit for the variance
  //    of function values.
  //
  //    Input, double STEP[N], determines the size and shape of the
  //    initial simplex.  The relative magnitudes of its elements should reflect
  //    the units of the variables.
  //
  //    Input, int KONVGE, the convergence check is carried out
  //    every KONVGE iterations.
  //
  //    Input, int KCOUNT, the maximum number of function
  //    evaluations.
  //
  //    Output, int *ICOUNT, the number of function evaluations
  //    used.
  //
  //    Output, int *NUMRES, the number of restarts.
  //
  //    Output, int *IFAULT, error indicator.
  //    0, no errors detected.
  //    1, REQMIN, N, or KONVGE has an illegal value.
  //    2, iteration terminated because KCOUNT was exceeded without convergence.
  //
  {
    double ccoeff = 0.5;
    double del;
    double dn;
    double dnn;
    double ecoeff = 2.0;
    double eps = 0.001;
    int i;
    int ihi;
    int ilo;
    int j;
    int jcount;
    int l;
    int nn;
    // double *p;
    // double *p2star;
    // double *pbar;
    // double *pstar;
    double rcoeff = 1.0;
    double rq;
    double x;
    //  double *y;
    double y2star;
    double ylo;
    double ystar;
    double z;
    //
    //  Check the input parameters.
    //
    if (reqmin <= 0.0) {
      *ifault = 1;
      return;
    }

    if (n < 1) {
      *ifault = 1;
      return;
    }

    if (konvge < 1) {
      *ifault = 1;
      return;
    }

    Vector p(n * (n + 1));
    Vector pstar(n);
    Vector p2star(n);
    Vector pbar(n);
    Vector y(n + 1);

    *icount = 0;
    *numres = 0;

    jcount = konvge;
    dn = (double)(n);
    nn = n + 1;
    dnn = (double)(nn);
    del = 1.0;
    rq = reqmin * dn;
    //
    //  Initial or restarted loop.
    //
    for (;;) {
      for (i = 0; i < n; i++) {
        p[i + n * n] = start[i];
      }
      y[n] = fn(start);
      *icount = *icount + 1;

      for (j = 0; j < n; j++) {
        x = start[j];
        start[j] = start[j] + step[j] * del;
        for (i = 0; i < n; i++) {
          p[i + j * n] = start[i];
        }
        y[j] = fn(start);
        *icount = *icount + 1;
        start[j] = x;
      }
      //
      //  The simplex construction is complete.
      //
      //  Find highest and lowest Y values.  YNEWLO = Y(IHI) indicates
      //  the vertex of the simplex to be replaced.
      //
      ylo = y[0];
      ilo = 0;

      for (i = 1; i < nn; i++) {
        if (y[i] < ylo) {
          ylo = y[i];
          ilo = i;
        }
      }
      //
      //  Inner loop.
      //
      for (;;) {
        if (kcount <= *icount) {
          break;
        }
        *ynewlo = y[0];
        ihi = 0;

        for (i = 1; i < nn; i++) {
          if (*ynewlo < y[i]) {
            *ynewlo = y[i];
            ihi = i;
          }
        }
        //
        //  Calculate PBAR, the centroid of the simplex vertices
        //  excepting the vertex with Y value YNEWLO.
        //
        for (i = 0; i < n; i++) {
          z = 0.0;
          for (j = 0; j < nn; j++) {
            z = z + p[i + j * n];
          }
          z = z - p[i + ihi * n];
          pbar[i] = z / dn;
        }
        //
        //  Reflection through the centroid.
        //
        for (i = 0; i < n; i++) {
          pstar[i] = pbar[i] + rcoeff * (pbar[i] - p[i + ihi * n]);
        }
        ystar = fn(pstar);
        *icount = *icount + 1;
        //
        //  Successful reflection, so extension.
        //
        if (ystar < ylo) {
          for (i = 0; i < n; i++) {
            p2star[i] = pbar[i] + ecoeff * (pstar[i] - pbar[i]);
          }
          y2star = fn(p2star);
          *icount = *icount + 1;
          //
          //  Check extension.
          //
          if (ystar < y2star) {
            for (i = 0; i < n; i++) {
              p[i + ihi * n] = pstar[i];
            }
            y[ihi] = ystar;
          }
          //
          //  Retain extension or contraction.
          //
          else {
            for (i = 0; i < n; i++) {
              p[i + ihi * n] = p2star[i];
            }
            y[ihi] = y2star;
          }
        }
        //
        //  No extension.
        //
        else {
          l = 0;
          for (i = 0; i < nn; i++) {
            if (ystar < y[i]) {
              l = l + 1;
            }
          }

          if (1 < l) {
            for (i = 0; i < n; i++) {
              p[i + ihi * n] = pstar[i];
            }
            y[ihi] = ystar;
          }
          //
          //  Contraction on the Y(IHI) side of the centroid.
          //
          else if (l == 0) {
            for (i = 0; i < n; i++) {
              p2star[i] = pbar[i] + ccoeff * (p[i + ihi * n] - pbar[i]);
            }
            y2star = fn(p2star);
            *icount = *icount + 1;
            //
            //  Contract the whole simplex.
            //
            if (y[ihi] < y2star) {
              for (j = 0; j < nn; j++) {
                for (i = 0; i < n; i++) {
                  p[i + j * n] = (p[i + j * n] + p[i + ilo * n]) * 0.5;
                  xmin[i] = p[i + j * n];
                }
                y[j] = fn(xmin);
                *icount = *icount + 1;
              }
              ylo = y[0];
              ilo = 0;

              for (i = 1; i < nn; i++) {
                if (y[i] < ylo) {
                  ylo = y[i];
                  ilo = i;
                }
              }
              continue;
            }
            //
            //  Retain contraction.
            //
            else {
              for (i = 0; i < n; i++) {
                p[i + ihi * n] = p2star[i];
              }
              y[ihi] = y2star;
            }
          }
          //
          //  Contraction on the reflection side of the centroid.
          //
          else if (l == 1) {
            for (i = 0; i < n; i++) {
              p2star[i] = pbar[i] + ccoeff * (pstar[i] - pbar[i]);
            }
            y2star = fn(p2star);
            *icount = *icount + 1;
            //
            //  Retain reflection?
            //
            if (y2star <= ystar) {
              for (i = 0; i < n; i++) {
                p[i + ihi * n] = p2star[i];
              }
              y[ihi] = y2star;
            } else {
              for (i = 0; i < n; i++) {
                p[i + ihi * n] = pstar[i];
              }
              y[ihi] = ystar;
            }
          }
        }
        //
        //  Check if YLO improved.
        //
        if (y[ihi] < ylo) {
          ylo = y[ihi];
          ilo = ihi;
        }
        jcount = jcount - 1;

        if (0 < jcount) {
          continue;
        }
        //
        //  Check to see if minimum reached.
        //
        if (*icount <= kcount) {
          jcount = konvge;

          z = 0.0;
          for (i = 0; i < nn; i++) {
            z = z + y[i];
          }
          x = z / dnn;

          z = 0.0;
          for (i = 0; i < nn; i++) {
            z = z + pow(y[i] - x, 2);
          }

          if (z <= rq) {
            break;
          }
        }
      }
      //
      //  Factorial tests to check that YNEWLO is a local minimum.
      //
      for (i = 0; i < n; i++) {
        xmin[i] = p[i + ilo * n];
      }
      *ynewlo = y[ilo];

      if (kcount < *icount) {
        *ifault = 2;
        break;
      }

      *ifault = 0;

      for (i = 0; i < n; i++) {
        del = step[i] * eps;
        xmin[i] = xmin[i] + del;
        z = fn(xmin);
        *icount = *icount + 1;
        if (z < *ynewlo) {
          *ifault = 2;
          break;
        }
        xmin[i] = xmin[i] - del - del;
        z = fn(xmin);
        *icount = *icount + 1;
        if (z < *ynewlo) {
          *ifault = 2;
          break;
        }
        xmin[i] = xmin[i] + del;
      }

      if (*ifault == 0) {
        break;
      }
      //
      //  Restart the procedure.
      //
      for (i = 0; i < n; i++) {
        start[i] = xmin[i];
      }
      del = eps;
      *numres = *numres + 1;
    }
  }
}  // namespace NelderMeadStatlib
