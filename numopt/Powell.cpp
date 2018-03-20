/*
The following copyright notice refers to the additional code required
to convert Powell's original NEWUOA function to C++.  Copyright for
the original code (as translated by f2c) appears below.

  Copyright (C) 2012 Google LLC. All Rights Reserved.

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

#include "numopt/Powell.hpp"
#include <cmath>
#include "LinAlg/Vector.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "numopt.hpp"

using std::min;

namespace PowellNewUOAImpl {
  class NewUOATargetFun {
   public:
    NewUOATargetFun(const BOOM::Target &f) : f_(f), number_of_evaluations_(0) {}

    double operator()(long n, const double *x) {
      ++number_of_evaluations_;
      x_.resize(n);
      x_.assign(x, x + n);
      return f_(x_);
    }

    int number_of_evaluations() const { return number_of_evaluations_; }

   private:
    BOOM::Target f_;
    mutable BOOM::Vector x_;
    mutable int number_of_evaluations_;
  };
  //----------------------------------------------------------------------
  int newuoa_(NewUOATargetFun &target, long *n, long *npt, double *x,
              double *rhobeg, double *rhoend, long *iprint, long *maxfun,
              double *w);

  int newuob_(NewUOATargetFun &target, long *, long *, double *, double *,
              double *, long *, long *, double *, double *, double *, double *,
              double *, double *, double *, double *, double *, double *,
              long *, double *, double *, double *);

  int biglag_(long *, long *, double *, double *, double *, double *, long *,
              long *, long *, double *, double *, double *, double *, double *,
              double *, double *, double *);
  int bigden_(long *, long *, double *, double *, double *, double *, long *,
              long *, long *, long *, double *, double *, double *, double *,
              double *, double *, double *);
  int update_(long *, long *, double *, double *, long *, long *, double *,
              double *, long *, double *);

}  // namespace PowellNewUOAImpl

namespace BOOM {

  PowellMinimizer::PowellMinimizer(const Target &f)
      : f_(f),
        minimum_(infinity()),
        initial_stepsize_(1.0),
        desired_precision_(1e-6),
        number_of_interpolating_points_(-1),
        max_number_of_function_evaulations_(5000) {}

  void PowellMinimizer::minimize(const Vector &x) {
    minimizing_x_ = x;
    PowellNewUOAImpl::NewUOATargetFun target(f_);

    long n = x.size();

    if (number_of_interpolating_points_ < 0) {
      number_of_interpolating_points_ = 2 * n + 1;
    }

    double rho_begin = initial_stepsize_;
    double rho_end = desired_precision_;
    long iprint = 0;
    long npt = number_of_interpolating_points_;
    long wdim = (npt + 13) * (npt + n) + 3 * n * (n + 3) / 2 + 1;
    Vector w(wdim);
    newuoa_(target, &n, &npt, minimizing_x_.data(), &rho_begin, &rho_end,
            &iprint, &max_number_of_function_evaulations_, w.data());
    minimum_ = f_(minimizing_x_);
    number_of_function_evaluations_ = target.number_of_evaluations();
  }

  void PowellMinimizer::set_evaluation_limit(long limit) {
    if (limit < 0) {
      report_error(
          "The maximum number of function evaluations must be "
          "positive. in PowellMinimizer::set_evaluation_limit().");
    }
    max_number_of_function_evaulations_ = limit;
  }

  void PowellMinimizer::set_precision(double precision) {
    if (precision <= 0) {
      report_error(
          "Precision argument must be positive in "
          "PowellMinimizer::set_precision.");
    }
    desired_precision_ = precision;
  }

  void PowellMinimizer::set_initial_stepsize(double stepsize) {
    if (stepsize <= 0) {
      report_error(
          "Stepsize argument must be positive in "
          "PowellMinimizer::set_initial_stepsize.");
    }
    initial_stepsize_ = stepsize;
  }

}  // namespace BOOM

namespace PowellNewUOAImpl {
  // README file from NEWUOA.
  /***********************************************************************
       This is the Fortran version of NEWUOA. Its purpose is to seek
  the least value of a function F of several variables, when derivatives
  are not available, where F is specified by the user through a subroutine
  called CALFUN. The algorithm is intended to change the variables to values
  that are close to a local minimum of F. The user, however, should assume
  responsibility for finding out if the calculations are satisfactory, by
  considering carefully the values of F that occur. The method is described
  in the report "The NEWUOA software for unconstrained optimization without
  derivatives", which is available on the web at www.damtp.cam.ac.uk, where
  you have to click on Numerical Analysis and then on Reports, the number
  of the report being NA2004/08. Let N be the number of variables. The main
  new feature of the method is that quadratic models are updated using only
  about NPT=2N+1 interpolation conditions, the remaining freedom being taken
  up by minimizing the Frobenius norm of the change to the second derivative
  matrix of the model.

       The new software was developed from UOBYQA, which also forms quadratic
  models from interpolation conditions. That method requires NPT=(N+1)(N+2)/2
  conditions, however, because they have to define all the parameters of the
  model. The least Frobenius norm updating procedure with NPT=2N+1 is usually
  much more efficient when N is large, because the work of each iteration is
  much less than before, and in some experiments the number of calculations
  of the objective function seems to be only of magnitude N.

       The attachments in sequence are a suitable Makefile, followed by a main
  program and a CALFUN routine for the Chebyquad problems, in order to provide
  an example for testing. Then NEWUOA and its five auxiliary routines, namely
  NEWUOB, BIGDEN, BIGLAG, TRSAPP and UPDATE, are given. Finally, the computed
  output that the author obtained for the Chebyquad problems is listed.

       The way of calling NEWUOA should be clear from the Chebyquad example
  and from the comments of that subroutine. It is hoped that the software will
  be helpful to much future research and to many applications. There are no
  restrictions on or charges for its use. If you wish to refer to it, please
  cite the DAMTP report that is mentioned above, which has been submitted for
  publication in the proceedings of the 40th Workshop on Large Scale Nonlinear
  Optimization (Erice, Italy, 2004).

  December 16th, 2004                    M.J.D. Powell (mjdp@cam.ac.uk)
  ***********************************************************************/

  using std::atan;
  using std::cos;
  using std::max;
  using std::sin;
  using std::sqrt;

  /* newuoa.f -- translated by f2c (version 20090411).
     You must link the resulting object file with libf2c:
          on Microsoft Windows system, link with libf2c.lib;
          on Linux or Unix systems, link with .../path/to/libf2c.a -lm
          or, if you install libf2c.a in a standard place, with -lf2c -lm
          -- in that order, at the end of the command line, as in
                  cc *.o -lf2c -lm
          Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

                  http://www.netlib.org/f2c/libf2c.zip
  */

  int newuoa_(NewUOATargetFun &target, long *n, long *npt, double *x,
              double *rhobeg, double *rhoend, long *iprint, long *maxfun,
              double *w) {
    /* Local variables */
    static long id, np, iw, igq, ihq, ixb, ifv, ipq, ivl, ixn, ixo, ixp, ndim,
        nptm, ibmat, izmat;

    /*     This subroutine seeks the least value of a function of many
     * variables, */
    /*     by a trust region method that forms quadratic models by
     * interpolation. */
    /*     There can be some freedom in the interpolation conditions, which is
     */
    /*     taken up by minimizing the Frobenius norm of the change to the second
     */
    /*     derivative of the quadratic model, beginning with a zero matrix. The
     */
    /*     arguments of the subroutine are as follows. */

    /*     N must be set to the number of variables and must be at least two. */
    /*     NPT is the number of interpolation conditions. Its value must be in
     * the */
    /*       interval [N+2,(N+1)(N+2)/2]. */
    /*     Initial values of the variables must be set in X(1),X(2),...,X(N).
     * They */
    /*       will be changed to the values that give the least calculated F. */
    /*     RHOBEG and RHOEND must be set to the initial and final values of a
     * trust */
    /*       region radius, so both must be positive with RHOEND<=RHOBEG.
     * Typically */
    /*       RHOBEG should be about one tenth of the greatest expected change to
     * a */
    /*       variable, and RHOEND should indicate the accuracy that is required
     * in */
    /*       the final values of the variables. */
    /*     The value of IPRINT should be set to 0, 1, 2 or 3, which controls the
     */
    /*       amount of printing. Specifically, there is no output if IPRINT=0
     * and */
    /*       there is output only at the return if IPRINT=1. Otherwise, each new
     */
    /*       value of RHO is printed, with the best vector of variables so far
     * and */
    /*       the corresponding value of the objective function. Further, each
     * new */
    /*       value of F with its variables are output if IPRINT=3. */
    /*     MAXFUN must be set to an upper bound on the number of calls of
     * CALFUN. */
    /*     The array W will be used for working space. Its length must be at
     * least */
    /*     (NPT+13)*(NPT+N)+3*N*(N+3)/2. */

    /*     SUBROUTINE CALFUN (N,X,F) must be provided by the user. It must set F
     * to */
    /*     the value of the objective function for the variables
     * X(1),X(2),...,X(N). */

    /*     Partition the working space array, so that different parts of it can
     * be */
    /*     treated separately by the subroutine that performs the main
     * calculation. */

    /* Parameter adjustments */
    --w;
    --x;

    /* Function Body */
    np = *n + 1;
    nptm = *npt - np;
    if (*npt < *n + 2 || *npt > (*n + 2) * np / 2) {
      //        s_wsfe(&io___3);
      //        e_wsfe();
      goto L20;
    }
    ndim = *npt + *n;
    ixb = 1;
    ixo = ixb + *n;
    ixn = ixo + *n;
    ixp = ixn + *n;
    ifv = ixp + *n * *npt;
    igq = ifv + *npt;
    ihq = igq + *n;
    ipq = ihq + *n * np / 2;
    ibmat = ipq + *npt;
    izmat = ibmat + ndim * *n;
    id = izmat + *npt * nptm;
    ivl = id + *n;
    iw = ivl + ndim;

    /*     The above settings provide a partition of W for subroutine NEWUOB. */
    /*     The partition requires the first NPT*(NPT+N)+5*N*(N+3)/2 elements of
     */
    /*     W plus the space that is needed by the last array of NEWUOB. */

    newuob_(target, n, npt, &x[1], rhobeg, rhoend, iprint, maxfun, &w[ixb],
            &w[ixo], &w[ixn], &w[ixp], &w[ifv], &w[igq], &w[ihq], &w[ipq],
            &w[ibmat], &w[izmat], &ndim, &w[id], &w[ivl], &w[iw]);
  L20:
    return 0;
  } /* newuoa_ */

  //----------------------------------------------------------------------

  /* newuob.f -- translated by f2c (version 20090411).
     You must link the resulting object file with libf2c:
          on Microsoft Windows system, link with libf2c.lib;
          on Linux or Unix systems, link with .../path/to/libf2c.a -lm
          or, if you install libf2c.a in a standard place, with -lf2c -lm
          -- in that order, at the end of the command line, as in
                  cc *.o -lf2c -lm
          Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

                  http://www.netlib.org/f2c/libf2c.zip
  */

  /* Table of constant values */

  int newuob_(NewUOATargetFun &target, long *n, long *npt, double *x,
              double *rhobeg, double *rhoend, long *iprint, long *maxfun,
              double *xbase, double *xopt, double *xnew, double *xpt,
              double *fval, double *gq, double *hq, double *pq, double *bmat,
              double *zmat, long *ndim, double *d__, double *vlag, double *w) {
    /* Format strings */
    // static char fmt_320[] = "(/4x,\002Return from NEWUOA because CALFUN has "
    //         "been\002,\002 called MAXFUN times.\002)";
    // static char fmt_330[] = "(/4x,\002Function number\002,i6,\002    F =\002"
    //         ",1pd18.10,\002    The corresponding X is:\002/(2x,5d15.6))";
    // static char fmt_370[] = "(/4x,\002Return from NEWUOA because a trus"
    //         "t\002,\002 region step has failed to reduce Q.\002)";
    // static char fmt_500[] = "(5x)";
    // static char fmt_510[] = "(/4x,\002New RHO =\002,1pd11.4,5x,\002Number o"
    //         "f\002,\002 function values =\002,i6)";
    // static char fmt_520[] = "(4x,\002Least value of F =\002,1pd23.15,9x,\002"
    //         "The corresponding X is:\002/(2x,5d15.6))";
    // static char fmt_550[] = "(/4x,\002At the return from NEWUOA\002,5x,\002N"
    //         "umber of function values =\002,i6)";

    /* System generated locals */
    long xpt_dim1, xpt_offset, bmat_dim1, bmat_offset, zmat_dim1, zmat_offset,
        i__1, i__2, i__3;
    double d__1, d__2, d__3;

    /* Local variables */
    static double f;
    static long i__, j, k, ih, nf, nh, ip, jp;
    static double dx;
    static long np, nfm;
    static double one;
    static long idz;
    static double dsq, rho;
    static long ipt, jpt;
    static double sum, fbeg, diff, half, beta;
    static long nfmm;
    static double gisq;
    static long knew;
    static double temp, suma, sumb, fopt, bsum, gqsq;
    static long kopt, nptm;
    static double zero, xipt, xjpt, sumz, diffa, diffb, diffc, hdiag, alpha,
        delta, recip, reciq, fsave;
    static long ksave, nfsav, itemp;
    static double dnorm, ratio, dstep, tenth, vquad;
    static long ktemp;
    static double tempq;
    static long itest;
    static double rhosq;
    static double detrat, crvmin;
    static long nftest;
    static double distsq;
    extern int trsapp_(long *, long *, double *, double *, double *, double *,
                       double *, double *, double *, double *, double *,
                       double *, double *, double *);
    static double xoptsq;

    /* Fortran I/O blocks */
    // static cilist io___55 = { 0, 6, 0, fmt_320, 0 };
    // static cilist io___56 = { 0, 6, 0, fmt_330, 0 };
    // static cilist io___61 = { 0, 6, 0, fmt_370, 0 };
    // static cilist io___68 = { 0, 6, 0, fmt_500, 0 };
    // static cilist io___69 = { 0, 6, 0, fmt_510, 0 };
    // static cilist io___70 = { 0, 6, 0, fmt_520, 0 };
    // static cilist io___71 = { 0, 6, 0, fmt_550, 0 };
    // static cilist io___72 = { 0, 6, 0, fmt_520, 0 };

    /*     The arguments N, NPT, X, RHOBEG, RHOEND, IPRINT and MAXFUN are
     * identical */
    /*       to the corresponding arguments in SUBROUTINE NEWUOA. */
    /*     XBASE will hold a shift of origin that should reduce the
     * contributions */
    /*       from rounding errors to values of the model and Lagrange functions.
     */
    /*     XOPT will be set to the displacement from XBASE of the vector of */
    /*       variables that provides the least calculated F so far. */
    /*     XNEW will be set to the displacement from XBASE of the vector of */
    /*       variables for the current calculation of F. */
    /*     XPT will contain the interpolation point coordinates relative to
     * XBASE. */
    /*     FVAL will hold the values of F at the interpolation points. */
    /*     GQ will hold the gradient of the quadratic model at XBASE. */
    /*     HQ will hold the explicit second derivatives of the quadratic model.
     */
    /*     PQ will contain the parameters of the implicit second derivatives of
     */
    /*       the quadratic model. */
    /*     BMAT will hold the last N columns of H. */
    /*     ZMAT will hold the factorization of the leading NPT by NPT submatrix
     * of */
    /*       H, this factorization being ZMAT times Diag(DZ) times ZMAT^T, where
     */
    /*       the elements of DZ are plus or minus one, as specified by IDZ. */
    /*     NDIM is the first dimension of BMAT and has the value NPT+N. */
    /*     D is reserved for trial steps from XOPT. */
    /*     VLAG will contain the values of the Lagrange functions at a new point
     * X. */
    /*       They are part of a product that requires VLAG to be of length NDIM.
     */
    /*     The array W will be used for working space. Its length must be at
     * least */
    /*       10*NDIM = 10*(NPT+N). */

    /*     Set some constants. */

    /* Parameter adjustments */
    zmat_dim1 = *npt;
    zmat_offset = 1 + zmat_dim1;
    zmat -= zmat_offset;
    xpt_dim1 = *npt;
    xpt_offset = 1 + xpt_dim1;
    xpt -= xpt_offset;
    --x;
    --xbase;
    --xopt;
    --xnew;
    --fval;
    --gq;
    --hq;
    --pq;
    bmat_dim1 = *ndim;
    bmat_offset = 1 + bmat_dim1;
    bmat -= bmat_offset;
    --d__;
    --vlag;
    --w;

    /* Function Body */
    half = .5;
    one = 1.;
    tenth = .1;
    zero = 0.;
    np = *n + 1;
    nh = *n * np / 2;
    nptm = *npt - np;
    nftest = std::max<long>(*maxfun, 1);

    /*     Set the initial elements of XPT, BMAT, HQ, PQ and ZMAT to zero. */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
      xbase[j] = x[j];
      i__2 = *npt;
      for (k = 1; k <= i__2; ++k) {
        /* L10: */
        xpt[k + j * xpt_dim1] = zero;
      }
      i__2 = *ndim;
      for (i__ = 1; i__ <= i__2; ++i__) {
        /* L20: */
        bmat[i__ + j * bmat_dim1] = zero;
      }
    }
    i__2 = nh;
    for (ih = 1; ih <= i__2; ++ih) {
      /* L30: */
      hq[ih] = zero;
    }
    i__2 = *npt;
    for (k = 1; k <= i__2; ++k) {
      pq[k] = zero;
      i__1 = nptm;
      for (j = 1; j <= i__1; ++j) {
        /* L40: */
        zmat[k + j * zmat_dim1] = zero;
      }
    }

    /*     Begin the initialization procedure. NF becomes one more than the
     * number */
    /*     of function values so far. The coordinates of the displacement of the
     */
    /*     next initial interpolation point from XBASE are set in XPT(NF,.). */

    rhosq = *rhobeg * *rhobeg;
    recip = one / rhosq;
    reciq = sqrt(half) / rhosq;
    nf = 0;
  L50:
    nfm = nf;
    nfmm = nf - *n;
    ++nf;
    if (nfm <= *n << 1) {
      if (nfm >= 1 && nfm <= *n) {
        xpt[nf + nfm * xpt_dim1] = *rhobeg;
      } else if (nfm > *n) {
        xpt[nf + nfmm * xpt_dim1] = -(*rhobeg);
      }
    } else {
      itemp = (nfmm - 1) / *n;
      jpt = nfm - itemp * *n - *n;
      ipt = jpt + itemp;
      if (ipt > *n) {
        itemp = jpt;
        jpt = ipt - *n;
        ipt = itemp;
      }
      xipt = *rhobeg;
      if (fval[ipt + np] < fval[ipt + 1]) {
        xipt = -xipt;
      }
      xjpt = *rhobeg;
      if (fval[jpt + np] < fval[jpt + 1]) {
        xjpt = -xjpt;
      }
      xpt[nf + ipt * xpt_dim1] = xipt;
      xpt[nf + jpt * xpt_dim1] = xjpt;
    }

    /*     Calculate the next value of F, label 70 being reached immediately */
    /*     after this calculation. The least function value so far and its index
     */
    /*     are required. */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
      /* L60: */
      x[j] = xpt[nf + j * xpt_dim1] + xbase[j];
    }
    goto L310;
  L70:
    fval[nf] = f;
    if (nf == 1) {
      fbeg = f;
      fopt = f;
      kopt = 1;
    } else if (f < fopt) {
      fopt = f;
      kopt = nf;
    }

    /*     Set the nonzero initial elements of BMAT and the quadratic model in
     */
    /*     the cases when NF is at most 2*N+1. */

    if (nfm <= *n << 1) {
      if (nfm >= 1 && nfm <= *n) {
        gq[nfm] = (f - fbeg) / *rhobeg;
        if (*npt < nf + *n) {
          bmat[nfm * bmat_dim1 + 1] = -one / *rhobeg;
          bmat[nf + nfm * bmat_dim1] = one / *rhobeg;
          bmat[*npt + nfm + nfm * bmat_dim1] = -half * rhosq;
        }
      } else if (nfm > *n) {
        bmat[nf - *n + nfmm * bmat_dim1] = half / *rhobeg;
        bmat[nf + nfmm * bmat_dim1] = -half / *rhobeg;
        zmat[nfmm * zmat_dim1 + 1] = -reciq - reciq;
        zmat[nf - *n + nfmm * zmat_dim1] = reciq;
        zmat[nf + nfmm * zmat_dim1] = reciq;
        ih = nfmm * (nfmm + 1) / 2;
        temp = (fbeg - f) / *rhobeg;
        hq[ih] = (gq[nfmm] - temp) / *rhobeg;
        gq[nfmm] = half * (gq[nfmm] + temp);
      }

      /*     Set the off-diagonal second derivatives of the Lagrange functions
       * and */
      /*     the initial quadratic model. */

    } else {
      ih = ipt * (ipt - 1) / 2 + jpt;
      if (xipt < zero) {
        ipt += *n;
      }
      if (xjpt < zero) {
        jpt += *n;
      }
      zmat[nfmm * zmat_dim1 + 1] = recip;
      zmat[nf + nfmm * zmat_dim1] = recip;
      zmat[ipt + 1 + nfmm * zmat_dim1] = -recip;
      zmat[jpt + 1 + nfmm * zmat_dim1] = -recip;
      hq[ih] = (fbeg - fval[ipt + 1] - fval[jpt + 1] + f) / (xipt * xjpt);
    }
    if (nf < *npt) {
      goto L50;
    }

    /*     Begin the iterative procedure, because the initial model is complete.
     */

    rho = *rhobeg;
    delta = rho;
    idz = 1;
    diffa = zero;
    diffb = zero;
    itest = 0;
    xoptsq = zero;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      xopt[i__] = xpt[kopt + i__ * xpt_dim1];
      /* L80: */
      /* Computing 2nd power */
      d__1 = xopt[i__];
      xoptsq += d__1 * d__1;
    }
  L90:
    nfsav = nf;

    /*     Generate the next trust region step and test its length. Set KNEW */
    /*     to -1 if the purpose of the next F will be to improve the model. */

  L100:
    knew = 0;
    trsapp_(n, npt, &xopt[1], &xpt[xpt_offset], &gq[1], &hq[1], &pq[1], &delta,
            &d__[1], &w[1], &w[np], &w[np + *n], &w[np + (*n << 1)], &crvmin);
    dsq = zero;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      /* L110: */
      /* Computing 2nd power */
      d__1 = d__[i__];
      dsq += d__1 * d__1;
    }
    /* Computing MIN */
    d__1 = delta, d__2 = sqrt(dsq);
    dnorm = min(d__1, d__2);
    if (dnorm < half * rho) {
      knew = -1;
      delta = tenth * delta;
      ratio = -1.;
      if (delta <= rho * 1.5) {
        delta = rho;
      }
      if (nf <= nfsav + 2) {
        goto L460;
      }
      temp = crvmin * .125 * rho * rho;
      /* Computing MAX */
      d__1 = std::max(diffa, diffb);
      if (temp <= std::max<double>(d__1, diffc)) {
        goto L460;
      }
      goto L490;
    }

    /*     Shift XBASE if XOPT may be too far from XBASE. First make the changes
     */
    /*     to BMAT that do not depend on ZMAT. */

  L120:
    if (dsq <= xoptsq * .001) {
      tempq = xoptsq * .25;
      i__1 = *npt;
      for (k = 1; k <= i__1; ++k) {
        sum = zero;
        i__2 = *n;
        for (i__ = 1; i__ <= i__2; ++i__) {
          /* L130: */
          sum += xpt[k + i__ * xpt_dim1] * xopt[i__];
        }
        temp = pq[k] * sum;
        sum -= half * xoptsq;
        w[*npt + k] = sum;
        i__2 = *n;
        for (i__ = 1; i__ <= i__2; ++i__) {
          gq[i__] += temp * xpt[k + i__ * xpt_dim1];
          xpt[k + i__ * xpt_dim1] -= half * xopt[i__];
          vlag[i__] = bmat[k + i__ * bmat_dim1];
          w[i__] = sum * xpt[k + i__ * xpt_dim1] + tempq * xopt[i__];
          ip = *npt + i__;
          i__3 = i__;
          for (j = 1; j <= i__3; ++j) {
            /* L140: */
            bmat[ip + j * bmat_dim1] =
                bmat[ip + j * bmat_dim1] + vlag[i__] * w[j] + w[i__] * vlag[j];
          }
        }
      }

      /*     Then the revisions of BMAT that depend on ZMAT are calculated. */

      i__3 = nptm;
      for (k = 1; k <= i__3; ++k) {
        sumz = zero;
        i__2 = *npt;
        for (i__ = 1; i__ <= i__2; ++i__) {
          sumz += zmat[i__ + k * zmat_dim1];
          /* L150: */
          w[i__] = w[*npt + i__] * zmat[i__ + k * zmat_dim1];
        }
        i__2 = *n;
        for (j = 1; j <= i__2; ++j) {
          sum = tempq * sumz * xopt[j];
          i__1 = *npt;
          for (i__ = 1; i__ <= i__1; ++i__) {
            /* L160: */
            sum += w[i__] * xpt[i__ + j * xpt_dim1];
          }
          vlag[j] = sum;
          if (k < idz) {
            sum = -sum;
          }
          i__1 = *npt;
          for (i__ = 1; i__ <= i__1; ++i__) {
            /* L170: */
            bmat[i__ + j * bmat_dim1] += sum * zmat[i__ + k * zmat_dim1];
          }
        }
        i__1 = *n;
        for (i__ = 1; i__ <= i__1; ++i__) {
          ip = i__ + *npt;
          temp = vlag[i__];
          if (k < idz) {
            temp = -temp;
          }
          i__2 = i__;
          for (j = 1; j <= i__2; ++j) {
            /* L180: */
            bmat[ip + j * bmat_dim1] += temp * vlag[j];
          }
        }
      }

      /*     The following instructions complete the shift of XBASE, including
       */
      /*     the changes to the parameters of the quadratic model. */

      ih = 0;
      i__2 = *n;
      for (j = 1; j <= i__2; ++j) {
        w[j] = zero;
        i__1 = *npt;
        for (k = 1; k <= i__1; ++k) {
          w[j] += pq[k] * xpt[k + j * xpt_dim1];
          /* L190: */
          xpt[k + j * xpt_dim1] -= half * xopt[j];
        }
        i__1 = j;
        for (i__ = 1; i__ <= i__1; ++i__) {
          ++ih;
          if (i__ < j) {
            gq[j] += hq[ih] * xopt[i__];
          }
          gq[i__] += hq[ih] * xopt[j];
          hq[ih] = hq[ih] + w[i__] * xopt[j] + xopt[i__] * w[j];
          /* L200: */
          bmat[*npt + i__ + j * bmat_dim1] = bmat[*npt + j + i__ * bmat_dim1];
        }
      }
      i__1 = *n;
      for (j = 1; j <= i__1; ++j) {
        xbase[j] += xopt[j];
        /* L210: */
        xopt[j] = zero;
      }
      xoptsq = zero;
    }

    /*     Pick the model step if KNEW is positive. A different choice of D */
    /*     may be made later, if the choice of D by BIGLAG causes substantial */
    /*     cancellation in DENOM. */

    if (knew > 0) {
      biglag_(n, npt, &xopt[1], &xpt[xpt_offset], &bmat[bmat_offset],
              &zmat[zmat_offset], &idz, ndim, &knew, &dstep, &d__[1], &alpha,
              &vlag[1], &vlag[*npt + 1], &w[1], &w[np], &w[np + *n]);
    }

    /*     Calculate VLAG and BETA for the current choice of D. The first NPT */
    /*     components of W_check will be held in W. */

    i__1 = *npt;
    for (k = 1; k <= i__1; ++k) {
      suma = zero;
      sumb = zero;
      sum = zero;
      i__2 = *n;
      for (j = 1; j <= i__2; ++j) {
        suma += xpt[k + j * xpt_dim1] * d__[j];
        sumb += xpt[k + j * xpt_dim1] * xopt[j];
        /* L220: */
        sum += bmat[k + j * bmat_dim1] * d__[j];
      }
      w[k] = suma * (half * suma + sumb);
      /* L230: */
      vlag[k] = sum;
    }
    beta = zero;
    i__1 = nptm;
    for (k = 1; k <= i__1; ++k) {
      sum = zero;
      i__2 = *npt;
      for (i__ = 1; i__ <= i__2; ++i__) {
        /* L240: */
        sum += zmat[i__ + k * zmat_dim1] * w[i__];
      }
      if (k < idz) {
        beta += sum * sum;
        sum = -sum;
      } else {
        beta -= sum * sum;
      }
      i__2 = *npt;
      for (i__ = 1; i__ <= i__2; ++i__) {
        /* L250: */
        vlag[i__] += sum * zmat[i__ + k * zmat_dim1];
      }
    }
    bsum = zero;
    dx = zero;
    i__2 = *n;
    for (j = 1; j <= i__2; ++j) {
      sum = zero;
      i__1 = *npt;
      for (i__ = 1; i__ <= i__1; ++i__) {
        /* L260: */
        sum += w[i__] * bmat[i__ + j * bmat_dim1];
      }
      bsum += sum * d__[j];
      jp = *npt + j;
      i__1 = *n;
      for (k = 1; k <= i__1; ++k) {
        /* L270: */
        sum += bmat[jp + k * bmat_dim1] * d__[k];
      }
      vlag[jp] = sum;
      bsum += sum * d__[j];
      /* L280: */
      dx += d__[j] * xopt[j];
    }
    beta = dx * dx + dsq * (xoptsq + dx + dx + half * dsq) + beta - bsum;
    vlag[kopt] += one;

    /*     If KNEW is positive and if the cancellation in DENOM is unacceptable,
     */
    /*     then BIGDEN calculates an alternative model step, XNEW being used for
     */
    /*     working space. */

    if (knew > 0) {
      /* Computing 2nd power */
      d__1 = vlag[knew];
      temp = one + alpha * beta / (d__1 * d__1);
      if (fabs(temp) <= .8) {
        bigden_(n, npt, &xopt[1], &xpt[xpt_offset], &bmat[bmat_offset],
                &zmat[zmat_offset], &idz, ndim, &kopt, &knew, &d__[1], &w[1],
                &vlag[1], &beta, &xnew[1], &w[*ndim + 1], &w[*ndim * 6 + 1]);
      }
    }

    /*     Calculate the next value of the objective function. */

  L290:
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
      xnew[i__] = xopt[i__] + d__[i__];
      /* L300: */
      x[i__] = xbase[i__] + xnew[i__];
    }
    ++nf;
  L310:
    if (nf > nftest) {
      --nf;
      // if (*iprint > 0) {
      //     s_wsfe(&io___55);
      //     e_wsfe();
      // }
      goto L530;
    }
    f = target(*n, &x[1]);
    // if (*iprint == 3) {
    //     s_wsfe(&io___56);
    //     do_fio(&c__1, (char *)&nf, (ftnlen)sizeof(long));
    //     do_fio(&c__1, (char *)&f, (ftnlen)sizeof(double));
    //     i__2 = *n;
    //     for (i__ = 1; i__ <= i__2; ++i__) {
    //         do_fio(&c__1, (char *)&x[i__], (ftnlen)sizeof(double));
    //     }
    //     e_wsfe();
    // }
    if (nf <= *npt) {
      goto L70;
    }
    if (knew == -1) {
      goto L530;
    }

    /*     Use the quadratic model to predict the change in F due to the step D,
     */
    /*     and set DIFF to the error of this prediction. */

    vquad = zero;
    ih = 0;
    i__2 = *n;
    for (j = 1; j <= i__2; ++j) {
      vquad += d__[j] * gq[j];
      i__1 = j;
      for (i__ = 1; i__ <= i__1; ++i__) {
        ++ih;
        temp = d__[i__] * xnew[j] + d__[j] * xopt[i__];
        if (i__ == j) {
          temp = half * temp;
        }
        /* L340: */
        vquad += temp * hq[ih];
      }
    }
    i__1 = *npt;
    for (k = 1; k <= i__1; ++k) {
      /* L350: */
      vquad += pq[k] * w[k];
    }
    diff = f - fopt - vquad;
    diffc = diffb;
    diffb = diffa;
    diffa = fabs(diff);
    if (dnorm > rho) {
      nfsav = nf;
    }

    /*     Update FOPT and XOPT if the new F is the least value of the objective
     */
    /*     function so far. The branch when KNEW is positive occurs if D is not
     */
    /*     a trust region step. */

    fsave = fopt;
    if (f < fopt) {
      fopt = f;
      xoptsq = zero;
      i__1 = *n;
      for (i__ = 1; i__ <= i__1; ++i__) {
        xopt[i__] = xnew[i__];
        /* L360: */
        /* Computing 2nd power */
        d__1 = xopt[i__];
        xoptsq += d__1 * d__1;
      }
    }
    ksave = knew;
    if (knew > 0) {
      goto L410;
    }

    /*     Pick the next value of DELTA after a trust region step. */

    if (vquad >= zero) {
      // if (*iprint > 0) {
      //     s_wsfe(&io___61);
      //     e_wsfe();
      // }
      goto L530;
    }
    ratio = (f - fsave) / vquad;
    if (ratio <= tenth) {
      delta = half * dnorm;
    } else if (ratio <= .7) {
      /* Computing MAX */
      d__1 = half * delta;
      delta = std::max<double>(d__1, dnorm);
    } else {
      /* Computing MAX */
      d__1 = half * delta, d__2 = dnorm + dnorm;
      delta = std::max<double>(d__1, d__2);
    }
    if (delta <= rho * 1.5) {
      delta = rho;
    }

    /*     Set KNEW to the index of the next interpolation point to be deleted.
     */

    /* Computing MAX */
    d__2 = tenth * delta;
    /* Computing 2nd power */
    d__1 = std::max<double>(d__2, rho);
    rhosq = d__1 * d__1;
    ktemp = 0;
    detrat = zero;
    if (f >= fsave) {
      ktemp = kopt;
      detrat = one;
    }
    i__1 = *npt;
    for (k = 1; k <= i__1; ++k) {
      hdiag = zero;
      i__2 = nptm;
      for (j = 1; j <= i__2; ++j) {
        temp = one;
        if (j < idz) {
          temp = -one;
        }
        /* L380: */
        /* Computing 2nd power */
        d__1 = zmat[k + j * zmat_dim1];
        hdiag += temp * (d__1 * d__1);
      }
      /* Computing 2nd power */
      d__2 = vlag[k];
      temp = (d__1 = beta * hdiag + d__2 * d__2, fabs(d__1));
      distsq = zero;
      i__2 = *n;
      for (j = 1; j <= i__2; ++j) {
        /* L390: */
        /* Computing 2nd power */
        d__1 = xpt[k + j * xpt_dim1] - xopt[j];
        distsq += d__1 * d__1;
      }
      if (distsq > rhosq) {
        /* Computing 3rd power */
        d__1 = distsq / rhosq;
        temp *= d__1 * (d__1 * d__1);
      }
      if (temp > detrat && k != ktemp) {
        detrat = temp;
        knew = k;
      }
      /* L400: */
    }
    if (knew == 0) {
      goto L460;
    }

    /*     Update BMAT, ZMAT and IDZ, so that the KNEW-th interpolation point */
    /*     can be moved. Begin the updating of the quadratic model, starting */
    /*     with the explicit second derivative term. */

  L410:
    update_(n, npt, &bmat[bmat_offset], &zmat[zmat_offset], &idz, ndim,
            &vlag[1], &beta, &knew, &w[1]);
    fval[knew] = f;
    ih = 0;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      temp = pq[knew] * xpt[knew + i__ * xpt_dim1];
      i__2 = i__;
      for (j = 1; j <= i__2; ++j) {
        ++ih;
        /* L420: */
        hq[ih] += temp * xpt[knew + j * xpt_dim1];
      }
    }
    pq[knew] = zero;

    /*     Update the other second derivative parameters, and then the gradient
     */
    /*     vector of the model. Also include the new interpolation point. */

    i__2 = nptm;
    for (j = 1; j <= i__2; ++j) {
      temp = diff * zmat[knew + j * zmat_dim1];
      if (j < idz) {
        temp = -temp;
      }
      i__1 = *npt;
      for (k = 1; k <= i__1; ++k) {
        /* L440: */
        pq[k] += temp * zmat[k + j * zmat_dim1];
      }
    }
    gqsq = zero;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      gq[i__] += diff * bmat[knew + i__ * bmat_dim1];
      /* Computing 2nd power */
      d__1 = gq[i__];
      gqsq += d__1 * d__1;
      /* L450: */
      xpt[knew + i__ * xpt_dim1] = xnew[i__];
    }

    /*     If a trust region step makes a small change to the objective
     * function, */
    /*     then calculate the gradient of the least Frobenius norm interpolant
     * at */
    /*     XBASE, and store it in W, using VLAG for a vector of right hand
     * sides. */

    if (ksave == 0 && delta == rho) {
      if (fabs(ratio) > .01) {
        itest = 0;
      } else {
        i__1 = *npt;
        for (k = 1; k <= i__1; ++k) {
          /* L700: */
          vlag[k] = fval[k] - fval[kopt];
        }
        gisq = zero;
        i__1 = *n;
        for (i__ = 1; i__ <= i__1; ++i__) {
          sum = zero;
          i__2 = *npt;
          for (k = 1; k <= i__2; ++k) {
            /* L710: */
            sum += bmat[k + i__ * bmat_dim1] * vlag[k];
          }
          gisq += sum * sum;
          /* L720: */
          w[i__] = sum;
        }

        /*     Test whether to replace the new quadratic model by the least
         * Frobenius */
        /*     norm interpolant, making the replacement if the test is
         * satisfied. */

        ++itest;
        if (gqsq < gisq * 100.) {
          itest = 0;
        }
        if (itest >= 3) {
          i__1 = *n;
          for (i__ = 1; i__ <= i__1; ++i__) {
            /* L730: */
            gq[i__] = w[i__];
          }
          i__1 = nh;
          for (ih = 1; ih <= i__1; ++ih) {
            /* L740: */
            hq[ih] = zero;
          }
          i__1 = nptm;
          for (j = 1; j <= i__1; ++j) {
            w[j] = zero;
            i__2 = *npt;
            for (k = 1; k <= i__2; ++k) {
              /* L750: */
              w[j] += vlag[k] * zmat[k + j * zmat_dim1];
            }
            /* L760: */
            if (j < idz) {
              w[j] = -w[j];
            }
          }
          i__1 = *npt;
          for (k = 1; k <= i__1; ++k) {
            pq[k] = zero;
            i__2 = nptm;
            for (j = 1; j <= i__2; ++j) {
              /* L770: */
              pq[k] += zmat[k + j * zmat_dim1] * w[j];
            }
          }
          itest = 0;
        }
      }
    }
    if (f < fsave) {
      kopt = knew;
    }

    /*     If a trust region step has provided a sufficient decrease in F, then
     */
    /*     branch for another trust region calculation. The case KSAVE>0 occurs
     */
    /*     when the new function value was calculated by a model step. */

    if (f <= fsave + tenth * vquad) {
      goto L100;
    }
    if (ksave > 0) {
      goto L100;
    }

    /*     Alternatively, find out if the interpolation points are close enough
     */
    /*     to the best point so far. */

    knew = 0;
  L460:
    distsq = delta * 4. * delta;
    i__2 = *npt;
    for (k = 1; k <= i__2; ++k) {
      sum = zero;
      i__1 = *n;
      for (j = 1; j <= i__1; ++j) {
        /* L470: */
        /* Computing 2nd power */
        d__1 = xpt[k + j * xpt_dim1] - xopt[j];
        sum += d__1 * d__1;
      }
      if (sum > distsq) {
        knew = k;
        distsq = sum;
      }
      /* L480: */
    }

    /*     If KNEW is positive, then set DSTEP, and branch back for the next */
    /*     iteration, which will generate a "model step". */

    if (knew > 0) {
      /* Computing MAX */
      /* Computing MIN */
      d__2 = tenth * sqrt(distsq), d__3 = half * delta;
      d__1 = min(d__2, d__3);
      dstep = std::max<double>(d__1, rho);
      dsq = dstep * dstep;
      goto L120;
    }
    if (ratio > zero) {
      goto L100;
    }
    if (max(delta, dnorm) > rho) {
      goto L100;
    }

    /*     The calculations with the current value of RHO are complete. Pick the
     */
    /*     next values of RHO and DELTA. */

  L490:
    if (rho > *rhoend) {
      delta = half * rho;
      ratio = rho / *rhoend;
      if (ratio <= 16.) {
        rho = *rhoend;
      } else if (ratio <= 250.) {
        rho = sqrt(ratio) * *rhoend;
      } else {
        rho = tenth * rho;
      }
      delta = max(delta, rho);
      // if (*iprint >= 2) {
      //     if (*iprint >= 3) {
      //      s_wsfe(&io___68);
      //      e_wsfe();
      //     }
      //     s_wsfe(&io___69);
      //     do_fio(&c__1, (char *)&rho, (ftnlen)sizeof(double));
      //     do_fio(&c__1, (char *)&nf, (ftnlen)sizeof(long));
      //     e_wsfe();
      //     s_wsfe(&io___70);
      //     do_fio(&c__1, (char *)&fopt, (ftnlen)sizeof(double));
      //     i__2 = *n;
      //     for (i__ = 1; i__ <= i__2; ++i__) {
      //      d__1 = xbase[i__] + xopt[i__];
      //      do_fio(&c__1, (char *)&d__1, (ftnlen)sizeof(double));
      //     }
      //     e_wsfe();
      // }
      goto L90;
    }

    /*     Return from the calculation, after another Newton-Raphson step, if */
    /*     it is too short to have been tried before. */

    if (knew == -1) {
      goto L290;
    }
  L530:
    if (fopt <= f) {
      i__2 = *n;
      for (i__ = 1; i__ <= i__2; ++i__) {
        /* L540: */
        x[i__] = xbase[i__] + xopt[i__];
      }
      f = fopt;
    }
    // if (*iprint >= 1) {
    //     s_wsfe(&io___71);
    //     do_fio(&c__1, (char *)&nf, (ftnlen)sizeof(long));
    //     e_wsfe();
    //     s_wsfe(&io___72);
    //     do_fio(&c__1, (char *)&f, (ftnlen)sizeof(double));
    //     i__2 = *n;
    //     for (i__ = 1; i__ <= i__2; ++i__) {
    //         do_fio(&c__1, (char *)&x[i__], (ftnlen)sizeof(double));
    //     }
    //     e_wsfe();
    // }
    return 0;
  } /* newuob_ */

  //----------------------------------------------------------------------

  /* bigden.f -- translated by f2c (version 20090411).
     You must link the resulting object file with libf2c:
          on Microsoft Windows system, link with libf2c.lib;
          on Linux or Unix systems, link with .../path/to/libf2c.a -lm
          or, if you install libf2c.a in a standard place, with -lf2c -lm
          -- in that order, at the end of the command line, as in
                  cc *.o -lf2c -lm
          Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

                  http://www.netlib.org/f2c/libf2c.zip
  */

  int bigden_(long *n, long *npt, double *xopt, double *xpt, double *bmat,
              double *zmat, long *idz, long *ndim, long *kopt, long *knew,
              double *d__, double *w, double *vlag, double *beta, double *s,
              double *wvec, double *prod) {
    /* System generated locals */
    long xpt_dim1, xpt_offset, bmat_dim1, bmat_offset, zmat_dim1, zmat_offset,
        wvec_dim1, wvec_offset, prod_dim1, prod_offset, i__1, i__2;
    double d__1;

    /* Local variables */
    static long i__, j, k;
    static double dd;
    static long jc;
    static double ds;
    static long ip, iu, nw;
    static double ss, den[9], one, par[9], tau, sum, two, diff, half, temp;
    static long ksav;
    static double step;
    static long nptm;
    static double zero, alpha, angle, denex[9];
    static long iterc;
    static double tempa, tempb, tempc;
    static long isave;
    static double ssden, dtest, quart, xoptd, twopi, xopts, denold, denmax,
        densav, dstemp, sumold, sstemp, xoptsq;

    /*     N is the number of variables. */
    /*     NPT is the number of interpolation equations. */
    /*     XOPT is the best interpolation point so far. */
    /*     XPT contains the coordinates of the current interpolation points. */
    /*     BMAT provides the last N columns of H. */
    /*     ZMAT and IDZ give a factorization of the first NPT by NPT submatrix
     * of H. */
    /*     NDIM is the first dimension of BMAT and has the value NPT+N. */
    /*     KOPT is the index of the optimal interpolation point. */
    /*     KNEW is the index of the interpolation point that is going to be
     * moved. */
    /*     D will be set to the step from XOPT to the new point, and on entry it
     */
    /*       should be the D that was calculated by the last call of BIGLAG. The
     */
    /*       length of the initial D provides a trust region bound on the final
     * D. */
    /*     W will be set to Wcheck for the final choice of D. */
    /*     VLAG will be set to Theta*Wcheck+e_b for the final choice of D. */
    /*     BETA will be set to the value that will occur in the updating formula
     */
    /*       when the KNEW-th interpolation point is moved to its new position.
     */
    /*     S, WVEC, PROD and the private arrays DEN, DENEX and PAR will be used
     */
    /*       for working space. */

    /*     D is calculated in a way that should provide a denominator with a
     * large */
    /*     modulus in the updating formula when the KNEW-th interpolation point
     * is */
    /*     shifted to the new position XOPT+D. */

    /*     Set some constants. */

    /* Parameter adjustments */
    zmat_dim1 = *npt;
    zmat_offset = 1 + zmat_dim1;
    zmat -= zmat_offset;
    xpt_dim1 = *npt;
    xpt_offset = 1 + xpt_dim1;
    xpt -= xpt_offset;
    --xopt;
    prod_dim1 = *ndim;
    prod_offset = 1 + prod_dim1;
    prod -= prod_offset;
    wvec_dim1 = *ndim;
    wvec_offset = 1 + wvec_dim1;
    wvec -= wvec_offset;
    bmat_dim1 = *ndim;
    bmat_offset = 1 + bmat_dim1;
    bmat -= bmat_offset;
    --d__;
    --w;
    --vlag;
    --s;

    /* Function Body */
    half = .5;
    one = 1.;
    quart = .25;
    two = 2.;
    zero = 0.;
    twopi = atan(one) * 8.;
    nptm = *npt - *n - 1;

    /*     Store the first NPT elements of the KNEW-th column of H in W(N+1) */
    /*     to W(N+NPT). */

    i__1 = *npt;
    for (k = 1; k <= i__1; ++k) {
      /* L10: */
      w[*n + k] = zero;
    }
    i__1 = nptm;
    for (j = 1; j <= i__1; ++j) {
      temp = zmat[*knew + j * zmat_dim1];
      if (j < *idz) {
        temp = -temp;
      }
      i__2 = *npt;
      for (k = 1; k <= i__2; ++k) {
        /* L20: */
        w[*n + k] += temp * zmat[k + j * zmat_dim1];
      }
    }
    alpha = w[*n + *knew];

    /*     The initial search direction D is taken from the last call of BIGLAG,
     */
    /*     and the initial S is set below, usually to the direction from X_OPT
     */
    /*     to X_KNEW, but a different direction to an interpolation point may */
    /*     be chosen, in order to prevent S from being nearly parallel to D. */

    dd = zero;
    ds = zero;
    ss = zero;
    xoptsq = zero;
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
      /* Computing 2nd power */
      d__1 = d__[i__];
      dd += d__1 * d__1;
      s[i__] = xpt[*knew + i__ * xpt_dim1] - xopt[i__];
      ds += d__[i__] * s[i__];
      /* Computing 2nd power */
      d__1 = s[i__];
      ss += d__1 * d__1;
      /* L30: */
      /* Computing 2nd power */
      d__1 = xopt[i__];
      xoptsq += d__1 * d__1;
    }
    if (ds * ds > dd * .99 * ss) {
      ksav = *knew;
      dtest = ds * ds / ss;
      i__2 = *npt;
      for (k = 1; k <= i__2; ++k) {
        if (k != *kopt) {
          dstemp = zero;
          sstemp = zero;
          i__1 = *n;
          for (i__ = 1; i__ <= i__1; ++i__) {
            diff = xpt[k + i__ * xpt_dim1] - xopt[i__];
            dstemp += d__[i__] * diff;
            /* L40: */
            sstemp += diff * diff;
          }
          if (dstemp * dstemp / sstemp < dtest) {
            ksav = k;
            dtest = dstemp * dstemp / sstemp;
            ds = dstemp;
            ss = sstemp;
          }
        }
        /* L50: */
      }
      i__2 = *n;
      for (i__ = 1; i__ <= i__2; ++i__) {
        /* L60: */
        s[i__] = xpt[ksav + i__ * xpt_dim1] - xopt[i__];
      }
    }
    ssden = dd * ss - ds * ds;
    iterc = 0;
    densav = zero;

    /*     Begin the iteration by overwriting S with a vector that has the */
    /*     required length and direction. */

  L70:
    ++iterc;
    temp = one / sqrt(ssden);
    xoptd = zero;
    xopts = zero;
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
      s[i__] = temp * (dd * s[i__] - ds * d__[i__]);
      xoptd += xopt[i__] * d__[i__];
      /* L80: */
      xopts += xopt[i__] * s[i__];
    }

    /*     Set the coefficients of the first two terms of BETA. */

    tempa = half * xoptd * xoptd;
    tempb = half * xopts * xopts;
    den[0] = dd * (xoptsq + half * dd) + tempa + tempb;
    den[1] = two * xoptd * dd;
    den[2] = two * xopts * dd;
    den[3] = tempa - tempb;
    den[4] = xoptd * xopts;
    for (i__ = 6; i__ <= 9; ++i__) {
      /* L90: */
      den[i__ - 1] = zero;
    }

    /*     Put the coefficients of Wcheck in WVEC. */

    i__2 = *npt;
    for (k = 1; k <= i__2; ++k) {
      tempa = zero;
      tempb = zero;
      tempc = zero;
      i__1 = *n;
      for (i__ = 1; i__ <= i__1; ++i__) {
        tempa += xpt[k + i__ * xpt_dim1] * d__[i__];
        tempb += xpt[k + i__ * xpt_dim1] * s[i__];
        /* L100: */
        tempc += xpt[k + i__ * xpt_dim1] * xopt[i__];
      }
      wvec[k + wvec_dim1] = quart * (tempa * tempa + tempb * tempb);
      wvec[k + (wvec_dim1 << 1)] = tempa * tempc;
      wvec[k + wvec_dim1 * 3] = tempb * tempc;
      wvec[k + (wvec_dim1 << 2)] = quart * (tempa * tempa - tempb * tempb);
      /* L110: */
      wvec[k + wvec_dim1 * 5] = half * tempa * tempb;
    }
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
      ip = i__ + *npt;
      wvec[ip + wvec_dim1] = zero;
      wvec[ip + (wvec_dim1 << 1)] = d__[i__];
      wvec[ip + wvec_dim1 * 3] = s[i__];
      wvec[ip + (wvec_dim1 << 2)] = zero;
      /* L120: */
      wvec[ip + wvec_dim1 * 5] = zero;
    }

    /*     Put the coefficents of THETA*Wcheck in PROD. */

    for (jc = 1; jc <= 5; ++jc) {
      nw = *npt;
      if (jc == 2 || jc == 3) {
        nw = *ndim;
      }
      i__2 = *npt;
      for (k = 1; k <= i__2; ++k) {
        /* L130: */
        prod[k + jc * prod_dim1] = zero;
      }
      i__2 = nptm;
      for (j = 1; j <= i__2; ++j) {
        sum = zero;
        i__1 = *npt;
        for (k = 1; k <= i__1; ++k) {
          /* L140: */
          sum += zmat[k + j * zmat_dim1] * wvec[k + jc * wvec_dim1];
        }
        if (j < *idz) {
          sum = -sum;
        }
        i__1 = *npt;
        for (k = 1; k <= i__1; ++k) {
          /* L150: */
          prod[k + jc * prod_dim1] += sum * zmat[k + j * zmat_dim1];
        }
      }
      if (nw == *ndim) {
        i__1 = *npt;
        for (k = 1; k <= i__1; ++k) {
          sum = zero;
          i__2 = *n;
          for (j = 1; j <= i__2; ++j) {
            /* L160: */
            sum += bmat[k + j * bmat_dim1] * wvec[*npt + j + jc * wvec_dim1];
          }
          /* L170: */
          prod[k + jc * prod_dim1] += sum;
        }
      }
      i__1 = *n;
      for (j = 1; j <= i__1; ++j) {
        sum = zero;
        i__2 = nw;
        for (i__ = 1; i__ <= i__2; ++i__) {
          /* L180: */
          sum += bmat[i__ + j * bmat_dim1] * wvec[i__ + jc * wvec_dim1];
        }
        /* L190: */
        prod[*npt + j + jc * prod_dim1] = sum;
      }
    }

    /*     Include in DEN the part of BETA that depends on THETA. */

    i__1 = *ndim;
    for (k = 1; k <= i__1; ++k) {
      sum = zero;
      for (i__ = 1; i__ <= 5; ++i__) {
        par[i__ - 1] =
            half * prod[k + i__ * prod_dim1] * wvec[k + i__ * wvec_dim1];
        /* L200: */
        sum += par[i__ - 1];
      }
      den[0] = den[0] - par[0] - sum;
      tempa = prod[k + prod_dim1] * wvec[k + (wvec_dim1 << 1)] +
              prod[k + (prod_dim1 << 1)] * wvec[k + wvec_dim1];
      tempb = prod[k + (prod_dim1 << 1)] * wvec[k + (wvec_dim1 << 2)] +
              prod[k + (prod_dim1 << 2)] * wvec[k + (wvec_dim1 << 1)];
      tempc = prod[k + prod_dim1 * 3] * wvec[k + wvec_dim1 * 5] +
              prod[k + prod_dim1 * 5] * wvec[k + wvec_dim1 * 3];
      den[1] = den[1] - tempa - half * (tempb + tempc);
      den[5] -= half * (tempb - tempc);
      tempa = prod[k + prod_dim1] * wvec[k + wvec_dim1 * 3] +
              prod[k + prod_dim1 * 3] * wvec[k + wvec_dim1];
      tempb = prod[k + (prod_dim1 << 1)] * wvec[k + wvec_dim1 * 5] +
              prod[k + prod_dim1 * 5] * wvec[k + (wvec_dim1 << 1)];
      tempc = prod[k + prod_dim1 * 3] * wvec[k + (wvec_dim1 << 2)] +
              prod[k + (prod_dim1 << 2)] * wvec[k + wvec_dim1 * 3];
      den[2] = den[2] - tempa - half * (tempb - tempc);
      den[6] -= half * (tempb + tempc);
      tempa = prod[k + prod_dim1] * wvec[k + (wvec_dim1 << 2)] +
              prod[k + (prod_dim1 << 2)] * wvec[k + wvec_dim1];
      den[3] = den[3] - tempa - par[1] + par[2];
      tempa = prod[k + prod_dim1] * wvec[k + wvec_dim1 * 5] +
              prod[k + prod_dim1 * 5] * wvec[k + wvec_dim1];
      tempb = prod[k + (prod_dim1 << 1)] * wvec[k + wvec_dim1 * 3] +
              prod[k + prod_dim1 * 3] * wvec[k + (wvec_dim1 << 1)];
      den[4] = den[4] - tempa - half * tempb;
      den[7] = den[7] - par[3] + par[4];
      tempa = prod[k + (prod_dim1 << 2)] * wvec[k + wvec_dim1 * 5] +
              prod[k + prod_dim1 * 5] * wvec[k + (wvec_dim1 << 2)];
      /* L210: */
      den[8] -= half * tempa;
    }

    /*     Extend DEN so that it holds all the coefficients of DENOM. */

    sum = zero;
    for (i__ = 1; i__ <= 5; ++i__) {
      /* Computing 2nd power */
      d__1 = prod[*knew + i__ * prod_dim1];
      par[i__ - 1] = half * (d__1 * d__1);
      /* L220: */
      sum += par[i__ - 1];
    }
    denex[0] = alpha * den[0] + par[0] + sum;
    tempa = two * prod[*knew + prod_dim1] * prod[*knew + (prod_dim1 << 1)];
    tempb = prod[*knew + (prod_dim1 << 1)] * prod[*knew + (prod_dim1 << 2)];
    tempc = prod[*knew + prod_dim1 * 3] * prod[*knew + prod_dim1 * 5];
    denex[1] = alpha * den[1] + tempa + tempb + tempc;
    denex[5] = alpha * den[5] + tempb - tempc;
    tempa = two * prod[*knew + prod_dim1] * prod[*knew + prod_dim1 * 3];
    tempb = prod[*knew + (prod_dim1 << 1)] * prod[*knew + prod_dim1 * 5];
    tempc = prod[*knew + prod_dim1 * 3] * prod[*knew + (prod_dim1 << 2)];
    denex[2] = alpha * den[2] + tempa + tempb - tempc;
    denex[6] = alpha * den[6] + tempb + tempc;
    tempa = two * prod[*knew + prod_dim1] * prod[*knew + (prod_dim1 << 2)];
    denex[3] = alpha * den[3] + tempa + par[1] - par[2];
    tempa = two * prod[*knew + prod_dim1] * prod[*knew + prod_dim1 * 5];
    denex[4] = alpha * den[4] + tempa +
               prod[*knew + (prod_dim1 << 1)] * prod[*knew + prod_dim1 * 3];
    denex[7] = alpha * den[7] + par[3] - par[4];
    denex[8] = alpha * den[8] +
               prod[*knew + (prod_dim1 << 2)] * prod[*knew + prod_dim1 * 5];

    /*     Seek the value of the angle that maximizes the modulus of DENOM. */

    sum = denex[0] + denex[1] + denex[3] + denex[5] + denex[7];
    denold = sum;
    denmax = sum;
    isave = 0;
    iu = 49;
    temp = twopi / (double)(iu + 1);
    par[0] = one;
    i__1 = iu;
    for (i__ = 1; i__ <= i__1; ++i__) {
      angle = (double)i__ * temp;
      par[1] = cos(angle);
      par[2] = sin(angle);
      for (j = 4; j <= 8; j += 2) {
        par[j - 1] = par[1] * par[j - 3] - par[2] * par[j - 2];
        /* L230: */
        par[j] = par[1] * par[j - 2] + par[2] * par[j - 3];
      }
      sumold = sum;
      sum = zero;
      for (j = 1; j <= 9; ++j) {
        /* L240: */
        sum += denex[j - 1] * par[j - 1];
      }
      if (fabs(sum) > fabs(denmax)) {
        denmax = sum;
        isave = i__;
        tempa = sumold;
      } else if (i__ == isave + 1) {
        tempb = sum;
      }
      /* L250: */
    }
    if (isave == 0) {
      tempa = sum;
    }
    if (isave == iu) {
      tempb = denold;
    }
    step = zero;
    if (tempa != tempb) {
      tempa -= denmax;
      tempb -= denmax;
      step = half * (tempa - tempb) / (tempa + tempb);
    }
    angle = temp * ((double)isave + step);

    /*     Calculate the new parameters of the denominator, the new VLAG vector
     */
    /*     and the new D. Then test for convergence. */

    par[1] = cos(angle);
    par[2] = sin(angle);
    for (j = 4; j <= 8; j += 2) {
      par[j - 1] = par[1] * par[j - 3] - par[2] * par[j - 2];
      /* L260: */
      par[j] = par[1] * par[j - 2] + par[2] * par[j - 3];
    }
    *beta = zero;
    denmax = zero;
    for (j = 1; j <= 9; ++j) {
      *beta += den[j - 1] * par[j - 1];
      /* L270: */
      denmax += denex[j - 1] * par[j - 1];
    }
    i__1 = *ndim;
    for (k = 1; k <= i__1; ++k) {
      vlag[k] = zero;
      for (j = 1; j <= 5; ++j) {
        /* L280: */
        vlag[k] += prod[k + j * prod_dim1] * par[j - 1];
      }
    }
    tau = vlag[*knew];
    dd = zero;
    tempa = zero;
    tempb = zero;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      d__[i__] = par[1] * d__[i__] + par[2] * s[i__];
      w[i__] = xopt[i__] + d__[i__];
      /* Computing 2nd power */
      d__1 = d__[i__];
      dd += d__1 * d__1;
      tempa += d__[i__] * w[i__];
      /* L290: */
      tempb += w[i__] * w[i__];
    }
    if (iterc >= *n) {
      goto L340;
    }
    if (iterc > 1) {
      densav = max(densav, denold);
    }
    if (fabs(denmax) <= fabs(densav) * 1.1) {
      goto L340;
    }
    densav = denmax;

    /*     Set S to half the gradient of the denominator with respect to D. */
    /*     Then branch for the next iteration. */

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      temp = tempa * xopt[i__] + tempb * d__[i__] - vlag[*npt + i__];
      /* L300: */
      s[i__] = tau * bmat[*knew + i__ * bmat_dim1] + alpha * temp;
    }
    i__1 = *npt;
    for (k = 1; k <= i__1; ++k) {
      sum = zero;
      i__2 = *n;
      for (j = 1; j <= i__2; ++j) {
        /* L310: */
        sum += xpt[k + j * xpt_dim1] * w[j];
      }
      temp = (tau * w[*n + k] - alpha * vlag[k]) * sum;
      i__2 = *n;
      for (i__ = 1; i__ <= i__2; ++i__) {
        /* L320: */
        s[i__] += temp * xpt[k + i__ * xpt_dim1];
      }
    }
    ss = zero;
    ds = zero;
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
      /* Computing 2nd power */
      d__1 = s[i__];
      ss += d__1 * d__1;
      /* L330: */
      ds += d__[i__] * s[i__];
    }
    ssden = dd * ss - ds * ds;
    if (ssden >= dd * 1e-8 * ss) {
      goto L70;
    }

    /*     Set the vector W before the RETURN from the subroutine. */

  L340:
    i__2 = *ndim;
    for (k = 1; k <= i__2; ++k) {
      w[k] = zero;
      for (j = 1; j <= 5; ++j) {
        /* L350: */
        w[k] += wvec[k + j * wvec_dim1] * par[j - 1];
      }
    }
    vlag[*kopt] += one;
    return 0;
  } /* bigden_ */

  //----------------------------------------------------------------------

  /* biglag.f -- translated by f2c (version 20090411).
     You must link the resulting object file with libf2c:
          on Microsoft Windows system, link with libf2c.lib;
          on Linux or Unix systems, link with .../path/to/libf2c.a -lm
          or, if you install libf2c.a in a standard place, with -lf2c -lm
          -- in that order, at the end of the command line, as in
                  cc *.o -lf2c -lm
          Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

                  http://www.netlib.org/f2c/libf2c.zip
  */

  /* Subroutine */ int biglag_(long *n, long *npt, double *xopt, double *xpt,
                               double *bmat, double *zmat, long *idz,
                               long *ndim, long *knew, double *delta,
                               double *d__, double *alpha, double *hcol,
                               double *gc, double *gd, double *s, double *w) {
    /* System generated locals */
    long xpt_dim1, xpt_offset, bmat_dim1, bmat_offset, zmat_dim1, zmat_offset,
        i__1, i__2;
    double d__1;

    /* Local variables */
    static long i__, j, k;
    static double dd, gg;
    static long iu;
    static double sp, ss, cf1, cf2, cf3, cf4, cf5, dhd, cth, one, tau, sth, sum,
        half, temp, step;
    static long nptm;
    static double zero, angle, scale, denom;
    static long iterc, isave;
    static double delsq, tempa, tempb, twopi, taubeg, tauold, taumax;

    /*     N is the number of variables. */
    /*     NPT is the number of interpolation equations. */
    /*     XOPT is the best interpolation point so far. */
    /*     XPT contains the coordinates of the current interpolation points. */
    /*     BMAT provides the last N columns of H. */
    /*     ZMAT and IDZ give a factorization of the first NPT by NPT submatrix
     * of H. */
    /*     NDIM is the first dimension of BMAT and has the value NPT+N. */
    /*     KNEW is the index of the interpolation point that is going to be
     * moved. */
    /*     DELTA is the current trust region bound. */
    /*     D will be set to the step from XOPT to the new point. */
    /*     ALPHA will be set to the KNEW-th diagonal element of the H matrix. */
    /*     HCOL, GC, GD, S and W will be used for working space. */

    /*     The step D is calculated in a way that attempts to maximize the
     * modulus */
    /*     of LFUNC(XOPT+D), subject to the bound ||D|| .LE. DELTA, where LFUNC
     * is */
    /*     the KNEW-th Lagrange function. */

    /*     Set some constants. */

    /* Parameter adjustments */
    zmat_dim1 = *npt;
    zmat_offset = 1 + zmat_dim1;
    zmat -= zmat_offset;
    xpt_dim1 = *npt;
    xpt_offset = 1 + xpt_dim1;
    xpt -= xpt_offset;
    --xopt;
    bmat_dim1 = *ndim;
    bmat_offset = 1 + bmat_dim1;
    bmat -= bmat_offset;
    --d__;
    --hcol;
    --gc;
    --gd;
    --s;
    --w;

    /* Function Body */
    half = .5;
    one = 1.;
    zero = 0.;
    twopi = atan(one) * 8.;
    delsq = *delta * *delta;
    nptm = *npt - *n - 1;

    /*     Set the first NPT components of HCOL to the leading elements of the
     */
    /*     KNEW-th column of H. */

    iterc = 0;
    i__1 = *npt;
    for (k = 1; k <= i__1; ++k) {
      /* L10: */
      hcol[k] = zero;
    }
    i__1 = nptm;
    for (j = 1; j <= i__1; ++j) {
      temp = zmat[*knew + j * zmat_dim1];
      if (j < *idz) {
        temp = -temp;
      }
      i__2 = *npt;
      for (k = 1; k <= i__2; ++k) {
        /* L20: */
        hcol[k] += temp * zmat[k + j * zmat_dim1];
      }
    }
    *alpha = hcol[*knew];

    /*     Set the unscaled initial direction D. Form the gradient of LFUNC at
     */
    /*     XOPT, and multiply D by the second derivative matrix of LFUNC. */

    dd = zero;
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
      d__[i__] = xpt[*knew + i__ * xpt_dim1] - xopt[i__];
      gc[i__] = bmat[*knew + i__ * bmat_dim1];
      gd[i__] = zero;
      /* L30: */
      /* Computing 2nd power */
      d__1 = d__[i__];
      dd += d__1 * d__1;
    }
    i__2 = *npt;
    for (k = 1; k <= i__2; ++k) {
      temp = zero;
      sum = zero;
      i__1 = *n;
      for (j = 1; j <= i__1; ++j) {
        temp += xpt[k + j * xpt_dim1] * xopt[j];
        /* L40: */
        sum += xpt[k + j * xpt_dim1] * d__[j];
      }
      temp = hcol[k] * temp;
      sum = hcol[k] * sum;
      i__1 = *n;
      for (i__ = 1; i__ <= i__1; ++i__) {
        gc[i__] += temp * xpt[k + i__ * xpt_dim1];
        /* L50: */
        gd[i__] += sum * xpt[k + i__ * xpt_dim1];
      }
    }

    /*     Scale D and GD, with a sign change if required. Set S to another */
    /*     vector in the initial two dimensional subspace. */

    gg = zero;
    sp = zero;
    dhd = zero;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      /* Computing 2nd power */
      d__1 = gc[i__];
      gg += d__1 * d__1;
      sp += d__[i__] * gc[i__];
      /* L60: */
      dhd += d__[i__] * gd[i__];
    }
    scale = *delta / sqrt(dd);
    if (sp * dhd < zero) {
      scale = -scale;
    }
    temp = zero;
    if (sp * sp > dd * .99 * gg) {
      temp = one;
    }
    tau = scale * (fabs(sp) + half * scale * fabs(dhd));
    if (gg * delsq < tau * .01 * tau) {
      temp = one;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      d__[i__] = scale * d__[i__];
      gd[i__] = scale * gd[i__];
      /* L70: */
      s[i__] = gc[i__] + temp * gd[i__];
    }

    /*     Begin the iteration by overwriting S with a vector that has the */
    /*     required length and direction, except that termination occurs if */
    /*     the given D and S are nearly parallel. */

  L80:
    ++iterc;
    dd = zero;
    sp = zero;
    ss = zero;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      /* Computing 2nd power */
      d__1 = d__[i__];
      dd += d__1 * d__1;
      sp += d__[i__] * s[i__];
      /* L90: */
      /* Computing 2nd power */
      d__1 = s[i__];
      ss += d__1 * d__1;
    }
    temp = dd * ss - sp * sp;
    if (temp <= dd * 1e-8 * ss) {
      goto L160;
    }
    denom = sqrt(temp);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      s[i__] = (dd * s[i__] - sp * d__[i__]) / denom;
      /* L100: */
      w[i__] = zero;
    }

    /*     Calculate the coefficients of the objective function on the circle,
     */
    /*     beginning with the multiplication of S by the second derivative
     * matrix. */

    i__1 = *npt;
    for (k = 1; k <= i__1; ++k) {
      sum = zero;
      i__2 = *n;
      for (j = 1; j <= i__2; ++j) {
        /* L110: */
        sum += xpt[k + j * xpt_dim1] * s[j];
      }
      sum = hcol[k] * sum;
      i__2 = *n;
      for (i__ = 1; i__ <= i__2; ++i__) {
        /* L120: */
        w[i__] += sum * xpt[k + i__ * xpt_dim1];
      }
    }
    cf1 = zero;
    cf2 = zero;
    cf3 = zero;
    cf4 = zero;
    cf5 = zero;
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
      cf1 += s[i__] * w[i__];
      cf2 += d__[i__] * gc[i__];
      cf3 += s[i__] * gc[i__];
      cf4 += d__[i__] * gd[i__];
      /* L130: */
      cf5 += s[i__] * gd[i__];
    }
    cf1 = half * cf1;
    cf4 = half * cf4 - cf1;

    /*     Seek the value of the angle that maximizes the modulus of TAU. */

    taubeg = cf1 + cf2 + cf4;
    taumax = taubeg;
    tauold = taubeg;
    isave = 0;
    iu = 49;
    temp = twopi / (double)(iu + 1);
    i__2 = iu;
    for (i__ = 1; i__ <= i__2; ++i__) {
      angle = (double)i__ * temp;
      cth = cos(angle);
      sth = sin(angle);
      tau = cf1 + (cf2 + cf4 * cth) * cth + (cf3 + cf5 * cth) * sth;
      if (fabs(tau) > fabs(taumax)) {
        taumax = tau;
        isave = i__;
        tempa = tauold;
      } else if (i__ == isave + 1) {
        tempb = tau;
      }
      /* L140: */
      tauold = tau;
    }
    if (isave == 0) {
      tempa = tau;
    }
    if (isave == iu) {
      tempb = taubeg;
    }
    step = zero;
    if (tempa != tempb) {
      tempa -= taumax;
      tempb -= taumax;
      step = half * (tempa - tempb) / (tempa + tempb);
    }
    angle = temp * ((double)isave + step);

    /*     Calculate the new D and GD. Then test for convergence. */

    cth = cos(angle);
    sth = sin(angle);
    tau = cf1 + (cf2 + cf4 * cth) * cth + (cf3 + cf5 * cth) * sth;
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
      d__[i__] = cth * d__[i__] + sth * s[i__];
      gd[i__] = cth * gd[i__] + sth * w[i__];
      /* L150: */
      s[i__] = gc[i__] + gd[i__];
    }
    if (fabs(tau) <= fabs(taubeg) * 1.1) {
      goto L160;
    }
    if (iterc < *n) {
      goto L80;
    }
  L160:
    return 0;
  } /* biglag_ */

  //----------------------------------------------------------------------
  /* trsapp.f -- translated by f2c (version 20090411).
     You must link the resulting object file with libf2c:
          on Microsoft Windows system, link with libf2c.lib;
          on Linux or Unix systems, link with .../path/to/libf2c.a -lm
          or, if you install libf2c.a in a standard place, with -lf2c -lm
          -- in that order, at the end of the command line, as in
                  cc *.o -lf2c -lm
          Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

                  http://www.netlib.org/f2c/libf2c.zip
  */

  int trsapp_(long *n, long *npt, double *xopt, double *xpt, double *gq,
              double *hq, double *pq, double *delta, double *step, double *d__,
              double *g, double *hd, double *hs, double *crvmin) {
    /* System generated locals */
    long xpt_dim1, xpt_offset, i__1, i__2;
    double d__1, d__2;

    /* Local variables */
    static long i__, j, k;
    static double dd, cf, dg, gg;
    static long ih;
    static double ds, sg;
    static long iu;
    static double ss, dhd, dhs, cth, sgk, shs, sth, qadd, half, qbeg, qred,
        qmin, temp, qsav, qnew, zero, ggbeg, alpha, angle, reduc;
    static long iterc;
    static double ggsav, delsq, tempa, tempb;
    static long isave;
    static double bstep, ratio, twopi;
    static long itersw;
    static double angtest;
    static long itermax;

    /*     N is the number of variables of a quadratic objective function, Q
     * say. */
    /*     The arguments NPT, XOPT, XPT, GQ, HQ and PQ have their usual
     * meanings, */
    /*       in order to define the current quadratic model Q. */
    /*     DELTA is the trust region radius, and has to be positive. */
    /*     STEP will be set to the calculated trial step. */
    /*     The arrays D, G, HD and HS will be used for working space. */
    /*     CRVMIN will be set to the least curvature of H along the conjugate */
    /*       directions that occur, except that it is set to zero if STEP goes
     */
    /*       all the way to the trust region boundary. */

    /*     The calculation of STEP begins with the truncated conjugate gradient
     */
    /*     method. If the boundary of the trust region is reached, then further
     */
    /*     changes to STEP may be made, each one being in the 2D space spanned
     */
    /*     by the current STEP and the corresponding gradient of Q. Thus STEP */
    /*     should provide a substantial reduction to Q within the trust region.
     */

    /*     Initialization, which includes setting HD to H times XOPT. */

    /* Parameter adjustments */
    xpt_dim1 = *npt;
    xpt_offset = 1 + xpt_dim1;
    xpt -= xpt_offset;
    --xopt;
    --gq;
    --hq;
    --pq;
    --step;
    --d__;
    --g;
    --hd;
    --hs;

    /* Function Body */
    half = .5;
    zero = 0.;
    twopi = atan(1.) * 8.;
    delsq = *delta * *delta;
    iterc = 0;
    itermax = *n;
    itersw = itermax;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      /* L10: */
      d__[i__] = xopt[i__];
    }
    goto L170;

    /*     Prepare for the first line search. */

  L20:
    qred = zero;
    dd = zero;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      step[i__] = zero;
      hs[i__] = zero;
      g[i__] = gq[i__] + hd[i__];
      d__[i__] = -g[i__];
      /* L30: */
      /* Computing 2nd power */
      d__1 = d__[i__];
      dd += d__1 * d__1;
    }
    *crvmin = zero;
    if (dd == zero) {
      goto L160;
    }
    ds = zero;
    ss = zero;
    gg = dd;
    ggbeg = gg;

    /*     Calculate the step to the trust region boundary and the product HD.
     */

  L40:
    ++iterc;
    temp = delsq - ss;
    bstep = temp / (ds + sqrt(ds * ds + dd * temp));
    goto L170;
  L50:
    dhd = zero;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
      /* L60: */
      dhd += d__[j] * hd[j];
    }

    /*     Update CRVMIN and set the step-length ALPHA. */

    alpha = bstep;
    if (dhd > zero) {
      temp = dhd / dd;
      if (iterc == 1) {
        *crvmin = temp;
      }
      *crvmin = min(*crvmin, temp);
      /* Computing MIN */
      d__1 = alpha, d__2 = gg / dhd;
      alpha = min(d__1, d__2);
    }
    qadd = alpha * (gg - half * alpha * dhd);
    qred += qadd;

    /*     Update STEP and HS. */

    ggsav = gg;
    gg = zero;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      step[i__] += alpha * d__[i__];
      hs[i__] += alpha * hd[i__];
      /* L70: */
      /* Computing 2nd power */
      d__1 = g[i__] + hs[i__];
      gg += d__1 * d__1;
    }

    /*     Begin another conjugate direction iteration if required. */

    if (alpha < bstep) {
      if (qadd <= qred * .01) {
        goto L160;
      }
      if (gg <= ggbeg * 1e-4) {
        goto L160;
      }
      if (iterc == itermax) {
        goto L160;
      }
      temp = gg / ggsav;
      dd = zero;
      ds = zero;
      ss = zero;
      i__1 = *n;
      for (i__ = 1; i__ <= i__1; ++i__) {
        d__[i__] = temp * d__[i__] - g[i__] - hs[i__];
        /* Computing 2nd power */
        d__1 = d__[i__];
        dd += d__1 * d__1;
        ds += d__[i__] * step[i__];
        /* L80: */
        /* Computing 2nd power */
        d__1 = step[i__];
        ss += d__1 * d__1;
      }
      if (ds <= zero) {
        goto L160;
      }
      if (ss < delsq) {
        goto L40;
      }
    }
    *crvmin = zero;
    itersw = iterc;

    /*     Test whether an alternative iteration is required. */

  L90:
    if (gg <= ggbeg * 1e-4) {
      goto L160;
    }
    sg = zero;
    shs = zero;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      sg += step[i__] * g[i__];
      /* L100: */
      shs += step[i__] * hs[i__];
    }
    sgk = sg + shs;
    angtest = sgk / sqrt(gg * delsq);
    if (angtest <= -.99) {
      goto L160;
    }

    /*     Begin the alternative iteration by calculating D and HD and some */
    /*     scalar products. */

    ++iterc;
    temp = sqrt(delsq * gg - sgk * sgk);
    tempa = delsq / temp;
    tempb = sgk / temp;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      /* L110: */
      d__[i__] = tempa * (g[i__] + hs[i__]) - tempb * step[i__];
    }
    goto L170;
  L120:
    dg = zero;
    dhd = zero;
    dhs = zero;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      dg += d__[i__] * g[i__];
      dhd += hd[i__] * d__[i__];
      /* L130: */
      dhs += hd[i__] * step[i__];
    }

    /*     Seek the value of the angle that minimizes Q. */

    cf = half * (shs - dhd);
    qbeg = sg + cf;
    qsav = qbeg;
    qmin = qbeg;
    isave = 0;
    iu = 49;
    temp = twopi / (double)(iu + 1);
    i__1 = iu;
    for (i__ = 1; i__ <= i__1; ++i__) {
      angle = (double)i__ * temp;
      cth = cos(angle);
      sth = sin(angle);
      qnew = (sg + cf * cth) * cth + (dg + dhs * cth) * sth;
      if (qnew < qmin) {
        qmin = qnew;
        isave = i__;
        tempa = qsav;
      } else if (i__ == isave + 1) {
        tempb = qnew;
      }
      /* L140: */
      qsav = qnew;
    }
    if ((double)isave == zero) {
      tempa = qnew;
    }
    if (isave == iu) {
      tempb = qbeg;
    }
    angle = zero;
    if (tempa != tempb) {
      tempa -= qmin;
      tempb -= qmin;
      angle = half * (tempa - tempb) / (tempa + tempb);
    }
    angle = temp * ((double)isave + angle);

    /*     Calculate the new STEP and HS. Then test for convergence. */

    cth = cos(angle);
    sth = sin(angle);
    reduc = qbeg - (sg + cf * cth) * cth - (dg + dhs * cth) * sth;
    gg = zero;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      step[i__] = cth * step[i__] + sth * d__[i__];
      hs[i__] = cth * hs[i__] + sth * hd[i__];
      /* L150: */
      /* Computing 2nd power */
      d__1 = g[i__] + hs[i__];
      gg += d__1 * d__1;
    }
    qred += reduc;
    ratio = reduc / qred;
    if (iterc < itermax && ratio > .01) {
      goto L90;
    }
  L160:
    return 0;

    /*     The following instructions act as a subroutine for setting the vector
     */
    /*     HD to the vector D multiplied by the second derivative matrix of Q.
     */
    /*     They are called from three different places, which are distinguished
     */
    /*     by the value of ITERC. */

  L170:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
      /* L180: */
      hd[i__] = zero;
    }
    i__1 = *npt;
    for (k = 1; k <= i__1; ++k) {
      temp = zero;
      i__2 = *n;
      for (j = 1; j <= i__2; ++j) {
        /* L190: */
        temp += xpt[k + j * xpt_dim1] * d__[j];
      }
      temp *= pq[k];
      i__2 = *n;
      for (i__ = 1; i__ <= i__2; ++i__) {
        /* L200: */
        hd[i__] += temp * xpt[k + i__ * xpt_dim1];
      }
    }
    ih = 0;
    i__2 = *n;
    for (j = 1; j <= i__2; ++j) {
      i__1 = j;
      for (i__ = 1; i__ <= i__1; ++i__) {
        ++ih;
        if (i__ < j) {
          hd[j] += hq[ih] * d__[i__];
        }
        /* L210: */
        hd[i__] += hq[ih] * d__[j];
      }
    }
    if (iterc == 0) {
      goto L20;
    }
    if (iterc <= itersw) {
      goto L50;
    }
    goto L120;
  } /* trsapp_ */

  //----------------------------------------------------------------------

  /* update.f -- translated by f2c (version 20090411).
     You must link the resulting object file with libf2c:
          on Microsoft Windows system, link with libf2c.lib;
          on Linux or Unix systems, link with .../path/to/libf2c.a -lm
          or, if you install libf2c.a in a standard place, with -lf2c -lm
          -- in that order, at the end of the command line, as in
                  cc *.o -lf2c -lm
          Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

                  http://www.netlib.org/f2c/libf2c.zip
  */

  int update_(long *n, long *npt, double *bmat, double *zmat, long *idz,
              long *ndim, double *vlag, double *beta, long *knew, double *w) {
    /* System generated locals */
    long bmat_dim1, bmat_offset, zmat_dim1, zmat_offset, i__1, i__2;
    double d__1, d__2;

    /* Local variables */
    static long i__, j, ja, jb, jl, jp;
    static double one, tau, temp;
    static long nptm;
    static double zero;
    static long iflag;
    static double scala, scalb, alpha, denom, tempa, tempb, tausq;

    /*     The arrays BMAT and ZMAT with IDZ are updated, in order to shift the
     */
    /*     interpolation point that has index KNEW. On entry, VLAG contains the
     */
    /*     components of the vector Theta*Wcheck+e_b of the updating formula */
    /*     (6.11), and BETA holds the value of the parameter that has this name.
     */
    /*     The vector W is used for working space. */

    /*     Set some constants. */

    /* Parameter adjustments */
    zmat_dim1 = *npt;
    zmat_offset = 1 + zmat_dim1;
    zmat -= zmat_offset;
    bmat_dim1 = *ndim;
    bmat_offset = 1 + bmat_dim1;
    bmat -= bmat_offset;
    --vlag;
    --w;

    /* Function Body */
    one = 1.;
    zero = 0.;
    nptm = *npt - *n - 1;

    /*     Apply the rotations that put zeros in the KNEW-th row of ZMAT. */

    jl = 1;
    i__1 = nptm;
    for (j = 2; j <= i__1; ++j) {
      if (j == *idz) {
        jl = *idz;
      } else if (zmat[*knew + j * zmat_dim1] != zero) {
        /* Computing 2nd power */
        d__1 = zmat[*knew + jl * zmat_dim1];
        /* Computing 2nd power */
        d__2 = zmat[*knew + j * zmat_dim1];
        temp = sqrt(d__1 * d__1 + d__2 * d__2);
        tempa = zmat[*knew + jl * zmat_dim1] / temp;
        tempb = zmat[*knew + j * zmat_dim1] / temp;
        i__2 = *npt;
        for (i__ = 1; i__ <= i__2; ++i__) {
          temp = tempa * zmat[i__ + jl * zmat_dim1] +
                 tempb * zmat[i__ + j * zmat_dim1];
          zmat[i__ + j * zmat_dim1] = tempa * zmat[i__ + j * zmat_dim1] -
                                      tempb * zmat[i__ + jl * zmat_dim1];
          /* L10: */
          zmat[i__ + jl * zmat_dim1] = temp;
        }
        zmat[*knew + j * zmat_dim1] = zero;
      }
      /* L20: */
    }

    /*     Put the first NPT components of the KNEW-th column of HLAG into W, */
    /*     and calculate the parameters of the updating formula. */

    tempa = zmat[*knew + zmat_dim1];
    if (*idz >= 2) {
      tempa = -tempa;
    }
    if (jl > 1) {
      tempb = zmat[*knew + jl * zmat_dim1];
    }
    i__1 = *npt;
    for (i__ = 1; i__ <= i__1; ++i__) {
      w[i__] = tempa * zmat[i__ + zmat_dim1];
      if (jl > 1) {
        w[i__] += tempb * zmat[i__ + jl * zmat_dim1];
      }
      /* L30: */
    }
    alpha = w[*knew];
    tau = vlag[*knew];
    tausq = tau * tau;
    denom = alpha * *beta + tausq;
    vlag[*knew] -= one;

    /*     Complete the updating of ZMAT when there is only one nonzero element
     */
    /*     in the KNEW-th row of the new matrix ZMAT, but, if IFLAG is set to
     * one, */
    /*     then the first column of ZMAT will be exchanged with another one
     * later. */

    iflag = 0;
    if (jl == 1) {
      temp = sqrt((fabs(denom)));
      tempb = tempa / temp;
      tempa = tau / temp;
      i__1 = *npt;
      for (i__ = 1; i__ <= i__1; ++i__) {
        /* L40: */
        zmat[i__ + zmat_dim1] =
            tempa * zmat[i__ + zmat_dim1] - tempb * vlag[i__];
      }
      if (*idz == 1 && temp < zero) {
        *idz = 2;
      }
      if (*idz >= 2 && temp >= zero) {
        iflag = 1;
      }
    } else {
      /*     Complete the updating of ZMAT in the alternative case. */

      ja = 1;
      if (*beta >= zero) {
        ja = jl;
      }
      jb = jl + 1 - ja;
      temp = zmat[*knew + jb * zmat_dim1] / denom;
      tempa = temp * *beta;
      tempb = temp * tau;
      temp = zmat[*knew + ja * zmat_dim1];
      scala = one / sqrt(fabs(*beta) * temp * temp + tausq);
      scalb = scala * sqrt((fabs(denom)));
      i__1 = *npt;
      for (i__ = 1; i__ <= i__1; ++i__) {
        zmat[i__ + ja * zmat_dim1] =
            scala * (tau * zmat[i__ + ja * zmat_dim1] - temp * vlag[i__]);
        /* L50: */
        zmat[i__ + jb * zmat_dim1] =
            scalb *
            (zmat[i__ + jb * zmat_dim1] - tempa * w[i__] - tempb * vlag[i__]);
      }
      if (denom <= zero) {
        if (*beta < zero) {
          ++(*idz);
        }
        if (*beta >= zero) {
          iflag = 1;
        }
      }
    }

    /*     IDZ is reduced in the following case, and usually the first column */
    /*     of ZMAT is exchanged with a later one. */

    if (iflag == 1) {
      --(*idz);
      i__1 = *npt;
      for (i__ = 1; i__ <= i__1; ++i__) {
        temp = zmat[i__ + zmat_dim1];
        zmat[i__ + zmat_dim1] = zmat[i__ + *idz * zmat_dim1];
        /* L60: */
        zmat[i__ + *idz * zmat_dim1] = temp;
      }
    }

    /*     Finally, update the matrix BMAT. */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
      jp = *npt + j;
      w[jp] = bmat[*knew + j * bmat_dim1];
      tempa = (alpha * vlag[jp] - tau * w[jp]) / denom;
      tempb = (-(*beta) * w[jp] - tau * vlag[jp]) / denom;
      i__2 = jp;
      for (i__ = 1; i__ <= i__2; ++i__) {
        bmat[i__ + j * bmat_dim1] =
            bmat[i__ + j * bmat_dim1] + tempa * vlag[i__] + tempb * w[i__];
        if (i__ > *npt) {
          bmat[jp + (i__ - *npt) * bmat_dim1] = bmat[i__ + j * bmat_dim1];
        }
        /* L70: */
      }
    }
    return 0;
  } /* update_ */

}  // namespace PowellNewUOAImpl
