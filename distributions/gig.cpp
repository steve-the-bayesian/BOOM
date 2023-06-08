/*
  Copyright (C) 2005-2020 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

// The code in this file was modified from the source code from the R package
// GIGrvg by Josef Leydold and Wolfgang Hormann.  It contains code implementing
// the algorithm from their paper "Generating Generalized Inverse Gaussian
// Random Variates" which appeared in Statistics and Computing (2014) volume 24,
// pages 547–557.  The modifications are consistent with the (GPL>=2) license
// governing the GIGrvg package.
//
// The code was modified by Steven L. Scott to (a) return a single draw, (b)
// make it independent of R, (c) better adhere to code style in the rest of
// BOOM, and (d), make the order of the arguments agree with the author's paper
// and the wikipedia entry on the GIG distribution.

#include <cmath>
#include "distributions.hpp"  // for rgamma
#include "distributions/rng.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/math_utils.hpp"

namespace {
  using BOOM::RNG;
  double _gig_mode(double lambda, double omega);
  double _rgig_ROU_noshift_mt(RNG &rng, double lambda, double lambda_old,
                              double omega, double alpha);
  double _rgig_newapproach1_mt(RNG &rng, double lambda, double lambda_old,
                               double omega, double alpha);
  double _rgig_ROU_shift_alt_mt(RNG &rng, double lambda, double lambda_old,
                                double omega, double alpha);

  double _unur_bessel_k_nuasympt (double x, double nu, int islog,
                                  int expon_scaled);
}  // namespace

namespace BOOM {
  // Evaluate the density of the GIG distribution.
  //  Args:
  //    x:  argument of the density
  //    lambda:  parameter (lambda - 1) appearing in the exponent of x.
  //    psi: coefficent of x in th exponential.
  //    chi: coefficent of 1/x in the exponential.
  //    logscale:  If true return the log density.  If false, return the density.
  double dgig(double x, double lambda, double psi, double chi, bool logscale) {
    /* check GIG parameters: */
    /* we handle invalid input as in the R core function dnorm() */
    if (! (std::isfinite(lambda) && std::isfinite(chi) && std::isfinite(psi)) ) {
      return std::numeric_limits<double>::quiet_NaN();
    } else {
      /* check GIG parameters */
      if ( (chi <  0. || psi < 0)      ||
           (chi == 0. && lambda <= 0.) ||
           (psi == 0. && lambda >= 0.) ) {
        return std::numeric_limits<double>::quiet_NaN();
      }
    }

    double LOGNORMCONSTANT;
    /* compute normalization constant */
    if (psi == 0.) {
      /* case: Inverse Gamma distribution */
      LOGNORMCONSTANT = -lambda * log(0.5 * chi) - std::lgamma(-lambda);
    } else if (chi == 0.) {
      /* case: Gamma distribution */
      LOGNORMCONSTANT = lambda * log(0.5 * psi) - std::lgamma(lambda);
    } else {      /* general case: */
      /*   pow(psi/chi, lambda/2.) / (2. * bessel_k(sqrt(psi*chi),lambda,1)); */
      double alambda = fabs(lambda);
      double beta = sqrt(psi * chi);
      LOGNORMCONSTANT = 0.5*lambda*log(psi / chi) - M_LN2;
      if (alambda < 50.) {
        /* threshold value 50 is selected by experiments */
        LOGNORMCONSTANT -= log(bessel_k(beta, alambda, 2)) - beta;
      }
      else {
        LOGNORMCONSTANT -= _unur_bessel_k_nuasympt(beta, alambda, true, false);
      }
    }

    /* evaluate density */
    if (std::isnan(x)) {
      return x;
    } else if (!std::isfinite(x) || x <= 0.) {
      return (logscale) ? BOOM::negative_infinity() : 0.;
    } else {
      double res = LOGNORMCONSTANT + ((lambda-1.)*log(x) - 0.5*(chi / x + psi * x));
      return logscale ? res : exp(res);
    }
    return BOOM::negative_infinity();
  }

  // The mean of the GIG distribution.
  double gig_mean(double lambda, double psi, double chi) {
    double arg = sqrt(psi * chi);
    // TODO: swap in the standard library function once it is widely available (c++17).
    // double numerator = sqrt(chi) * std::cyl_bessel_k(lambda + 1, arg);
    // double denominator = sqrt(psi) * std::cyl_bessel_k(lambda, arg);
    double numerator = sqrt(chi) * bessel_k(arg, lambda + 1, 1);
    double denominator = sqrt(psi) * bessel_k(arg, lambda, 1);
    return numerator / denominator;
  }

  // Simulate a random draw from the GIG distribution.
  //
  //
  double rgig_mt(RNG &rng, double lambda, double psi, double chi) {

    double omega, alpha;     /* parameters of standard distribution */

    /* check GIG parameters: */
    if ( !(std::isfinite(lambda) && std::isfinite(chi) && std::isfinite(psi)) ||
         (chi <  0. || psi < 0)      ||
         (chi == 0. && lambda <= 0.) ||
         (psi == 0. && lambda >= 0.) ) {
      std::ostringstream err;
      err << "invalid parameters for GIG distribution: lambda = " << lambda
          << ", chi = " << chi
          << ", psi = " << psi;
      report_error(err.str());
    }

    constexpr double ZTOL = std::numeric_limits<double>::epsilon() * 10;

    if (chi < ZTOL) {
      /* special cases which are basically Gamma and Inverse Gamma distribution */
      if (lambda > 0.0) {
        return rgamma(lambda, 2.0/psi);
      }
      else {
        return 1.0/rgamma(-lambda, 2.0/psi);
      }
    } else if (psi < ZTOL) {
      /* special cases which are basically Gamma and Inverse Gamma distribution */
      if (lambda > 0.0) {
        return 1.0 / rgamma_mt(rng, lambda, 2.0 / chi);
      } else {
        return rgamma_mt(rng, -lambda, 2.0 / chi);
      }

    } else {
      double lambda_old = lambda;
      if (lambda < 0.) {
        lambda = -lambda;
      }
      alpha = sqrt(chi / psi);
      omega = sqrt(psi * chi);

      if (lambda > 2. || omega > 3.) {
        /* Ratio-of-uniforms with shift by 'mode', alternative implementation */
        return _rgig_ROU_shift_alt_mt(rng, lambda, lambda_old, omega, alpha);
      } else if (lambda >= 1. -2.25 * omega * omega || omega > 0.2) {
        /* Ratio-of-uniforms without shift */
        return _rgig_ROU_noshift_mt(rng, lambda, lambda_old, omega, alpha);
      } else if (lambda >= 0. && omega > 0.) {
        /* New approach, constant hat in log-concave part. */
        return _rgig_newapproach1_mt(rng, lambda, lambda_old, omega, alpha);
      } else {
        std::ostringstream err;
        err << "parameters must satisfy lambda >= 0 and omega > 0.\n"
            << "lambda = " << lambda << "\n"
            << "omega = " << omega;
        report_error(err.str());
      }
      return BOOM::negative_infinity();
    }
  }   /* end of rgig */

}  // namespace


//===========================================================================
namespace {
  // Implementation functions methods.
  using BOOM::RNG;
  //---------------------------------------------------------------------------
  // Return the mode of the GIG(lambda, omega) distribution.
  double _gig_mode(double lambda, double omega) {
    if (lambda >= 1.)
      /* mode of fgig(x) */
      return (sqrt((lambda -1.0) * (lambda - 1.0) + omega * omega)
              + (lambda - 1.0)) / omega;
    else
      /* 0 <= lambda < 1: use mode of f(1/x) */
      return omega / (sqrt((1.0 - lambda) * (1.0 - lambda) + omega * omega)
                      + (1.0 - lambda));
  }

  //---------------------------------------------------------------------------
  // Simulate a GIG random variable from the ratio-of-uniforms method without shift.
  //   Dagpunar (1988), Sect.~4.6.2
  //   Lehner (1989)
  double _rgig_ROU_noshift_mt(RNG &rng, double lambda, double lambda_old,
                            double omega, double alpha) {

    double xm, nc;     /* location of mode; c=log(f(xm)) normalization constant */
    double ym, um;     /* location of maximum of x*sqrt(f(x)); umax of MBR */
    double s, t;       /* auxiliary variables */
    double V, X;    /* random variables */

    /* -- Setup -------------------------------------------------------------- */

    /* shortcuts */
    t = 0.5 * (lambda - 1.0);
    s = 0.25 * omega;

    /* mode = location of maximum of sqrt(f(x)) */
    xm = _gig_mode(lambda, omega);

    /* normalization constant: c = log(sqrt(f(xm))) */
    nc = t * log(xm) - s * (xm + 1.0 / xm);

    /* location of maximum of x*sqrt(f(x)):           */
    /* we need the positive root of                   */
    /*    omega/2*y^2 - (lambda+1)*y - omega/2 = 0    */
    ym = ((lambda+1.) + sqrt((lambda+1.)*(lambda+1.) + omega*omega))/omega;

    /* boundaries of minmal bounding rectangle:                   */
    /* we us the "normalized" density f(x) / f(xm). hence         */
    /* upper boundary: vmax = 1.                                  */
    /* left hand boundary: umin = 0.                              */
    /* right hand boundary: umax = ym * sqrt(f(ym)) / sqrt(f(xm)) */
    um = exp(0.5*(lambda+1.)*log(ym) - s*(ym + 1./ym) - nc);

    /* -- Generate sample ---------------------------------------------------- */

    do {
      double U = um * rng();              /* U(0,umax) */
      V = rng();                   /* U(0,vmax) */
      X = U/V;
    }                              /* Acceptance/Rejection */
    while (((log(V)) > (t*log(X) - s*(X + 1./X) - nc)));

    return (lambda_old < 0.) ? (alpha / X) : (alpha * X);
  }


  /*---------------------------------------------------------------------------*/

  // Type 4:
  // New approach, constant hat in log-concave part.
  // Draw sample from GIG distribution.
  //
  // Case: 0 < lambda < 1, 0 < omega < 1
  //
  // Parameters:
  //   lambda .. parameter for distribution
  //   omega ... parameter for distribution
  //
  // Return:
  //   random draw
  double _rgig_newapproach1_mt (RNG &rng, double lambda, double lambda_old,
                              double omega, double alpha) {
    /* parameters for hat function */
    double A[3], Atot;  /* area below hat */
    double k0;          /* maximum of PDF */
    double k1, k2;      /* multiplicative constant */

    double xm;          /* location of mode */
    double x0;          /* splitting point T-concave / T-convex */
    double a;           /* auxiliary variable */

    double X;     /* random numbers */
    double hx;          /* hat at X */

    /* -- Check arguments ---------------------------------------------------- */

    if (lambda >= 1. || omega >1.) {
      std::ostringstream err;
      err << "invalid parameters passed to the 'type 4' implementation of rgig:\n"
          << "Need lambda < 1:   " << lambda << "\n"
          << "Need omegea <= 1:  " << omega;
      BOOM::report_error(err.str());
    }

    /* -- Setup -------------------------------------------------------------- */

    /* mode = location of maximum of sqrt(f(x)) */
    xm = _gig_mode(lambda, omega);

    /* splitting point */
    x0 = omega/(1.-lambda);

    /* domain [0, x_0] */
    k0 = exp((lambda-1.)*log(xm) - 0.5*omega*(xm + 1./xm));     /* = f(xm) */
    A[0] = k0 * x0;

    /* domain [x_0, Infinity] */
    if (x0 >= 2.0 / omega) {
      k1 = 0.0;
      A[1] = 0.0;
      k2 = pow(x0, lambda - 1.0);
      A[2] = k2 * 2. * exp(-omega * x0 /2.0) / omega;
    } else {
      /* domain [x_0, 2/omega] */
      k1 = exp(-omega);
      A[1] = (lambda == 0.0)
          ? k1 * log(2.0 / (omega * omega))
          : k1 / lambda * (pow(2.0 / omega, lambda) - pow(x0, lambda));

      /* domain [2/omega, Infinity] */
      k2 = pow(2/omega, lambda-1.);
      A[2] = k2 * 2 * exp(-1.)/omega;
    }

    /* total area */
    Atot = A[0] + A[1] + A[2];

    /* -- Generate sample ---------------------------------------------------- */
    do {

      /* get uniform random number */
      double V = Atot * rng();

      do {

        /* domain [0, x_0] */
        if (V <= A[0]) {
          X = x0 * V / A[0];
          hx = k0;
          break;
        }

        /* domain [x_0, 2/omega] */
        V -= A[0];
        if (V <= A[1]) {
          if (lambda == 0.) {
            X = omega * exp(exp(omega)*V);
            hx = k1 / X;
          }
          else {
            X = pow(pow(x0, lambda) + (lambda / k1 * V), 1./lambda);
            hx = k1 * pow(X, lambda - 1.0);
          }
          break;
        }

        /* domain [max(x0,2/omega), Infinity] */
        V -= A[1];
        a = (x0 > 2./omega) ? x0 : 2./omega;
        X = -2.0 / omega * log(exp(-omega / 2.0 * a) - omega/(2.0 * k2) * V);
        hx = k2 * exp(-omega / 2.0 * X);
        break;

      } while(0);

      /* accept or reject */
      double U = rng() * hx;

      if (log(U) <= (lambda - 1.0) * log(X) - omega / 2.0 * (X + 1.0 / X)) {
        /* store random point */
        return (lambda_old < 0.) ? (alpha / X) : (alpha * X);
      }
    } while(1);
    return BOOM::negative_infinity();
  } /* end of _rgig_newapproach1() */

  //---------------------------------------------------------------------------
  // Type 8:
  // Ratio-of-uniforms with shift by 'mode', alternative implementation.
  //   Dagpunar (1989)
  //   Lehner (1989)
  //---------------------------------------------------------------------------
  double _rgig_ROU_shift_alt_mt(RNG &rng, double lambda, double lambda_old,
                              double omega, double alpha) {
    double s, t;       /* auxiliary variables */
    double V, X;    /* random variables */

    double p, q;       /* coefficents of depressed cubic */
    double fi, fak;    /* auxiliary results for Cardano's rule */

    double y1, y2;     /* roots of (1/x)*sqrt(f((1/x)+m)) */

    double uplus, uminus;  /* maximum and minimum of x*sqrt(f(x+m)) */

    /* -- Setup -------------------------------------------------------------- */

    /* shortcuts */
    t = 0.5 * (lambda-1.);
    s = 0.25 * omega;

    /* mode = location of maximum of sqrt(f(x)) */
    double xm = _gig_mode(lambda, omega);

    /* normalization constant: c = log(sqrt(f(xm))) */
    double nc = t * log(xm) - s*(xm + 1.0 / xm);

    /* location of minimum and maximum of (1/x)*sqrt(f(1/x+m)):  */

    /* compute coeffients of cubic equation y^3+a*y^2+b*y+c=0 */
    double a = -(2.0 * (lambda + 1.0) / omega + xm);       /* < 0 */
    double b = (2.0 * (lambda - 1.0) * xm/ omega - 1.0);
    double c = xm;

    /* we need the roots in (0,xm) and (xm,inf) */

    /* substitute y=z-a/3 for depressed cubic equation z^3+p*z+q=0 */
    p = b - a * a / 3.0;
    q = (2.0 * a * a * a) / 27.0 - (a * b) / 3.0 + c;

    /* use Cardano's rule */
    fi = acos(-q/(2.*sqrt(-(p*p*p)/27.)));
    fak = 2.*sqrt(-p/3.);
    y1 = fak * cos(fi/3.) - a/3.;
    y2 = fak * cos(fi/3. + 4./3.*M_PI) - a/3.;

    /* boundaries of minmal bounding rectangle:                  */
    /* we us the "normalized" density f(x) / f(xm). hence        */
    /* upper boundary: vmax = 1.                                 */
    /* left hand boundary: uminus = (y2-xm) * sqrt(f(y2)) / sqrt(f(xm)) */
    /* right hand boundary: uplus = (y1-xm) * sqrt(f(y1)) / sqrt(f(xm)) */
    uplus  = (y1-xm) * exp(t*log(y1) - s*(y1 + 1./y1) - nc);
    uminus = (y2-xm) * exp(t*log(y2) - s*(y2 + 1./y2) - nc);

    /* -- Generate sample ---------------------------------------------------- */

    do {
      double U = uminus + rng() * (uplus - uminus);    /* U(u-,u+)  */
      V = rng();                                /* U(0,vmax) */
      X = U/V + xm;
    }                                         /* Acceptance/Rejection */
    while ((X <= 0.) || ((log(V)) > (t*log(X) - s*(X + 1./X) - nc)));

    return (lambda_old < 0.) ? (alpha / X) : (alpha * X);

  } /* end of _rgig_ROU_shift_alt() */


  //---------------------------------------------------------------------------*/
  // Asymptotic expansion of Bessel K_nu(x) function                           */
  // when BOTH  nu and x  are large.                                           */
  //                                                                           */
  // parameters:                                                               */
  //   x            ... argument for K_nu()                                    */
  //   nu           ... order or Bessel function                               */
  //   islog        ... return logarithm of result TRUE and result when FALSE  */
  //   expon_scaled ... return exp(-x)*K_nu(x) when TRUE and K_nu(x) when FALSE*/
  //                                                                           */
  //---------------------------------------------------------------------------*/
  //                                                                           */
  // references:                                                               */
  // ##  Abramowitz & Stegun , p.378, __ 9.7.8. __                             */
  //                                                                           */
  // ## K_nu(nu * z) ~ sqrt(pi/(2*nu)) * exp(-nu*eta)/(1+z^2)^(1/4)            */
  // ##                   * {1 - u_1(t)/nu + u_2(t)/nu^2 - ... }               */
  //                                                                           */
  // ## where   t = 1 / sqrt(1 + z^2),                                         */
  // ##       eta = sqrt(1 + z^2) + log(z / (1 + sqrt(1+z^2)))                 */
  // ##                                                                        */
  // ## and u_k(t)  from  p.366  __ 9.3.9 __                                   */
  //                                                                           */
  // ## u0(t) = 1                                                              */
  // ## u1(t) = (3*t - 5*t^3)/24                                               */
  // ## u2(t) = (81*t^2 - 462*t^4 + 385*t^6)/1152                              */
  // ## ...                                                                    */
  //                                                                           */
  // ## with recursion  9.3.10    for  k = 0, 1, .... :                        */
  // ##                                                                        */
  // ## u_{k+1}(t) = t^2/2 * (1 - t^2) * u'_k(t) +                             */
  // ##            1/8  \int_0^t (1 - 5*s^2)* u_k(s) ds                        */
  //---------------------------------------------------------------------------*/
  //                                                                           */
  // Original implementation in R code (R package "Bessel" v. 0.5-3) by        */
  //   Martin Maechler, Date: 23 Nov 2009, 13:39                               */
  //                                                                           */
  // Translated into C code by Kemal Dingic, Oct. 2011.                        */
  //                                                                           */
  // Modified by Josef Leydold on Tue Nov  1 13:22:09 CET 2011                 */
  double _unur_bessel_k_nuasympt (double x, double nu, int islog,
                                  int expon_scaled) {
    constexpr double M_LNPI = 1.14472988584940017414342735135;      /* ln(pi) */

    double z;                   /* rescaled argument for K_nu() */
    double sz, t, t2, eta;      /* auxiliary variables */
    double d, u1t,u2t,u3t,u4t;  /* (auxiliary) results for Debye polynomials */
    double res;                 /* value of log(K_nu(x)) [= result] */

    /* rescale: we comute K_nu(z * nu) */
    z = x / nu;

    /* auxiliary variables */
    sz = hypot(1,z);   /* = sqrt(1+z^2) */
    t = 1. / sz;
    t2 = t*t;

    eta = (expon_scaled) ? (1./(z + sz)) : sz;
    eta += log(z) - log1p(sz);                  /* = log(z/(1+sz)) */

    /* evaluate Debye polynomials u_j(t) */
    u1t = (t * (3. - 5.*t2))/24.;
    u2t = t2 * (81. + t2*(-462. + t2 * 385.))/1152.;
    u3t = t*t2 * (30375. + t2 * (-369603. + t2 * (765765. - t2 * 425425.)))/414720.;
    u4t = t2*t2 * (4465125.
                   + t2 * (-94121676.
                           + t2 * (349922430.
                                   + t2 * (-446185740.
                                           + t2 * 185910725.)))) / 39813120.;
    d = (-u1t + (u2t + (-u3t + u4t/nu)/nu)/nu)/nu;

    /* log(K_nu(x)) */
    res = log1p(d) - nu*eta - 0.5*(log(2.*nu*sz) - M_LNPI);

    return (islog ? res : exp(res));
  }

}  // namespace
