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

/*
 *  Mathlib : A C Library of Special Functions
 *  Copyright (C) 1998-2001  The R Development Core Team
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation; either version 2.1 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *

 * Mathlib.h  should contain ALL headers from R's C code in `src/nmath'
   ---------  such that ``the Math library'' can be used by simply

   ``#include "Bmath.hpp" ''

   and nothing else.
*/
#ifndef BOOM_BMATH_HPP
#define BOOM_BMATH_HPP

#include <cmath>
#include "distributions/rng.hpp"
#include <vector>
#include "cpputil/report_error.hpp"

/*-- Mathlib as part of R --  define this for standalone : */
/* #undef MATHLIB_STANDALONE */

        /* Undo SGI Madness */

#ifdef ftrunc
# undef ftrunc
#endif
#ifdef qexp
# undef qexp
#endif
#ifdef qgamma
# undef qgamma
#endif


/* ----- The following constants and entry points are part of the R API ---- */

/* 30 Decimal-place constants */
/* Computed with bc -l (scale=32; proper round) */

/* SVID & X/Open Constants */
/* Names from Solaris math.h */

#ifndef M_E
#define M_E             2.718281828459045235360287471353        /* e */
#endif

#ifndef M_LOG2E
#define M_LOG2E         1.442695040888963407359924681002        /* log2(e) */
#endif

#ifndef M_LOG10E
#define M_LOG10E        0.434294481903251827651128918917        /* log10(e) */
#endif

#ifndef M_LN2
#define M_LN2           0.693147180559945309417232121458        /* ln(2) */
#endif

#ifndef M_LN10
#define M_LN10          2.302585092994045684017991454684        /* ln(10) */
#endif

#ifndef M_PI
#define M_PI            3.141592653589793238462643383280        /* pi */
#endif

#ifndef M_2PI
#define M_2PI           6.283185307179586476925286766559        /* 2*pi */
#endif

#ifndef M_PI_2
#define M_PI_2          1.570796326794896619231321691640        /* pi/2 */
#endif

#ifndef M_PI_4
#define M_PI_4          0.785398163397448309615660845820        /* pi/4 */
#endif

#ifndef M_1_PI
#define M_1_PI          0.318309886183790671537767526745        /* 1/pi */
#endif

#ifndef M_2_PI
#define M_2_PI          0.636619772367581343075535053490        /* 2/pi */
#endif

#ifndef M_2_SQRTPI
#define M_2_SQRTPI      1.128379167095512573896158903122        /* 2/sqrt(pi) */
#endif

#ifndef M_SQRT2
#define M_SQRT2         1.414213562373095048801688724210        /* sqrt(2) */
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2       0.707106781186547524400844362105        /* 1/sqrt(2) */
#endif

/* R-Specific Constants */

#ifndef M_SQRT_3
#define M_SQRT_3        1.732050807568877293527446341506        /* sqrt(3) */
#endif

#ifndef M_SQRT_32
#define M_SQRT_32       5.656854249492380195206754896838        /* sqrt(32) */
#endif

#ifndef M_LOG10_2
#define M_LOG10_2       0.301029995663981195213738894724        /* log10(2) */
#endif

#ifndef M_SQRT_PI
#define M_SQRT_PI       1.772453850905516027298167483341        /* sqrt(pi) */
#endif

#ifndef M_1_SQRT_2PI
#define M_1_SQRT_2PI    0.398942280401432677939946059934        /* 1/sqrt(2pi) */
#endif

#ifndef M_SQRT_2dPI
#define M_SQRT_2dPI     0.797884560802865355879892119869        /* sqrt(2/pi) */
#endif


#ifndef M_LN_SQRT_PI
#define M_LN_SQRT_PI    0.572364942924700087071713675677        /* log(sqrt(pi)) */
#endif

#ifndef M_LN_SQRT_2PI
#define M_LN_SQRT_2PI   0.918938533204672741780329736406        /* log(sqrt(2*pi)) */
#endif

#ifndef M_LN_SQRT_PId2
#define M_LN_SQRT_PId2  0.225791352644727432363097614947        /* log(sqrt(pi/2)) */
#endif

#define rround  fround
#define prec    fprec

        /* R's versions with !R_FINITE checks */

namespace Rmath{

  using BOOM::RNG;
  using BOOM::report_error;

  double R_log(double x);
  double R_pow(double x, double y);
  double R_pow_di(double, int);

  /* Random Number Generators */

  double        norm_rand(BOOM::RNG &);
  double        unif_rand(BOOM::RNG &);
  double        exp_rand(BOOM::RNG &);

  /* Normal Distribution */

// #define pnorm pnorm5
// #define qnorm qnorm5
// #define dnorm dnorm4

  double        dnorm(double, double, double, int);
  double        pnorm(double, double, double, int, int);
  double        qnorm(double, double, double, int, int);
  double        rnorm(double, double);
  double        rnorm_mt(RNG &, double, double);
  void  pnorm_both(double, double *, double *, int, int);/* both tails */

  /* Uniform Distribution */

  double        dunif(double, double, double, int);
  double        punif(double, double, double, int, int);
  double        qunif(double, double, double, int, int);
  double        runif(double, double);
  double        runif_mt(RNG &, double, double);

  /* Gamma Distribution */

  double        dgamma(double, double, double, int);
  double        pgamma(double, double, double, int, int);
  double        qgamma(double, double, double, int, int);
  double        rgamma(double, double);
  double        rgamma_mt(BOOM::RNG &, double, double);

  /* Beta Distribution */

  double        dbeta(double, double, double, int);
  double        pbeta(double, double, double, int, int);
  double        qbeta(double, double, double, int, int);
  double        pbeta_raw(double, double, double, int, int);
  double        rbeta(double, double);
  double        rbeta_mt(RNG &, double, double);

  /* Lognormal Distribution */

  double        dlnorm(double, double, double, int);
  double        plnorm(double, double, double, int, int);
  double        qlnorm(double, double, double, int, int);
  double        rlnorm(double, double);
  double        rlnorm_mt(RNG &, double, double);

  /* Chi-squared Distribution */

  double        dchisq(double, double, int);
  double        pchisq(double, double, int, int);
  double        qchisq(double, double, int, int);
  double        rchisq(double);
  double        rchisq_mt(BOOM::RNG &, double);

  /* Non-central Chi-squared Distribution */

  double        dnchisq(double, double, double, int);
  double        pnchisq(double, double, double, int, int);
  double        qnchisq(double, double, double, int, int);
  double        rnchisq(double, double);
  double        rnchisq_mt(RNG &, double, double);

  /* F Distibution */

  double        df(double, double, double, int);
  double        pf(double, double, double, int, int);
  double        qf(double, double, double, int, int);
  double        rf(double, double);
  double        rf_mt(RNG &, double, double);

  /* Student t Distibution */

  double        dt(double, double, int);
  double        pt(double, double, int, int);
  double        qt(double, double, int, int);
  double        rt(double);
  double        rt_mt(RNG &, double);

  /* Binomial Distribution */

  double        dbinom(double, double, double, int);
  double        pbinom(double, double, double, int, int);
  double        qbinom(double, double, double, int, int);
  unsigned int rbinom(int, double);
  unsigned int rbinom_mt(RNG &, int, double);

  void         rmultinom_mt(RNG &, int, const std::vector<double> &prob,
                            std::vector<int> &result);
  void         rmultinom(int n, const std::vector<double> &prob,
                         std::vector<int> &result);
  std::vector<int> rmultinom_mt(RNG &rng, int n,
                                const std::vector<double> &prob);
  std::vector<int> rmultinom(int n, const std::vector<double> &prob);

  /* Cauchy Distribution */

  double        dcauchy(double, double, double, int);
  double        pcauchy(double, double, double, int, int);
  double        qcauchy(double, double, double, int, int);
  double        rcauchy(double, double);
  double        rcauchy_mt(RNG &, double, double);

  /* Exponential Distribution */

  double        dexp(double, double, int);
  double        pexp(double, double, int, int);
  double        qexp(double, double, int, int);
  double        rexp(double);
  double        rexp_mt(RNG &, double);

  /* Geometric Distribution */

  double        dgeom(double, double, int);
  double        pgeom(double, double, int, int);
  double        qgeom(double, double, int, int);
  double        rgeom(double);
  double        rgeom_mt(RNG &, double);

  /* Hypergeometric Distibution */

  double        dhyper(double, double, double, double, int);
  double        phyper(double, double, double, double, int, int);
  double        qhyper(double, double, double, double, int, int);
  double        rhyper(double, double, double);
  double        rhyper_mt(RNG &, double, double, double);

  /* Negative Binomial Distribution */

  double        dnbinom(double, double, double, int);
  double        pnbinom(double, double, double, int, int);
  double        qnbinom(double, double, double, int, int);
  double        rnbinom(double, double);
  double        rnbinom_mt(RNG &, double, double);

  /* Poisson Distribution */

  double        dpois(double, double, int);
  double        ppois(double, double, int, int);
  double        qpois(double, double, int, int);
  double        rpois(double);
  double        rpois_mt(BOOM::RNG &, double);

  /* Weibull Distribution */

  double        dweibull(double, double, double, int);
  double        pweibull(double, double, double, int, int);
  double        qweibull(double, double, double, int, int);
  double        rweibull(double, double);
  double        rweibull_mt(RNG &, double, double);

  /* Logistic Distribution */

  double        dlogis(double, double, double, int);
  double        plogis(double, double, double, int, int);
  double        qlogis(double, double, double, int, int);
  double        rlogis(double, double);
  double        rlogis_mt(RNG &, double, double);

  /* Non-central Beta Distribution */

  double        dnbeta(double, double, double, double, int);
  double        pnbeta(double, double, double, double, int, int);
  double        qnbeta(double, double, double, double, int, int);

  /* Non-central F Distribution */

  double        pnf(double, double, double, double, int, int);
  double        qnf(double, double, double, double, int, int);

  /* Non-central Student t Distribution */

  double        pnt(double, double, double, int, int);
  double        qnt(double, double, double, int, int);

  /* Studentized Range Distribution */

  double        ptukey(double, double, double, double, int, int);
  double        qtukey(double, double, double, double, int, int);

  /* Wilcoxon Rank Sum Distribution */

  double dwilcox(double, double, double, int);
  double pwilcox(double, double, double, int, int);
  double qwilcox(double, double, double, int, int);
  double rwilcox(double, double);

  /* Wilcoxon Signed Rank Distribution */

  double dsignrank(double, double, int);
  double psignrank(double, double, int, int);
  double qsignrank(double, double, int, int);
  double rsignrank(double);

  /* Gamma and Related Functions */
  inline double gammafn(double x){return ::tgamma(x);}

  // lgammafn returns log(abs(Gamma(x))) the two argument version is
  // used when Gamma(x) might be negative.  The second argument
  // returns the sign of Gamma(x)
  inline double lgammafn(double x){
    // Note that std::lgamma is not guaranteed to be thread safe, but
    // on many compilers it is implemented in terms of lgamma_r, which
    // is thread safe.  http://linux.die.net/man/3/lgamma_r offers a
    // test for whether lgamma_r is present on a given system.
#if defined(_BSD_SOURCE) || defined(_SVID_SOURCE)
    int signgam = 1;
    return lgamma_r(x, &signgam);
#else
    return std::lgamma(x);
#endif
  }

  double        digamma(double);
  double        trigamma(double);
  double        tetragamma(double);
  double        pentagamma(double);

  double        beta(double, double);
  double        lbeta(double, double);

  double        choose(double, double);
  double        lchoose(double, double);
  double        bessel_k(double, double, double);

  /* General Support Functions */

  double        pythag(double, double);
  using ::expm1;
  using ::log1p;

  double        sign(double);
  double        fprec(double, double);
  double        fround(double, double);
  double        fsign(double, double);
  double        ftrunc(double);
  inline double trunc(double x){return ftrunc(x);}

/* ----------------- Private part of the header file ------------------- */

  double        d1mach(int);
  double        gamma_cody(double);
}
#ifdef MATHLIB_STANDALONE
#ifndef MATHLIB_PRIVATE_H

#define ISNAN(x)       R_IsNaNorNA(x)
#define R_FINITE(x)    R_finite(x)
int R_IsNaNorNA(double);
int R_finite(double);

#ifdef _WIN32
# define NA_REAL (*_imp__NA_REAL)
# define R_NegInf (*_imp__R_NegInf)
# define R_PosInf (*_imp__R_PosInf)
# define N01_kind (*_imp__N01_kind)
# endif

#endif /* not MATHLIB_PRIVATE_H */
#endif /* MATHLIB_STANDALONE */

#endif /* BMATH_H */
