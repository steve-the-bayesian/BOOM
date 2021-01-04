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

#ifndef BOOM_RMATH_DISTRIBUTIONS_HPP
#define BOOM_RMATH_DISTRIBUTIONS_HPP

// This file protects the user from including Rmath.h, which sets a
// bunch of dangerous #define's (e.g. trunc, which is used by C++ as a
// file input mode.)

#include <vector>
#include "distributions/rng.hpp"

namespace BOOM {
  // This file brings Rmath functions into BOOM scope and adjusts some
  // default arguments

  /*---------- in rtriangle.c -------------------*/
  double rtrap(double, double, double, double);
  double rtriangle(double, double, double);
  double rtrap_mt(RNG &, double, double, double, double);
  double rtriangle_mt(RNG &, double, double, double);

  double dtriangle(double, double, double, double, bool log = false);
  double ptriangle(double, double, double, double, bool);
  double qtriangle(double, double, double, double);

  //============ stuff below wraps Rmath  functions =======

  double dnorm(double x, double mu = 0, double sig = 1, bool log = false);
  double pnorm(double x, double mu = 0, double sig = 1, bool low = true,
               bool log = false);
  double qnorm(double p, double mu = 0, double sig = 1, bool low = true,
               bool log = false);
  double rnorm(double mu = 0, double sig = 1);
  double rnorm_mt(RNG &, double mu = 0, double sig = 1);

  /* Uniform Distribution */
  double dunif(double x, double lo = 0, double hi = 1, bool log = false);
  double punif(double x, double lo = 0, double hi = 1, bool low = true,
               bool log = false);
  double qunif(double p, double lo = 0, double hi = 1, bool low = true,
               bool log = false);
  double runif(double lo = 0, double hi = 1);
  double runif_mt(RNG &, double lo = 0, double hi = 1);

  /* Gamma Distribution */

  double dgamma(double x, double a = 1, double b = 1, bool log = false);
  double pgamma(double x, double a = 1, double b = 1, bool low = true,
                bool log = false);
  double qgamma(double p, double a = 1, double b = 1, bool low = true,
                bool log = false);
  double rgamma(double a = 1, double b = 1);
  double rgamma_mt(RNG &, double a = 1, double b = 1);

  /* Beta Distribution */

  double dbeta(double x, double a = 1, double b = 1, bool log = false);
  double pbeta(double x, double a = 1, double b = 1, bool low = true,
               bool log = false);
  double qbeta(double p, double a = 1, double b = 1, bool low = true,
               bool log = false);
  double rbeta(double a = 1, double b = 1);
  double rbeta_mt(RNG &, double a = 1, double b = 1);

  /* Lognormal Distribution */

  double dlnorm(double x, double mu = 0, double sig = 1, bool log = false);
  double plnorm(double x, double mu = 0, double sig = 1, bool low = true,
                bool log = false);
  double qlnorm(double p, double mu = 0, double sig = 1, bool low = true,
                bool log = false);
  double rlnorm(double mu = 0, double sig = 1);
  double rlnorm_mt(RNG &, double mu = 0, double sig = 1);

  /* Chi-squared Distribution */

  double dchisq(double x, double df, bool log = false);
  double pchisq(double x, double df, bool low = true, bool log = false);
  double qchisq(double q, double df, bool low = true, bool log = false);
  double rchisq(double df);
  double rchisq_mt(RNG &, double df);

  /* Non-central Chi-squared Distribution */

  double dnchisq(double x, double df, double ncp, bool log = false);
  double pnchisq(double x, double df, double ncp, bool low = true,
                 bool log = false);
  double qnchisq(double p, double df, double ncp, bool low = true,
                 bool log = false);
  //  double rnchisq(double df, double ncp);

  /* F Distibution */

  double df(double x, double dfn, double dfd, bool log = false);
  double pf(double x, double dfn, double dfd, bool low = true,
            bool log = false);
  double qf(double p, double dfn, double dfd, bool low = true,
            bool log = false);
  double rf(double dfn, double dfd);
  double rf_mt(RNG &, double dfn, double dfd);

  /* Student t Distibution */

  double dt(double x, double df, bool log = false);
  double pt(double x, double df, bool low = true, bool log = false);
  double qt(double p, double df, bool low = true, bool log = false);
  double rt(double df);
  double rt_mt(RNG &, double df);

  /* Binomial Distribution */

  double dbinom(double x, double n, double p, bool log = false);
  double pbinom(double x, double n, double p, bool low = true,
                bool log = false);
  double qbinom(double p, double n, double prob, bool low = true,
                bool log = false);
  unsigned rbinom(int n, double prob);
  unsigned rbinom_mt(RNG &, int n, double prob);

  /* Multinomial distribution */
  void rmultinom_mt(RNG &rng, int64_t n, const std::vector<double> &prob,
                    std::vector<int> &result);
  std::vector<int> rmultinom_mt(RNG &rng, int64_t n,
                                const std::vector<double> &prob);
  void rmultinom(int64_t n, const std::vector<double> &prob,
                 std::vector<int> &result);
  std::vector<int> rmultinom(int64_t n, const std::vector<double> &prob);

  /* Cauchy Distribution */

  double dcauchy(double x, double mu = 0, double sig = 1, bool log = false);
  double pcauchy(double x, double mu = 0, double sig = 1, bool low = true,
                 bool log = false);
  double qcauchy(double p, double mu = 0, double sig = 1, bool low = true,
                 bool log = false);
  double rcauchy(double mu, double sig);
  double rcauchy_mt(RNG &, double mu, double sig);

  /* Exponential Distribution */

  double dexp(double x, double lam = 1, bool log = false);
  double pexp(double x, double lam = 1, bool low = true, bool log = false);
  double qexp(double p, double lam = 1, bool low = true, bool log = false);
  double rexp(double lam = 1);
  double rexp_mt(RNG &, double lam = 1);

  /* Geometric Distribution */

  double dgeom(double x, double p, bool log = false);
  double pgeom(double x, double p, bool low = true, bool log = false);
  double qgeom(double p, double prob, bool low = true, bool log = false);
  double rgeom(double p);
  double rgeom_mt(RNG &, double p);

  /* Hypergeometric Distibution */

  double dhyper(double x, double Ntrue, double Nfalse, double n,
                bool log = false);
  double phyper(double x, double Ntrue, double Nfalse, double n,
                bool low = true, bool log = false);
  double qhyper(double p, double Ntrue, double Nfalse, double n,
                bool low = true, bool log = false);
  double rhyper(double Ntrue, double Nfalse, double n);
  double rhyper_mt(RNG &, double Ntrue, double Nfalse, double n);

  /* Negative Binomial Distribution */
  /*
   *   For integer n the neg.binom distribution returns the probability
   *   of x failures before the nth success in a sequence of Bernoulli
   *   trials.  Also.. y~Poisson(lamba) lambda~Gamma(a,b) => y ~ NB(n =
   *   alpha, p = beta/(1+beta))
   */

  double dnbinom(double x, double n = 1, double p = .5, bool log = false);
  double pnbinom(double x, double n = 1, double p = .5, bool low = true,
                 bool log = false);
  double qnbinom(double p, double n = 1, double prob = .5, bool low = true,
                 bool log = false);
  double rnbinom(double n = 1, double p = .5);
  double rnbinom_mt(RNG &, double n = 1, double p = .5);

  /* Poisson Distribution */

  double dpois(double x, double lam = 1, bool log = false);
  double ppois(double x, double lam = 1, bool low = true, bool log = false);
  double qpois(double p, double lam = 1, bool low = true, bool log = false);
  double rpois(double lam = 1);
  double rpois_mt(RNG &, double lam = 1);

  /* Weibull Distribution */

  double dweibull(double x, double shape = 1, double scale = 1,
                  bool log = false);
  double pweibull(double x, double shape = 1, double scale = 1, bool low = true,
                  bool log = false);
  double qweibull(double p, double shape = 1, double scale = 1, bool low = true,
                  bool log = false);
  double rweibull(double shape = 1, double scale = 1);
  double rweibull_mt(RNG &, double shape = 1, double scale = 1);

  /* Logistic Distribution */

  double dlogis(double x, double mu = 0, double sig = 1, bool log = false);
  double plogis(double x, double mu = 0, double sig = 1, bool low = true,
                bool log = false);
  double qlogis(double p, double mu = 0, double sig = 1, bool low = true,
                bool log = false);
  double rlogis(double mu = 0, double sig = 1);
  double rlogis_mt(RNG &, double mu = 0, double sig = 1);

  /* Non-central Beta Distribution */

  double dnbeta(double x, double a = 1, double b = 1, double lam = 0,
                bool log = false);
  double pnbeta(double x, double a = 1, double b = 1, double lam = 0,
                bool low = true, bool log = false);

  /* Non-central F Distribution */

  double pnf(double x, double dfn, double dfd, double lam, bool low = true,
             bool log = false);

  /* Non-central Student t Distribution */

  double pnt(double x, double df, double mu = 0, bool low = true,
             bool log = false);

  /* Gamma and Related Functions */
  double gamma(double);
  double lgamma(double);
  double digamma(double);
  double trigamma(double);
  double tetragamma(double);
  double pentagamma(double);

  double beta(double a, double b);
  double lbeta(double a, double b);

  double choose(double n, double p);
  double lchoose(double n, double p);

  /* Bessel Functions */
  // Bessesl functions have been removed because they are included in
  // the new C++ standard

  //   double bessel_i(double x, double a, double expo);
  //   double bessel_j(double x, double a);
  double bessel_k(double x, double a, double expo);
  //   double bessel_y(double x, double a);

  /* General Support Functions */

  double pythag(double, double);
  double sign(double);
  double fprec(double, double);
  double fround(double, double);
  double fsign(double, double);
  double ftrunc(double);

  struct unknown_error {};
}  // namespace BOOM

#endif  // BOOM_RMATH_DISTRIBUTIONS_HPP
