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
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
   USA
 */

#include "distributions/Rmath_dist.hpp"
#define MATHLIB_STANDALONE
#include "Bmath/Bmath.hpp"

namespace BOOM {
#undef dnorm
  double dnorm(double x, double mu, double sig, bool log) {
    return Rmath::dnorm(x, mu, sig, log);
  }

#undef pnorm
  double pnorm(double x, double mu, double sig, bool low, bool log) {
    return Rmath::pnorm(x, mu, sig, low, log);
  }

#undef qnorm
  double qnorm(double p, double mu, double sig, bool low, bool log) {
    return Rmath::qnorm(p, mu, sig, low, log);
  }

  double rnorm(double mu, double sig) { return Rmath::rnorm(mu, sig); }
  double rnorm_mt(RNG &rng, double mu, double sig) {
    return Rmath::rnorm_mt(rng, mu, sig);
  }

  /*--- Uniform Distribution ---*/
  double dunif(double x, double lo, double hi, bool log) {
    return Rmath::dunif(x, lo, hi, log);
  }
  double punif(double x, double lo, double hi, bool low, bool log) {
    return Rmath::punif(x, lo, hi, low, log);
  }
  double qunif(double p, double lo, double hi, bool low, bool log) {
    return Rmath::qunif(p, lo, hi, low, log);
  }
  double runif(double lo, double hi) { return Rmath::runif(lo, hi); }
  double runif_mt(RNG &rng, double lo, double hi) {
    return Rmath::runif_mt(rng, lo, hi);
  }

  /*--- Gamma Distribution ---*/
  double dgamma(double x, double a, double b, bool log) {
    return Rmath::dgamma(x, a, 1.0 / b, log);
  }
  double pgamma(double x, double a, double b, bool low, bool log) {
    return Rmath::pgamma(x, a, 1.0 / b, low, log);
  }
  double qgamma(double p, double a, double b, bool low, bool log) {
    return Rmath::qgamma(p, a, 1.0 / b, low, log);
  }
  double rgamma(double a, double b) { return Rmath::rgamma(a, 1.0 / b); }
  double rgamma_mt(RNG &rng, double a, double b) {
    return Rmath::rgamma_mt(rng, a, 1.0 / b);
  }

  /* Beta Distribution */
  double dbeta(double x, double a, double b, bool log) {
    return Rmath::dbeta(x, a, b, log);
  }
  double pbeta(double x, double a, double b, bool low, bool log) {
    return Rmath::pbeta(x, a, b, low, log);
  }
  double qbeta(double p, double a, double b, bool low, bool log) {
    return Rmath::qbeta(p, a, b, low, log);
  }
  double rbeta(double a, double b) { return Rmath::rbeta(a, b); }
  double rbeta_mt(RNG &rng, double a, double b) {
    return Rmath::rbeta_mt(rng, a, b);
  }

  /* Lognormal Distribution */
  double dlnorm(double x, double mu, double sig, bool log) {
    return Rmath::dlnorm(x, mu, sig, log);
  }
  double plnorm(double x, double mu, double sig, bool low, bool log) {
    return Rmath::plnorm(x, mu, sig, low, log);
  }
  double qlnorm(double p, double mu, double sig, bool low, bool log) {
    return Rmath::qlnorm(p, mu, sig, low, log);
  }
  double rlnorm(double mu, double sig) { return Rmath::rlnorm(mu, sig); }
  double rlnorm_mt(RNG &rng, double mu, double sig) {
    return Rmath::rlnorm_mt(rng, mu, sig);
  }

  /* Chi-squared Distribution */
  double dchisq(double x, double df, bool log) {
    return Rmath::dchisq(x, df, log);
  }
  double pchisq(double x, double df, bool low, bool log) {
    return Rmath::pchisq(x, df, low, log);
  }
  double qchisq(double p, double df, bool low, bool log) {
    return Rmath::qchisq(p, df, low, log);
  }
  double rchisq(double df) { return Rmath::rchisq(df); }
  double rchisq_mt(RNG &rng, double df) { return Rmath::rchisq_mt(rng, df); }

  /* Non-central Chi-squared Distribution */
  double dnchisq(double x, double df, double ncp, bool log) {
    return Rmath::dnchisq(x, df, ncp, log);
  }
  double pnchisq(double x, double df, double ncp, bool low, bool log) {
    return Rmath::pnchisq(x, df, ncp, low, log);
  }
  double qnchisq(double p, double df, double ncp, bool low, bool log) {
    return Rmath::qnchisq(p, df, ncp, low, log);
  }

  /* F Distibution */
  double df(double x, double ndf, double ddf, bool log) {
    return Rmath::df(x, ndf, ddf, log);
  }
  double pf(double x, double ndf, double ddf, bool low, bool log) {
    return Rmath::pf(x, ndf, ddf, low, log);
  }
  double qf(double p, double ndf, double ddf, bool low, bool log) {
    return Rmath::qf(p, ndf, ddf, low, log);
  }
  double rf(double ndf, double ddf) { return Rmath::rf(ndf, ddf); }
  double rf_mt(RNG &rng, double ndf, double ddf) {
    return Rmath::rf_mt(rng, ndf, ddf);
  }

  /* Student t Distibution */
  double dt(double x, double df, bool log) { return Rmath::dt(x, df, log); }
  double pt(double x, double df, bool low, bool log) {
    return Rmath::pt(x, df, low, log);
  }
  double qt(double p, double df, bool low, bool log) {
    return Rmath::qt(p, df, low, log);
  }
  double rt(double df) { return Rmath::rt(df); }
  double rt_mt(RNG &rng, double df) { return Rmath::rt_mt(rng, df); }

  /* Binomial Distribution */
  double dbinom(double x, double n, double p, bool log) {
    return Rmath::dbinom(x, n, p, log);
  }
  double pbinom(double x, double n, double p, bool low, bool log) {
    return Rmath::pbinom(x, n, p, low, log);
  }
  double qbinom(double p, double n, double prob, bool low, bool log) {
    return Rmath::qbinom(p, n, prob, low, log);
  }
  unsigned rbinom(int n, double p) { return Rmath::rbinom(n, p); }
  unsigned rbinom_mt(RNG &rng, int n, double p) {
    return Rmath::rbinom_mt(rng, n, p);
  }

  /* Multinomial Distribution */
  std::vector<int> rmultinom_mt(RNG &rng, int64_t n,
                                const std::vector<double> &prob) {
    std::vector<int> result;
    Rmath::rmultinom_mt(rng, n, prob, result);
    return result;
  }

  void rmultinom(int64_t n, const std::vector<double> &prob,
                 std::vector<int> &result) {
    Rmath::rmultinom_mt(BOOM::GlobalRng::rng, n, prob, result);
  }

  std::vector<int> rmultinom(int64_t n, const std::vector<double> &prob) {
    std::vector<int> result;
    Rmath::rmultinom_mt(BOOM::GlobalRng::rng, n, prob, result);
    return result;
  }

  void rmultinom_mt(BOOM::RNG &rng, int64_t n, const std::vector<double> &prob,
                    std::vector<int> &result) {
    Rmath::rmultinom_mt(rng, n, prob, result);
  }

  /* Cauchy Distribution */
  double dcauchy(double x, double mu, double scal, bool log) {
    return Rmath::dcauchy(x, mu, scal, log);
  }
  double pcauchy(double x, double mu, double scal, bool low, bool log) {
    return Rmath::pcauchy(x, mu, scal, low, log);
  }
  double qcauchy(double p, double mu, double sig, bool low, bool log) {
    return Rmath::qcauchy(p, mu, sig, low, log);
  }
  double rcauchy(double mu, double sig) { return Rmath::rcauchy(mu, sig); }
  double rcauchy_mt(RNG &rng, double mu, double sig) {
    return Rmath::rcauchy_mt(rng, mu, sig);
  }

  /* Exponential Distribution */
  double dexp(double x, double lam, bool log) {
    return Rmath::dexp(x, 1.0 / lam, log);
  }
  double pexp(double x, double lam, bool low, bool log) {
    return Rmath::pexp(x, 1.0 / lam, low, log);
  }
  double qexp(double p, double lam, bool low, bool log) {
    return Rmath::qexp(p, 1.0 / lam, low, log);
  }
  double rexp(double lam) { return Rmath::rexp(1.0 / lam); }
  double rexp_mt(RNG &rng, double lam) {
    return Rmath::rexp_mt(rng, 1.0 / lam);
  }

  /* Geometric Distribution */
  double dgeom(double x, double p, bool log) { return Rmath::dgeom(x, p, log); }
  double pgeom(double x, double p, bool low, bool log) {
    return Rmath::pgeom(x, p, low, log);
  }
  double qgeom(double p, double prob, bool low, bool log) {
    return Rmath::qgeom(p, prob, low, log);
  }
  double rgeom(double p) { return Rmath::rgeom(p); }
  double rgeom_mt(RNG &rng, double p) { return Rmath::rgeom_mt(rng, p); }

  /* Hypergeometric Distibution */
  double dhyper(double x, double r, double b, double n, bool log) {
    return Rmath::dhyper(x, r, b, n, log);
  }
  double phyper(double x, double r, double b, double n, bool low, bool log) {
    return Rmath::phyper(x, r, b, n, low, log);
  }
  double qhyper(double p, double r, double b, double n, bool low, bool log) {
    return Rmath::qhyper(p, r, b, n, low, log);
  }
  double rhyper(double r, double b, double n) { return Rmath::rhyper(r, b, n); }
  double rhyper_mt(RNG &rng, double r, double b, double n) {
    return Rmath::rhyper_mt(rng, r, b, n);
  }

  /* Negative Binomial Distribution */
  double dnbinom(double x, double n, double p, bool log) {
    return Rmath::dnbinom(x, n, p, log);
  }
  double pnbinom(double x, double n, double p, bool low, bool log) {
    return Rmath::pnbinom(x, n, p, low, log);
  }
  double qnbinom(double p, double n, double prob, bool low, bool log) {
    return Rmath::qnbinom(p, n, prob, low, log);
  }
  double rnbinom(double n, double p) { return Rmath::rnbinom(n, p); }
  double rnbinom_mt(RNG &rng, double n, double p) {
    return Rmath::rnbinom_mt(rng, n, p);
  }

  /* Poisson Distribution */
  double dpois(double x, double lam, bool log) {
    return Rmath::dpois(x, lam, log);
  }
  double ppois(double x, double lam, bool low, bool log) {
    return Rmath::ppois(x, lam, low, log);
  }
  double qpois(double p, double lam, bool low, bool log) {
    return Rmath::qpois(p, lam, low, log);
  }
  double rpois(double lam) { return Rmath::rpois(lam); }
  double rpois_mt(RNG &rng, double lam) { return Rmath::rpois_mt(rng, lam); }

  /* Weibull Distribution */
  double dweibull(double x, double shape, double scale, bool log) {
    return Rmath::dweibull(x, shape, scale, log);
  }
  double pweibull(double x, double shape, double scale, bool low, bool log) {
    return Rmath::pweibull(x, shape, scale, low, log);
  }
  double qweibull(double p, double shape, double scale, bool low, bool log) {
    return Rmath::qweibull(p, shape, scale, low, log);
  }
  double rweibull(double shape, double scale) {
    return Rmath::rweibull(shape, scale);
  }
  double rweibull_mt(RNG &rng, double shape, double scale) {
    return Rmath::rweibull_mt(rng, shape, scale);
  }

  /* Logistic Distribution */
  double dlogis(double x, double mu, double sig, bool log) {
    return Rmath::dlogis(x, mu, sig, log);
  }
  double plogis(double x, double mu, double sig, bool low, bool log) {
    return Rmath::plogis(x, mu, sig, low, log);
  }
  double qlogis(double p, double mu, double sig, bool low, bool log) {
    return Rmath::qlogis(p, mu, sig, low, log);
  }
  double rlogis(double mu, double sig) { return Rmath::rlogis(mu, sig); }
  double rlogis_mt(RNG &rng, double mu, double sig) {
    return Rmath::rlogis_mt(rng, mu, sig);
  }

  /* Non-central Beta Distribution */
  double dnbeta(double x, double a, double b, double lam, bool log) {
    return Rmath::dnbeta(x, a, b, lam, log);
  }
  double pnbeta(double x, double a, double b, double lam, bool low, bool log) {
    return Rmath::pnbeta(x, a, b, lam, low, log);
  }

  /* Non-central F Distribution */
  double pnf(double x, double dfn, double dfd, double lam, bool low, bool log) {
    return Rmath::pnf(x, dfn, dfd, lam, low, log);
  }

  /* Non-central Student t Distribution */
  double pnt(double x, double df, double mu, bool low, bool log) {
    return Rmath::pnt(x, df, mu, low, log);
  }

  /* Gamma and Related Functions */
  double gamma(double x) { return Rmath::gammafn(x); }
  double lgamma(double x) { return Rmath::lgammafn(x); }
  double digamma(double x) { return Rmath::digamma(x); }
  double trigamma(double x) { return Rmath::trigamma(x); }
  double tetragamma(double x) { return Rmath::tetragamma(x); }
  double pentagamma(double x) { return Rmath::pentagamma(x); }

  double beta(double a, double b) { return Rmath::beta(a, b); }
  double lbeta(double a, double b) { return Rmath::lbeta(a, b); }

  double choose(double n, double p) { return Rmath::choose(n, p); }
  double lchoose(double n, double p) { return Rmath::lchoose(n, p); }

  double bessel_k(double a, double b, double c) {
    return Rmath::bessel_k(a, b, c);
  }

}  // namespace BOOM
