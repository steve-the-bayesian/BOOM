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

#ifndef BOOM_DISTRIBUTIONS_HPP
#define BOOM_DISTRIBUTIONS_HPP

#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"

#include "distributions/Rmath_dist.hpp"
#include "distributions/rng.hpp"

#include <vector>
#include "uint.hpp"

namespace BOOM {
  class VectorView;
  class ConstVectorView;

  class TnSampler {
    // Implements adaptive rejection sampling for drawing from the
    // truncated normal distribution given x>a>0.
   public:
    explicit TnSampler(double a);      // Set the truncation point.
    double draw(RNG &);                // simluate a value
    void add_point(double x);          // adds the point to the hull
    double f(double x) const;          // log of the target distribution
    double df(double x) const;         // derivative of logf at x
    double h(double x, uint k) const;  // evaluates the outer hull at x
    std::ostream &print(std::ostream &out) const;

   private:
    std::vector<double> x;
    // points that have been tried thus far, stored in ascending
    // order

    std::vector<double> logf;
    // function values corresponding to values in x

    std::vector<double> dlogf;
    // derivatives of the log target density evaluated at x

    std::vector<double> knots;
    // contains the points of intersection between the tangent lines
    // to logf at x.  First knot is x[0].  Later knots satisfy x[i-1]
    // < knots[i] < x[i].

    std::vector<double> cdf;  // cdf[i] = cdf[i-1] + the integral of
    // the hull from knots[i] to
    // knots[i+1].  cdf.back() assumes a
    // final knot at infinity

    void update_cdf();
    void refresh_knots();
    double compute_knot(uint k) const;
    using IT = std::vector<double>::iterator;
  };

  class Tn2Sampler {
    // implements adaptive rejection sampling for drawinf from the
    // truncated standard normal distribution with 0 < lo < x < hi
   public:
    Tn2Sampler(double lo, double hi);
    double draw(RNG &);
    void add_point(double x);
    double f(double x) const;
    double df(double x) const;

    // hull is the envelope distribution on the log scale
    double hull(double x, uint k) const;

   private:
    // x contains the values of the points that have been tried.
    // initialized with lo and hi
    std::vector<double> x;

    // logf contains the values of the target density at all the
    // points in x.
    std::vector<double> logf;

    // derivatives corresponding to logf
    std::vector<double> dlogf;

    // first knot is at lo.  last is at hi.  interior knots contain
    // the point of intersection of the tangent lines to logf at the
    // points in x
    std::vector<double> knots;

    // cdf[0] = integral of first bit of hull.  cdf[i] = cdf[i-1] +
    // integral of hull part i;
    std::vector<double> cdf;

    void update_cdf();
    void refresh_knots();
    double compute_knot(uint k) const;
    using IT = std::vector<double>::iterator;
  };

  double trun_norm(double);
  double trun_norm_mt(RNG &, double);

  // Returns the mean and the variance of the truncated normal
  // distribution, where the untruncated distribution is N(mu, sigma).
  // If positive_support is true, the region of support is from
  // cutpoint to infinity.  Otherwise the region of support is from
  // cutpoint to -infinity.
  //
  // On output, *mean and *variance contain the mean and variance of
  // the truncated distribution.
  //
  // Be aware that the standard deviation of the untruncated
  // distribution is input.  The variance of the truncated
  // distribution is output.
  void trun_norm_moments(double mu, double sigma, double cutpoint,
                         bool positive_support, double *mean, double *variance);

  // Args:
  //   mu:  Mean of the (untruncated) normal distribution.
  //   sigma:  Standard deviation of the (untruncated) normal distribution.
  //   cutpoint:  The point of truncation.
  //   positive_support: If true, then draw a deviate from above the cutpoint.
  //     If false then draw a deviate from below the cutpoint.
  // Returns:
  //   A draw from the truncated normal distribution.
  double rtrun_norm(double mu, double sigma, double cutpoint,
                    bool positive_support = true);
  double rtrun_norm_mt(RNG &, double mu, double sigma, double cutpoint,
                       bool positive_support = true);

  double dtrun_norm(double, double, double, double, bool low = true,
                    bool log = false);
  double dtrun_norm_2(double, double, double, double, double, bool log = false);

  double rtrun_norm_2(double mu, double sig, double lo, double hi);
  double rtrun_norm_2_mt(RNG &, double mu, double sig, double lo, double hi);

  double dstudent(double y, double mu, double sigma, double df,
                  bool log = false);
  double rstudent(double mu, double sigma, double df);
  double rstudent_mt(RNG &rng, double mu, double sigma, double df);

  double rtrun_exp_mt(RNG &rng, double lam, double lo, double hi);
  double rtrun_exp(double lam, double lo, double hi);
  double rpiecewise_log_linear_mt(RNG &rng, double slope, double lo, double hi);

  double rlexp(double loglam);               // log E(lam).  loglam = log(lam)
  double rlexp_mt(RNG &rng, double loglam);  // log E(lam).  loglam = log(lam)

  // extreme value distribution with centrality parameter 'mu + gamma', where
  // gamma is Euler's constant -0.5772157... and variance 'sigma^2 * pi^2/6'.  

  double pexv(double x, double mu = 0, double sigma = 1, bool logscale = false);
  double dexv(double x, double mu = 0., double sigma = 1., bool logscale = false);
  double rexv_mt(RNG &rng, double mu = 0., double sigma = 1.);
  double rexv(double mu = 0., double sigma = 1.);

  // random integer uniform on lo to hi, inclusive
  int random_int(int lo, int hi);
  int random_int_mt(RNG &rng, int lo, int hi);

  // Returns an n-vector of independent normal deviates, each with mean mu and
  // standard deviation sigma.
  inline Vector rnorm_vector(int n, double mu, double sigma) {
    if (n <= 0) {
      return Vector(0);
    }
    Vector ans(n);
    for (int i = 0; i < n; ++i) {
      ans[i] = rnorm_mt(GlobalRng::rng, mu, sigma);
    }
    return ans;
  }

  //======================================================================
  // Several varieties of multivariate normal generation.
  //
  // basic rmvn checks the cholesky decomposition, if there is a
  // problem it calls rmvn_robust
  Vector rmvn(const Vector &Mu, const SpdMatrix &Sigma);
  Vector rmvn_mt(RNG &rng, const Vector &Mu, const SpdMatrix &Sigma);

  // rmvn_robust computes the spectral decomposition of Sigma which
  // can be done even if there is a zero pivot that would prevent the
  // Cholesky decomposition from working, so it can be used even if
  // Sigma is only positive semidefinite.
  Vector rmvn_robust(const Vector &Mu, const SpdMatrix &Sigma);
  Vector rmvn_robust_mt(RNG &rng, const Vector &Mu, const SpdMatrix &Sigma);

  // Simulate given the lower cholesky triangle of the variance matrix.
  Vector rmvn_L(const Vector &mu, const Matrix &L);
  Vector rmvn_L_mt(RNG &rng, const Vector &mu, const Matrix &L);

  Vector rmvn_ivar(const Vector &Mu, const SpdMatrix &Sigma_Inverse);
  Vector rmvn_ivar_mt(RNG &rng, const Vector &Mu, const SpdMatrix &precision);

  // Simulate using the upper cholesky triangle of the precision matrix.
  // Args:
  //   rng:  The U(0, 1) random number generator to use for the simulation.
  //   mean:  The mean of the distribution to be simulated.
  //   precision_upper_cholesky: The upper cholesky triangle of the precision
  //     matrix for the distribution to be simulated.
  // Returns:
  //   A draw from the N(mu, Sigma) distribution, where Sigma^{-1} = L * L',
  //   with L' = precision_upper_cholesky.
  Vector rmvn_precision_upper_cholesky_mt(
      RNG &rng, const Vector &mean, const Matrix &precision_upper_cholesky);

  // Simulate given the precision matrix, and the precision matrix
  // times the mean.  This form arises frequently in Bayesian
  // inference.
  Vector rmvn_suf(const SpdMatrix &Ivar, const Vector &IvarMu);
  Vector rmvn_suf_mt(RNG &rng, const SpdMatrix &Ivar, const Vector &IvarMu);

  //======================================================================
  // Evaluates the multivariate normal density function.
  // Args:
  //   y:  The location where the density is to be evalutated.
  //   mu: Mean of the distribution.
  //  Siginv:  Precision (inverse variance) matrix.
  //  ldsi:  Log determinant of sigma inverse.
  //  logscale:  If true then the log of the density is returned.
  //
  // Returns:
  //   The value of the multivariate normal density at the specified
  //   location.
  double dmvn(const Vector &y, const Vector &mu, const SpdMatrix &Siginv,
              double ldsi, bool logscale);
  double dmvn_zero_mean(const Vector &y, const SpdMatrix &Siginv, double ldsi,
                        bool logscale);
  double dmvn(const Vector &y, const Vector &mu, const SpdMatrix &Siginv,
              bool logscale);

  // Y~ matrix_normal(Mu, Siginv, Ominv) if
  // Vector(Y) ~ N(Vector(Mu), (Siginv \otimes Ominv)^{-1})

  Matrix rmatrix_normal_ivar(const Matrix &Mu, const SpdMatrix &Siginv,
                             const SpdMatrix &Ominv);
  Matrix rmatrix_normal_ivar_mt(RNG &rng, const Matrix &Mu,
                                const SpdMatrix &Siginv,
                                const SpdMatrix &Ominv);
  double dmatrix_normal_ivar(const Matrix &Y, const Matrix &Mu,
                             const SpdMatrix &Siginv, const SpdMatrix &Ominv,
                             bool logscale);
  double dmatrix_normal_ivar(const Matrix &Y, const Matrix &Mu,
                             const SpdMatrix &Siginv, double ldsi,
                             const SpdMatrix &Ominv, double ldoi,
                             bool logscale);

  //  uniforms shrinkage prior in usp.cpp
  double dusp(double x, double z0, bool logscale);
  double pusp(double x, double z0, bool logscale);
  double qusp(double p, double z0);

  double rusp(double z0);
  double rusp_mt(RNG &rng, double z0);

  //  SpdMatrix rWish( double,  SpdMatrix &);
  SpdMatrix rWish(double df, const SpdMatrix &sumsq_inv, bool inv = false);
  SpdMatrix rWish_mt(RNG &, double df, const SpdMatrix &sumsq_inv,
                     bool inv = false);
  SpdMatrix rWishChol(double df, const Matrix &sumsq_upper_chol,
                      bool inv = false);
  SpdMatrix rWishChol_mt(RNG &, double df, const Matrix &sumsq_upper_chol,
                         bool inv = false);
  double dWish(const SpdMatrix &S, const SpdMatrix &sumsq, double df,
               bool logscale, bool inv = false);
  inline double dWishinv(const SpdMatrix &S, const SpdMatrix &sumsq, double df,
                         bool logscale) {
    return dWish(S, sumsq, df, logscale, true);
  }

  double ddirichlet(const Vector &x, const Vector &nu, bool logscale);
  double ddirichlet(const VectorView &x, const Vector &nu, bool logscale);
  double ddirichlet(const Vector &x, const VectorView &nu, bool logscale);
  double ddirichlet(const VectorView &x, const VectorView &nu, bool logscale);
  double ddirichlet(const Vector &x, const ConstVectorView &nu, bool logscale);
  double ddirichlet(const ConstVectorView &x, const Vector &nu, bool logscale);
  double ddirichlet(const ConstVectorView &x, const ConstVectorView &nu,
                    bool logscale);
  double ddirichlet(const VectorView &x, const ConstVectorView &nu,
                    bool logscale);
  double ddirichlet(const ConstVectorView &x, const VectorView &nu,
                    bool logscale);

  Vector mdirichlet(const Vector &nu);
  double dirichlet_loglike(const Vector &nu, Vector *g, Matrix *h,
                           const Vector &sumlogpi, double nobs);

  Vector rdirichlet(const Vector &nu);
  Vector rdirichlet_mt(RNG &rng, const Vector &nu);
  Vector rdirichlet(const VectorView &nu);
  Vector rdirichlet_mt(RNG &rng, const VectorView &nu);
  Vector rdirichlet(const ConstVectorView &nu);
  Vector rdirichlet_mt(RNG &rng, const ConstVectorView &nu);

  uint rmulti(const Vector &);
  uint rmulti(const VectorView &);
  uint rmulti(const ConstVectorView &);
  uint rmulti_mt(RNG &rng, const Vector &);
  uint rmulti_mt(RNG &rng, const VectorView &);
  uint rmulti_mt(RNG &rng, const ConstVectorView &);

  int rmulti(int, int);
  int rmulti_mt(RNG &, int, int);

  double dmvt(const Vector &x, const Vector &mu, const SpdMatrix &Siginv,
              double nu, double ldsi, bool logscale);
  double dmvt(const Vector &x, const Vector &mu, const SpdMatrix &Siginv,
              double nu, bool logscale);

  Vector rmvt(const Vector &mu, const SpdMatrix &Sigma, double nu);
  Vector rmvt_ivar(const Vector &mu, const SpdMatrix &Sigma, double nu);
  Vector rmvt_mt(RNG &, const Vector &mu, const SpdMatrix &Sigma, double nu);
  Vector rmvt_ivar_mt(RNG &, const Vector &mu, const SpdMatrix &Sigma,
                      double nu);

}  // namespace BOOM

#endif  // BOOM_DISTRIBUTIONS_HPP
