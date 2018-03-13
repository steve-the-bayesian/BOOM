// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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

#include "Models/Glm/BinomialProbitModel.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    typedef BinomialProbitModel BPM;
    typedef BinomialRegressionData BRD;
  }  // namespace
  BPM::BinomialProbitModel(uint beta_dim, bool all)
      : ParamPolicy(new GlmCoefs(beta_dim, all)) {}

  BPM::BinomialProbitModel(const Vector &beta)
      : ParamPolicy(new GlmCoefs(beta)) {}

  BPM::BinomialProbitModel(const Ptr<GlmCoefs> &beta) : ParamPolicy(beta) {}

  BPM::BinomialProbitModel(const Matrix &X, const Vector &y, const Vector &n)
      : ParamPolicy(new GlmCoefs(X.ncol())) {
    int nr = nrow(X);
    for (int i = 0; i < nr; ++i) {
      uint yi = lround(y[i]);
      uint ni = lround(n[i]);
      NEW(BinomialRegressionData, dp)(yi, ni, X.row(i));
      add_data(dp);
    }
  }

  BPM::BinomialProbitModel(const BPM &rhs)
      : Model(rhs),
        MixtureComponent(rhs),
        GlmModel(rhs),
        NumOptModel(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs) {}

  BPM *BinomialProbitModel::clone() const {
    return new BinomialProbitModel(*this);
  }

  namespace {
    // Compute the probability of success (or failure) at a value of x.
    // Args:
    //   x:  The vector-like object at which the probability is desired.
    //   beta:  The probit regression coefficients multiplying x.
    //   success: If true, then the probability of success is
    //     returned.  Otherwise the probability of failure is returned.
    template <class VECTOR>
    double probit_success_probability(const VECTOR &x, const GlmCoefs &beta,
                                      bool success) {
      double eta = beta.predict(x);
      bool log_p = false;
      return pnorm(eta, 0, 1, success, log_p);
    }
  }  // namespace

  double BPM::success_probability(const Vector &x) const {
    return probit_success_probability(x, coef(), true);
  }
  double BPM::success_probability(const VectorView &x) const {
    return probit_success_probability(x, coef(), true);
  }
  double BPM::success_probability(const ConstVectorView &x) const {
    return probit_success_probability(x, coef(), true);
  }
  double BPM::failure_probability(const Vector &x) const {
    return probit_success_probability(x, coef(), false);
  }
  double BPM::failure_probability(const VectorView &x) const {
    return probit_success_probability(x, coef(), false);
  }
  double BPM::failure_probability(const ConstVectorView &x) const {
    return probit_success_probability(x, coef(), false);
  }

  double BPM::pdf(const Ptr<Data> &dp, bool logscale) const {
    return pdf(DAT(dp), logscale);
  }

  double BPM::pdf(const Data *dp, bool logscale) const {
    const BinomialRegressionData *rd =
        dynamic_cast<const BinomialRegressionData *>(dp);
    return logp(rd->y(), rd->n(), rd->x(), logscale);
  }

  double BPM::pdf(const Ptr<BRD> &dp, bool logscale) const {
    return logp(dp->y(), dp->n(), dp->x(), logscale);
  }

  double BPM::logp_1(bool y, const Vector &x, bool logscale) const {
    return pnorm(0, predict(x), 1, y, logscale);
  }

  // In many cases y and n will be set using integers, so they will
  // compare to integer literals exactly.  Only rarely are they non-integers.
  double BPM::logp(double y, double n, const Vector &x, bool logscale) const {
    if (n == 0) {
      double ans = y == 0 ? 0 : negative_infinity();
      return logscale ? ans : exp(ans);
    } else if (n == 1 && (y == 0 || y == 1)) {
      // This is a common special case of the more general calcualtion
      // in the next branch.  Special handling here for efficiency
      // reasons.
      return logp_1(y, x, logscale);
    } else {
      double p = pnorm(0, predict(x), 1);
      return dbinom(y, n, p, logscale);
    }
  }

  double BPM::Loglike(const Vector &beta, Vector &g, Matrix &h, uint nd) const {
    if (nd >= 2) return log_likelihood(beta, &g, &h);
    if (nd == 1) return log_likelihood(beta, &g, 0);
    return log_likelihood(beta, 0, 0);
  }

  double BPM::log_likelihood(const Vector &beta, Vector *g, Matrix *h,
                             bool initialize_derivs) const {
    const BPM::DatasetType &data(dat());
    if (initialize_derivs) {
      if (g) {
        g->resize(beta.size());
        *g = 0;
        if (h) {
          h->resize(beta.size(), beta.size());
          *h = 0;
        }
      }
    }
    double ans = 0;
    bool all_coefficients_included = (xdim() == beta.size());
    const Selector &inc(coef().inc());
    for (int i = 0; i < data.size(); ++i) {
      // y and n had been defined as uint's but y-n*p was computing
      // -n, which overflowed
      const double y = data[i]->y();
      const double n = data[i]->n();
      const Vector &x(data[i]->x());
      Vector reduced_x;
      if (!all_coefficients_included) {
        reduced_x = inc.select(x);
      }
      ConstVectorView X(all_coefficients_included ? x : reduced_x);

      const double eta = beta.dot(X);
      const double p = pnorm(eta);
      ans += dbinom(y, n, p, true);
      if (g) {
        // The derivative of p is phi(eta) * X.
        const double phi = dnorm(eta);
        const double phat = n > 0 ? y / n : 0;
        const double pq = p * (1 - p);
        // Scaled residual e, which is morally (phat - p) / pq.  But
        // we need to handle the case where pq == 0.
        double e;
        if (pq <= 0.0) {
          constexpr double epsilon = std::numeric_limits<double>::epsilon();
          if (fabs(y) < epsilon && fabs(p) < epsilon) {
            e = -1.0 / (1 - p);
          } else if (fabs(n - y) < epsilon && fabs(1 - p) < epsilon) {
            e = 0;
          } else {
            // Trouble here.  You've got a zero p or 1-p, but y is
            // neither 0 (zero p case) nor n (p == 1 case).
            ostringstream err;
            err << "In observation " << i << ", first derivative," << std::endl
                << "with y = " << y << std::endl
                << "and  n = " << n << std::endl
                << "p = " << p << std::endl
                << "eta = " << eta << std::endl
                << "reduced_x = " << reduced_x << std::endl
                << "beta = " << beta << std::endl;
            report_error(err.str());
            // Will never get here, but silence compiler warnings
            // about uninitialzed values.
            e = negative_infinity();
          }
        } else {
          e = (phat - p) / pq;
        }
        g->axpy(X, n * phi * e);
        if (h) {
          // Compute the relevant pieces needed to evaluate the
          // deriviative using the product rule.
          //
          // g = (nx) * phi * e, so
          // h = (nx) * [(dphi * e) + (de * phi)]
          //
          const double pqhat = phat * (1 - phat);
          double de;
          // Compute the derivative of e, handling the possibility
          // that pq == 0.
          if (pq > 0) {
            // Normal case
            de = -phi * (square(e) + pqhat / square(pq));
          } else {
            constexpr double epsilon = std::numeric_limits<double>::epsilon();
            if (fabs(p) < epsilon && y < epsilon) {
              // Here e = -1/(1-p), so de = 1/(1-p)^2 * (-phi x).
              // With p == 0 we know 1 - p = 1, so we can skip that part.
              de = -phi;
            } else if (fabs(1 - p) < epsilon && fabs(n - y) < epsilon) {
              // Here phat = 1 and p = 1, so e = 0.
              de = 0;
            } else {
              // Trouble here.  You've got a zero p or 1-p, but y is
              // neither 0 (zero p case) nor n (p == 1 case).
              ostringstream err;
              err << "In observation " << i << ", second derivative,"
                  << std::endl
                  << "with y = " << y << std::endl
                  << "and  n = " << n << std::endl
                  << "p = " << p << std::endl
                  << "eta = " << eta << std::endl
                  << "reduced_x = " << reduced_x << std::endl
                  << "beta = " << beta << std::endl;
              report_error(err.str());
              // Will never get here, but silence compiler warnings
              // about uninitialzed values.
              de = negative_infinity();
            }
          }
          double dphi = -phi * eta;
          // Derivative using the product rule.  Both de and dphi have
          // an extra factor of x^T that will be handled by add_outer.
          double d2 = n * (phi * de + e * dphi);
          h->add_outer(X, X, d2);
        }
      }
    }
    return ans;
  }

  d2TargetFunPointerAdapter BPM::log_likelihood_tf() const {
    return d2TargetFunPointerAdapter(
        [this](const Vector &beta, Vector *g, Matrix *h, bool reset_derivs) {
          return this->log_likelihood(beta, g, h, reset_derivs);
        });
  }

  SpdMatrix BPM::xtx() const {
    const std::vector<Ptr<BinomialRegressionData> > &d(dat());
    uint n = d.size();
    uint p = d[0]->xdim();
    SpdMatrix ans(p);
    for (uint i = 0; i < n; ++i) {
      double n = d[i]->n();
      ans.add_outer(d[i]->x(), n, false);
    }
    ans.reflect();
    return ans;
  }

}  // namespace BOOM
