// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#include "Models/Glm/ProbitRegression.hpp"
#include <functional>
#include "Models/Glm/RegressionModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    typedef ProbitRegressionModel PRM;
    typedef BinaryRegressionData BRD;
  }  // namespace

  PRM::ProbitRegressionModel(const Vector &beta)
      : ParamPolicy(new GlmCoefs(beta)) {}

  PRM::ProbitRegressionModel(const Matrix &X, const Vector &y)
      : ParamPolicy(new GlmCoefs(ncol(X))) {
    int n = nrow(X);
    for (int i = 0; i < n; ++i) {
      NEW(BinaryRegressionData, dp)(y[i] > .5, X.row(i));
      add_data(dp);
    }
  }

  PRM::ProbitRegressionModel(const ProbitRegressionModel &rhs)
      : Model(rhs),
        GlmModel(rhs),
        NumOptModel(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs) {}

  PRM *PRM::clone() const { return new PRM(*this); }

  GlmCoefs &PRM::coef() { return ParamPolicy::prm_ref(); }
  const GlmCoefs &PRM::coef() const { return ParamPolicy::prm_ref(); }
  Ptr<GlmCoefs> PRM::coef_prm() { return ParamPolicy::prm(); }
  const Ptr<GlmCoefs> PRM::coef_prm() const { return ParamPolicy::prm(); }

  double PRM::pdf(const Ptr<Data> &dp, bool logscale) const {
    return pdf(DAT(dp), logscale);
  }

  double PRM::pdf(const Ptr<BinaryRegressionData> &dp, bool logscale) const {
    return pdf(dp->y(), dp->x(), logscale);
  }

  double PRM::pdf(bool y, const Vector &x, bool logscale) const {
    double eta = predict(x);
    if (y) return pnorm(eta, 0, 1, true, logscale);
    return pnorm(eta, 0, 1, false, logscale);
  }

  double PRM::Loglike(const Vector &beta, Vector &g, Matrix &h, uint nd) const {
    if (nd == 0)
      return log_likelihood(beta, 0, 0);
    else if (nd == 1)
      return log_likelihood(beta, &g, 0);
    return log_likelihood(beta, &g, &h);
  }

  // see probit_loglike.tex for the calculus
  double PRM::log_likelihood(const Vector &beta, Vector *g, Matrix *h,
                             bool initialize_derivs) const {
    const PRM::DatasetType &data(dat());
    int n = data.size();
    const Selector &inclusion_indicators(coef().inc());
    int beta_dim = inclusion_indicators.nvars();
    if (beta.size() != beta_dim) {
      report_error("Wrong size argument supplied to log_likelihood.");
    }
    if (initialize_derivs) {
      if (g) {
        g->resize(beta_dim);
        *g = 0;
        if (h) {
          h->resize(beta_dim, beta_dim);
          *h = 0;
        }
      }
    }
    bool all_coefficients_included = beta_dim == xdim();
    double ans = 0;
    for (int i = 0; i < n; ++i) {
      bool y = data[i]->y();
      const Vector &x(data[i]->x());
      double eta = predict(x);
      double increment = pnorm(eta, 0, 1, y, true);
      ans += increment;
      if (g) {
        double logp = y ? increment : pnorm(eta, 0, 1, true, true);
        double p = exp(logp);
        double q = 1 - p;
        double v = p * q;
        double resid = (static_cast<double>(y) - p) / v;
        double phi = dnorm(eta);
        Vector included_x;
        if (all_coefficients_included) {
          g->axpy(x, phi * resid);
        } else {
          included_x = inclusion_indicators.select(x);
          g->axpy(included_x, phi * resid);
        }
        if (h) {
          double pe = phi * resid;
          if (all_coefficients_included) {
            h->add_outer(x, x, -pe * (pe + eta));
          } else {
            h->add_outer(included_x, included_x, -pe * (pe + eta));
          }
        }
      }
    }
    return ans;
  }

  d2TargetFunPointerAdapter PRM::log_likelihood_tf() const {
    return d2TargetFunPointerAdapter([this](const Vector &beta,
                                            Vector *gradient, Matrix *hessian,
                                            bool reset) {
      return this->log_likelihood(beta, gradient, hessian, reset);
    });
  }

  bool PRM::sim(const Vector &x, RNG &rng) const {
    return runif_mt(rng) < pnorm(predict(x));
  }

  Ptr<BinaryRegressionData> PRM::sim(RNG &rng) const {
    Vector x(xdim());
    x[0] = 1.0;
    for (int i = 1; i < x.size(); ++i) x[i] = rnorm_mt(rng);
    bool y = this->sim(x, rng);
    NEW(BinaryRegressionData, ans)(y, x);
    return ans;
  }

}  // namespace BOOM
