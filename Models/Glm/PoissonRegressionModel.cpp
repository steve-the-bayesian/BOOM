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

#include "Models/Glm/PoissonRegressionModel.hpp"
#include <functional>
#include "TargetFun/TargetFun.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "numopt.hpp"
#include "numopt/initialize_derivatives.hpp"

namespace BOOM {

  PoissonRegressionModel::PoissonRegressionModel(int xdim)
      : ParamPolicy(new GlmCoefs(xdim)) {}

  PoissonRegressionModel::PoissonRegressionModel(const Vector &beta)
      : ParamPolicy(new GlmCoefs(beta)) {}

  PoissonRegressionModel::PoissonRegressionModel(const Ptr<GlmCoefs> &beta)
      : ParamPolicy(beta) {}

  PoissonRegressionModel *PoissonRegressionModel::clone() const {
    return new PoissonRegressionModel(*this);
  }

  GlmCoefs &PoissonRegressionModel::coef() { return ParamPolicy::prm_ref(); }
  const GlmCoefs &PoissonRegressionModel::coef() const {
    return ParamPolicy::prm_ref();
  }

  Ptr<GlmCoefs> PoissonRegressionModel::coef_prm() {
    return ParamPolicy::prm();
  }

  const Ptr<GlmCoefs> PoissonRegressionModel::coef_prm() const {
    return ParamPolicy::prm();
  }

  double PoissonRegressionModel::log_likelihood(const Vector &beta, Vector *g,
                                                Matrix *h,
                                                bool reset_derivatives) const {
    // L = (E *lambda)^y exp(-E*lambda)
    //   ell = y * (log(E) + log(lambda)) - E*exp(x * beta)
    //       = yXbeta - E*exp(Xbeta)
    // dell  = (y - E*lambda) * x
    // ddell = -lambda * x * x'
    double ans = 0;
    const std::vector<Ptr<PoissonRegressionData> > &data(dat());
    const Selector &included(inc());
    int nvars = included.nvars();
    if (beta.size() != nvars) {
      ostringstream err;
      err << "Error in PoissonRegressionModel::log_likelihood.  Argument beta "
          << "is of dimension " << beta.size() << " but there are " << nvars
          << " included predictors." << endl;
      report_error(err.str());
    }
    initialize_derivatives(g, h, nvars, reset_derivatives);

    for (int i = 0; i < data.size(); ++i) {
      const Vector x = included.select(data[i]->x());
      int64_t y = data[i]->y();
      double lambda = 1.0;
      if (nvars > 0) {
        double eta = beta.dot(x);
        lambda = exp(eta);
      }
      double exposure = data[i]->exposure();
      ans += dpois(y, exposure * lambda, true);
      if (g) {
        g->axpy(x, (y - exposure * lambda));
        if (h) {
          h->add_outer(x, x, -lambda);
        }
      }
    }
    return ans;
  }

  double PoissonRegressionModel::Loglike(const Vector &beta, Vector &g,
                                         Matrix &h, uint nd) const {
    Vector *gp = NULL;
    Matrix *hp = NULL;
    if (nd > 0) gp = &g;
    if (nd > 1) hp = &h;
    return log_likelihood(beta, gp, hp, true);
  }

  void PoissonRegressionModel::mle() {
    Vector beta = included_coefficients();
    d2TargetFunPointerAdapter target(
        [this](const Vector &x, Vector *gradient, Matrix *hessian, bool reset) {
          return this->log_likelihood(x, gradient, hessian, reset);
        });
    Vector gradient;
    Matrix hessian;
    double function_value;
    std::string error_message;
    bool ok =
        max_nd2_careful(beta, gradient, hessian, function_value, Target(target),
                        dTarget(target), d2Target(target), 1e-5, error_message);
    if (!ok) {
      beta = 0;
    }
    set_included_coefficients(beta);
  }

  double PoissonRegressionModel::pdf(const Data *dp, bool logscale) const {
    // const PoissonRegressionData *d(
    //     dynamic_cast<const PoissonRegressionData *>(dp));
    const PoissonRegressionData *d(DAT(dp));
    double ans = logp(*d);
    return logscale ? ans : exp(ans);
  }

  double PoissonRegressionModel::logp(const PoissonRegressionData &data) const {
    double lambda = exp(predict(data.x()));
    return dpois(data.y(), data.exposure() * lambda, true);
  }

}  // namespace BOOM
