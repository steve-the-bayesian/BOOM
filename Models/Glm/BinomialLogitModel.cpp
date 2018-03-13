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
#include "Models/Glm/BinomialLogitModel.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "stats/logit.hpp"

namespace BOOM {
  namespace {
    typedef BinomialLogitModel BLM;
    typedef BinomialRegressionData BRD;
  }  // namespace

  BLM::BinomialLogitModel(uint beta_dim, bool all)
      : ParamPolicy(new GlmCoefs(beta_dim, all)), log_alpha_(0) {}

  BLM::BinomialLogitModel(const Vector &beta)
      : ParamPolicy(new GlmCoefs(beta)), log_alpha_(0) {}

  BLM::BinomialLogitModel(const Ptr<GlmCoefs> &beta)
      : ParamPolicy(beta), log_alpha_(0) {}

  BLM::BinomialLogitModel(const Matrix &X, const Vector &y, const Vector &n)
      : ParamPolicy(new GlmCoefs(X.ncol())), log_alpha_(0) {
    int nr = nrow(X);
    for (int i = 0; i < nr; ++i) {
      uint yi = lround(y[i]);
      uint ni = lround(n[i]);
      NEW(BinomialRegressionData, dp)(yi, ni, X.row(i));
      add_data(dp);
    }
  }

  BLM::BinomialLogitModel(const BLM &rhs)
      : Model(rhs),
        MixtureComponent(rhs),
        GlmModel(rhs),
        NumOptModel(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        log_alpha_(rhs.log_alpha_) {}

  BLM *BinomialLogitModel::clone() const {
    return new BinomialLogitModel(*this);
  }

  namespace {
    // Compute the probability of success (or failure) at a value of x.
    // Args:
    //   x:  The vector-like object at which the probability is desired.
    //   beta:  The logistic regression coefficients multiplying x.
    //   success: If true, then the probability of success is
    //     returned.  Otherwise the probability of failure is returned.
    template <class VECTOR>
    double logit_success_probability(const VECTOR &x, const GlmCoefs &beta,
                                     bool success) {
      double eta = beta.predict(x);
      return plogis(eta, 0, 1, success, false);
    }
  }  // namespace

  double BLM::success_probability(const Vector &x) const {
    return logit_success_probability(x, coef(), true);
  }
  double BLM::success_probability(const VectorView &x) const {
    return logit_success_probability(x, coef(), true);
  }
  double BLM::success_probability(const ConstVectorView &x) const {
    return logit_success_probability(x, coef(), true);
  }
  double BLM::failure_probability(const Vector &x) const {
    return logit_success_probability(x, coef(), false);
  }
  double BLM::failure_probability(const VectorView &x) const {
    return logit_success_probability(x, coef(), false);
  }
  double BLM::failure_probability(const ConstVectorView &x) const {
    return logit_success_probability(x, coef(), false);
  }

  double BLM::pdf(const Ptr<Data> &dp, bool logscale) const {
    return pdf(DAT(dp), logscale);
  }

  double BLM::pdf(const Data *dp, bool logscale) const {
    const BinomialRegressionData *rd =
        dynamic_cast<const BinomialRegressionData *>(dp);
    return logp(rd->y(), rd->n(), rd->x(), logscale);
  }

  double BLM::pdf(const Ptr<BRD> &dp, bool logscale) const {
    return logp(dp->y(), dp->n(), dp->x(), logscale);
  }

  double BLM::logp_1(bool y, const Vector &x, bool logscale) const {
    double btx = predict(x);
    double ans = -lope(btx);
    if (y) ans += btx;
    return logscale ? ans : exp(ans);
  }

  // In many cases y and n will be set using integers, so they will
  // compare to integer literals exactly.  Only rarely are they non-integers.
  double BLM::logp(double y, double n, const Vector &x, bool logscale) const {
    if (n == 0) {
      double ans = y == 0 ? 0 : negative_infinity();
      return logscale ? ans : exp(ans);
    } else if (n == 1 && (y == 0 || y == 1)) {
      // This is a common special case of the more general calcualtion
      // in the next branch.  Special handling here for efficiency
      // reasons.
      return logp_1(y, x, logscale);
    } else {
      double eta = predict(x);
      double p = logit_inv(eta);
      return dbinom(y, n, p, logscale);
    }
  }

  double BLM::Loglike(const Vector &beta, Vector &g, Matrix &h, uint nd) const {
    if (nd >= 2) return log_likelihood(beta, &g, &h);
    if (nd == 1) return log_likelihood(beta, &g, 0);
    return log_likelihood(beta, 0, 0);
  }

  double BLM::log_likelihood(const Vector &beta, Vector *g, Matrix *h,
                             bool initialize_derivs) const {
    const BLM::DatasetType &data(dat());
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
      double y = data[i]->y();
      double n = data[i]->n();
      const Vector &x(data[i]->x());
      Vector reduced_x;
      if (!all_coefficients_included) {
        reduced_x = inc.select(x);
      }
      ConstVectorView X(all_coefficients_included ? x : reduced_x);

      double eta = beta.dot(X) - log_alpha_;
      double p = logit_inv(eta);
      double loglike = dbinom(y, n, p, true);
      ans += loglike;
      if (g) {
        g->axpy(X, y - n * p);  // g += (y-n*p) * x;
        if (h) {
          h->add_outer(X, X, -n * p * (1 - p));  // h += -npq * x x^T
        }
      }
    }
    return ans;
  }

  d2TargetFunPointerAdapter BLM::log_likelihood_tf() const {
    return d2TargetFunPointerAdapter(
        [this](const Vector &x, Vector *g, Matrix *h, bool reset_derivs) {
          return this->log_likelihood(x, g, h, reset_derivs);
        });
  }

  SpdMatrix BLM::xtx() const {
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

  void BLM::set_nonevent_sampling_prob(double alpha) {
    if (alpha <= 0 || alpha > 1) {
      ostringstream err;
      err << "alpha (proportion of non-events retained in the data) "
          << "must be in (0,1]" << endl
          << "you set alpha = " << alpha << endl;
      report_error(err.str());
    }
    log_alpha_ = std::log(alpha);
  }

  double BLM::log_alpha() const { return log_alpha_; }

}  // namespace BOOM
