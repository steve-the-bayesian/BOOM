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

#include "Models/Glm/LogisticRegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/LogitSampler.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "TargetFun/LogPost.hpp"
#include "cpputil/lse.hpp"
#include "numopt.hpp"
#include "stats/logit.hpp"

namespace BOOM {

  LogisticRegressionModel::LogisticRegressionModel(uint beta_dim, bool all)
      : ParamPolicy(new GlmCoefs(beta_dim, all)), log_alpha_(0) {}

  LogisticRegressionModel::LogisticRegressionModel(const Vector &beta)
      : ParamPolicy(new GlmCoefs(beta)), log_alpha_(0) {}

  LogisticRegressionModel::LogisticRegressionModel(const Matrix &X,
                                                   const Vector &y,
                                                   bool add_int)
      : ParamPolicy(new GlmCoefs(X.ncol())), log_alpha_(0) {
    int n = nrow(X);
    for (int i = 0; i < n; ++i) {
      NEW(BinaryRegressionData, dp)(y[i] > .5, X.row(i));
      add_data(dp);
    }
  }

  LogisticRegressionModel::LogisticRegressionModel(
      const LogisticRegressionModel &rhs)
      : Model(rhs),
        GlmModel(rhs),
        NumOptModel(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        log_alpha_(rhs.log_alpha_) {}

  LogisticRegressionModel *LogisticRegressionModel::clone() const {
    return new LogisticRegressionModel(*this);
  }

  typedef LogisticRegressionModel LRM;
  typedef BinaryRegressionData BRD;

  double LRM::pdf(const Ptr<Data> &dp, bool logscale) const {
    Ptr<BRD> d = DAT(dp);
    double ans = logp(d->y(), d->x());
    return logscale ? ans : exp(ans);
  }

  double LRM::pdf(const Data *dp, bool logscale) const {
    const BRD *d = DAT(dp);
    double ans = logp(d->y(), d->x());
    return logscale ? ans : exp(ans);
  }

  double LRM::logp(bool y, const Vector &x) const {
    double btx = predict(x);
    double ans = -lope(btx);
    if (y) ans += btx;
    return ans;
  }

  double LRM::Loglike(const Vector &beta, Vector &g, Matrix &h, uint nd) const {
    if (nd >= 2) return log_likelihood(beta, &g, &h);
    if (nd == 1) return log_likelihood(beta, &g, 0);
    return log_likelihood(beta, 0, 0);
  }

  double LRM::log_likelihood(const Vector &beta, Vector *g, Matrix *h,
                             bool initialize_derivs) const {
    const LRM::DatasetType &data(dat());
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
    int n = data.size();
    bool all_coefficients_included = coef().nvars() == xdim();
    const Selector &inc(coef().inc());
    for (int i = 0; i < n; ++i) {
      bool y = data[i]->y();
      const Vector &x(data[i]->x());
      double eta = predict(x) + log_alpha_;
      double loglike = plogis(eta, 0, 1, y, true);
      ans += loglike;
      if (g) {
        double logp = y ? loglike : plogis(eta, 0, 1, true, true);
        double p = exp(logp);
        if (all_coefficients_included) {
          *g += (y - p) * x;
          if (h) {
            h->add_outer(x, x, -p * (1 - p));
          }
        } else {
          Vector reduced_x = inc.select(x);
          *g += (y - p) * reduced_x;
          if (h) {
            h->add_outer(reduced_x, reduced_x, -p * (1 - p));
          }
        }
      }
    }
    return ans;
  }

  d2TargetFunPointerAdapter LRM::log_likelihood_tf() const {
    return d2TargetFunPointerAdapter([this](const Vector &beta, Vector *g,
                                            Matrix *h, bool initialize_derivs) {
      return this->log_likelihood(beta, g, h, initialize_derivs);
    });
  }

  SpdMatrix LRM::xtx() const {
    const std::vector<Ptr<BinaryRegressionData> > &d(dat());
    uint n = d.size();
    uint p = d[0]->xdim();
    SpdMatrix ans(p);
    for (uint i = 0; i < n; ++i) ans.add_outer(d[i]->x(), 1.0, false);
    ans.reflect();
    return ans;
  }

  void LRM::set_nonevent_sampling_prob(double alpha) {
    if (alpha <= 0 || alpha > 1) {
      ostringstream err;
      err << "alpha (proportion of non-events retained in the data) "
          << "must be in (0,1]" << endl
          << "you set alpha = " << alpha << endl;
      report_error(err.str());
    }
    log_alpha_ = std::log(alpha);
  }

  double LRM::log_alpha() const { return log_alpha_; }

}  // namespace BOOM
