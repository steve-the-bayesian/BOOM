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

#include "Models/Glm/OrdinalCutpointModel.hpp"
#include "TargetFun/TargetFun.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "stats/Design.hpp"

#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>

namespace BOOM {
  inline double compute_delta(uint m, const Vector &v, uint maxscore) {
    if (m <= maxscore && m > 1) {
      return v(m - 2);
    } else if (m == 0) {
      return BOOM::negative_infinity();
    } else if (m == 1) {
      return 0.0;
    } else if (m == maxscore + 1) {
      return BOOM::infinity();
    }
    report_error("m out of bounds in OrdinalCutpointModel::delta");
    return 0.0;
  }

  typedef OrdinalCutpointModel OCM;

  // logically, delta goes from 2..Maxscore
  // with base zero vectors it goes from 0..Maxscore-2

  inline Vector make_delta(uint maxscore) {
    if (maxscore < 2) return Vector();
    Vector delta(maxscore - 1);
    for (int i = 0; i < delta.size(); ++i) delta[i] = i + 1;
    return delta;
  }

  OCM::OrdinalCutpointModel(const Vector &b, const Vector &d)
      : ParamPolicy(new GlmCoefs(b), new VectorParams(d)) {}

  OCM::OrdinalCutpointModel(const Vector &b, const Selector &Inc,
                            const Vector &d)
      : ParamPolicy(new GlmCoefs(b, Inc), new VectorParams(d)) {}

  OCM::OrdinalCutpointModel(const Selector &Inc, uint Maxscore)
      : ParamPolicy(new GlmCoefs(Vector(Inc.nvars(), 0.0), Inc),
                    new VectorParams(make_delta(Maxscore))) {}

  OCM::OrdinalCutpointModel(const Matrix &X, const Vector &Y)
      : ParamPolicy(new GlmCoefs(X.ncol()),
                    new VectorParams(make_delta(lround(max(Y))))) {
    uint n = Y.size();
    std::vector<uint> y_int(n);
    for (uint i = 0; i < n; ++i) y_int[i] = rint(Y[i]);  // round to nearest int

    std::vector<Ptr<OrdinalData> > ord_vec = make_ord_ptrs(y_int);
    for (uint i = 0; i < n; ++i) {
      dat().push_back(
          new OrdinalRegressionData(ord_vec[i], new VectorData(X.row(i))));
    }
    mle();
  }

  OCM::OrdinalCutpointModel(const OCM &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        GlmModel(rhs),
        NumOptModel(rhs) {}

  //  OCM * OCM::clone() const {return new OCM(*this);}

  double OCM::pdf(const Ptr<Data> &dp, bool logscale) const {
    Ptr<OrdinalRegressionData> dpo = DAT(dp);
    return pdf(dpo->y(), dpo->x(), logscale);
  }

  double OCM::pdf(const Ptr<OrdinalRegressionData> &dpo, bool logscale) const {
    return pdf(dpo->y(), dpo->x(), logscale);
  }

  double OCM::pdf(const OrdinalData &Y, const Vector &X, bool logscale) const {
    uint y = Y.value();
    return pdf(y, X, logscale);
  }

  double OCM::pdf(uint y, const Vector &X, bool logscale) const {
    uint M = maxscore();
    if (y > M) {
      report_error("ordinal data out of bounds in OrdinalCutpointModel::pdf");
    }
    double btx = predict(X);  // X may or may not contain intercept
    double F1 = y == M ? 1.0 : link_inv(delta(y + 1) - btx);
    double F0 = y == 0 ? 0.0 : link_inv(delta(y) - btx);
    double ans = F1 - F0;
    return logscale ? log(ans) : ans;
  }

  bool OCM::check_delta(const Vector &d) const {
    if (d.empty()) return true;  // a zero length vector is okay
    if (d[0] <= 0) return false;
    for (uint i = 1; i < d.size(); ++i) {
      if (d[i] <= d[i - 1]) {
        return false;
      }
    }
    return true;
  }

  double OCM::log_likelihood(const Vector &full_beta,
                             const Vector &delta) const {
    const std::vector<Ptr<OrdinalRegressionData> > &data(dat());
    int n = data.size();
    double ans = 0;
    int M = maxscore();
    for (int i = 0; i < n; ++i) {
      double eta = full_beta.dot(data[i]->x());
      uint y = data[i]->y();
      double F1 = y == M ? 1.0 : link_inv(compute_delta(y + 1, delta, M) - eta);
      double F0 = y == 0 ? 0.0 : link_inv(compute_delta(y, delta, M) - eta);
      ans += log(F1 - F0);
    }
    return ans;
  }

  GlmCoefs &OCM::coef() { return ParamPolicy::prm1_ref(); }
  const GlmCoefs &OCM::coef() const { return ParamPolicy::prm1_ref(); }
  Ptr<GlmCoefs> OCM::coef_prm() { return ParamPolicy::prm1(); }
  const Ptr<GlmCoefs> OCM::coef_prm() const { return ParamPolicy::prm1(); }

  Ptr<VectorParams> OCM::Delta_prm() { return ParamPolicy::prm2(); }
  const Ptr<VectorParams> OCM::Delta_prm() const { return ParamPolicy::prm2(); }

  const Vector &OCM::delta() const { return Delta_prm()->value(); }
  double OCM::delta(uint m) const {
    return compute_delta(m, delta(), maxscore());
  }

  void OCM::set_delta(const Vector &d) { Delta_prm()->set(d); }

  uint OCM::maxscore() const { return delta().size() + 1; }

  Ptr<OrdinalRegressionData> OCM::sim(RNG &rng) {
    if (!simulation_key_) {
      simulation_key_ = new FixedSizeIntCatKey(maxscore() + 1);
    }

    Vector x(xdim());
    x[0] = 1;
    for (int i = 1; i < x.size(); ++i) {
      x[i] = rnorm_mt(rng);
    }

    double eta = predict(x) + simulate_latent_variable(rng);
    int y = maxscore();
    for (uint m = 0; m < maxscore(); ++m) {
      if (eta < delta(m)) {
        y = m;
        break;
      }
    }
    NEW(OrdinalData, yp)(y, simulation_key_);
    NEW(OrdinalRegressionData, ans)(yp, new VectorData(x));
    return ans;
  }

  Vector fix_bad_delta(const Vector &beta, const Vector &delta);

  Vector fix_bad_delta(const Vector &beta, const Vector &delta) {
    //    assert(delta.lo() == 2);
    Vector gbeta(beta.size(), 0.0);
    Vector gdelta(delta.size(), 0.0);
    double lim = 0.0;
    if (delta.front() <= 0) gdelta.front() = -delta.front();
    for (uint i = 1; i < delta.size(); ++i) {
      if (delta[i] >= lim)
        gdelta[i] = gdelta[i - 1];
      else
        gdelta[i] = 1 + gdelta[i - 1] + (lim - delta[i]);
      lim = std::max(lim, delta[i]);
    }
    return concat(gbeta, gdelta);
  }

  double OCM::bd_loglike(Vector &gbeta, Vector &gdelta, Matrix &Hbeta,
                         Matrix &Hdelta, Matrix &Hbd, uint nd, bool b_derivs,
                         bool d_derivs) const {
    Vector beta(this->included_coefficients());
    const Vector &delta(this->delta());
    double ans = 0;
    if (b_derivs) {
      if (nd > 0) {
        gbeta.resize(beta.size());
        gbeta = 0.0;
        if (nd > 1) {
          Hbeta.resize(beta.size(), beta.size());
          Hbeta = 0.0;
        }
      }
    }
    if (d_derivs) {
      if (nd > 0) {
        gdelta = 0.0;
        if (nd > 1) Hdelta = 0.0;
      }
    }
    if (d_derivs && b_derivs && nd > 1) {
      Hbd = 0.0;
    }

    //------ local class -----------
    class vdelta {  // virtual wrapper for delta
      const Vector &d;
      uint mscr;

     public:
      explicit vdelta(const Vector &Delta)
          : d(Delta), mscr(d.size() + 1) {}

      double operator()(uint i) {
        if (i <= 0)
          return BOOM::negative_infinity();
        else if (i == 1)
          return 0.0;
        else if (i <= mscr)
          return d[i - 2];
        return BOOM::infinity();
      }
    };  //------- end of vdelta local class ------

    const DatasetType &v(dat());
    uint M = maxscore();
    vdelta Delta(delta);
    for (uint i = 0; i < v.size(); ++i) {
      const uint &y(v[i]->y());
      Vector x = coef().inc().select(v[i]->x());

      // the current model params are used to select the variables in
      // x() which are to be included.  The selection assures that x
      // and the function argument beta are of compatible dimension.

      double btx = beta.dot(x);
      double d1 = y == M ? 0 : Delta(y + 1) - btx;
      double d0 = y == 0 ? 0 : Delta(y) - btx;
      double F1 = y == M ? 1.0 : link_inv(d1);
      double F0 = y == 0 ? 0 : link_inv(d0);
      double prob = F1 - F0;
      ans += log(prob);
      if (nd > 0) {
        double f1 = y == M ? 0 : dlink_inv(d1) / prob;
        double f0 = y == 0 ? 0 : dlink_inv(d0) / prob;
        double df = f1 - f0;

        if (b_derivs) gbeta = -df * x;
        if (d_derivs) {
          gdelta = 0;
          if (y < M && y > 0) gdelta[y + 1] = f1;
          if (y >= 2) gdelta[y] = -f0;
        }
      }
    }
    return ans;
  }

  double OCM::Loglike(const Vector &beta_delta, Vector &g, Matrix &h,
                      uint nd) const {
    // model is parameterized so that Pr(y = m) = F(delta(m + 1)|eta) -
    // F(delta(m)|eta) if you draw a picture of F with cutpoints
    // delta, the area corresponding to the event Y=m lies to the
    // RIGHT of delta(m)

    int beta_dim = inc().nvars();
    Vector beta(ConstVectorView(beta_delta, 0, beta_dim));
    Vector delta(ConstVectorView(beta_delta, beta_dim));

    Vector gbeta, gdelta;
    Matrix Hbeta, Hdelta, Hbd;
    if (nd > 0) {
      gbeta = beta;
      gdelta = delta;
      if (nd > 1) {
        Hbeta = Matrix(beta.size(), beta.size());
        Hdelta = Matrix(delta.size(), delta.size());
        Hbd = Matrix(beta.size(), delta.size());
      }
    }
    double ans =
        bd_loglike(gbeta, gdelta, Hbeta, Hdelta, Hbd, nd, nd > 0, nd > 0);
    if (nd > 0) {
      g = concat(gbeta, gdelta);
      if (nd > 1) h = unpartition(Hbeta, Hbd, Hdelta);
    }
    return ans;
  }

  //======================================================================
  Vector OCM::CDF(const Vector &x) const {
    double eta = predict(x);
    Vector ans(maxscore() + 1);
    ans[0] = link_inv(-eta);
    for (uint i = 1; i < maxscore(); ++i) {
      ans[i] = link_inv(delta(i + 1) - eta);
    }
    ans[maxscore()] = 1;
    return ans;
  }

  void OCM::initialize_params() { mle(); }
  void OCM::initialize_params(const Vector &counts) {
    Vector hist = counts;
    hist.normalize_prob();
    double sum = hist[0];
    double b0 = qlogis(sum);
    uint I = 0;  // chaged from I=2 when adopting zero index vectors
    Vector Delta(delta());
    for (uint i = 1; i < maxscore(); ++i) {
      sum += hist[i];
      Delta(I++) = link_inv(sum - b0);
    }
    set_delta(Delta);
    Vector b(Beta());
    b = 0;
    b[0] = b0;
    set_Beta(b);
  }

  namespace {
    typedef OrdinalCutpointBetaLogLikelihood OCBLL;
    typedef OrdinalCutpointDeltaLogLikelihood OCDLL;
  }  // namespace

  OCDLL OCM::delta_log_likelihood() const { return OCDLL(this); }

  OCBLL OCM::beta_log_likelihood() const { return OCBLL(this); }

  OCBLL::OrdinalCutpointBetaLogLikelihood(const OCM *m) : m_(m) {}

  double OCBLL::operator()(const Vector &beta) const {
    const Vector &delta(m_->delta());
    return m_->log_likelihood(beta, delta);
  }

  OCDLL::OrdinalCutpointDeltaLogLikelihood(const OCM *m) : m_(m) {}

  double OCDLL::operator()(const Vector &delta) const {
    bool ok = m_->check_delta(delta);
    if (!ok) return BOOM::negative_infinity();
    const Vector &beta(m_->Beta());
    return m_->log_likelihood(beta, delta);
  }

}  // namespace BOOM
