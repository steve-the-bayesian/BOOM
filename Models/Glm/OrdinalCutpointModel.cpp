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

  namespace {
    // Some locally used free functions.
    
    // An observed value y implies a latent value z between
    // compute_lower_cutpoint(y) and compute_lower_cutpoint(y).
    inline double compute_upper_cutpoint(
        int y, const Vector &cutpoint_vector) {
      if (y < 0) {
        return negative_infinity();
      } else if (y == 0) {
        return 0;
      } else if (y <= cutpoint_vector.size()) {
        return cutpoint_vector[y - 1];
      } else {
        return infinity();
      }
    }

    inline double compute_lower_cutpoint(
        int y, const Vector &cutpoint_vector) {
      if (y <= 0) {
        return negative_infinity();
      } else if (y == 1) {
        return 0;
      } else if (y - 1 <= cutpoint_vector.size()) {
        return cutpoint_vector[y - 2];
      } else {
        return infinity();
      }
    }
    
    inline Vector make_default_cutpoint_vector(uint nlevels) {
      if (nlevels <= 2) return Vector();
      Vector cutpoints(nlevels - 2);
      for (int i = 0; i < cutpoints.size(); ++i) cutpoints[i] = i + 1;
      return cutpoints;
    }
  }  // namespace
  

  typedef OrdinalCutpointModel OCM;

  OCM::OrdinalCutpointModel(int xdim, int nlevels)
      : ParamPolicy(new GlmCoefs(xdim),
                    new VectorParams(make_default_cutpoint_vector(nlevels)))
  {
    if (xdim < 1) {
      report_error("Predictor dimension must be at least 1.");
    }
  }

  OCM::OrdinalCutpointModel(const Vector &b, const Vector &d)
      : ParamPolicy(new GlmCoefs(b), new VectorParams(d)) {
    check_cutpoints(d);
  }

  OCM::OrdinalCutpointModel(const Vector &b, const Selector &Inc,
                            const Vector &d)
      : ParamPolicy(new GlmCoefs(b, Inc), new VectorParams(d)) {
    check_cutpoints(d);
  }

  OCM::OrdinalCutpointModel(const Selector &Inc, uint nlevels)
      : ParamPolicy(new GlmCoefs(Vector(Inc.nvars(), 0.0), Inc),
                    new VectorParams(make_default_cutpoint_vector(nlevels))) {}

  OCM::OrdinalCutpointModel(const Matrix &X, const Vector &Y)
      : ParamPolicy(new GlmCoefs(X.ncol()),
                    new VectorParams(make_default_cutpoint_vector(
                        lround(max(Y)) - 1))) {
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
        NumOptModel(rhs),
        MixtureComponent(rhs)
  {}

  double OCM::pdf(const Data *dp, bool logscale) const {
    const OrdinalRegressionData *data_point =
        dynamic_cast<const OrdinalRegressionData*>(dp);
    return pdf(data_point->y(), data_point->x(), logscale);
  }

  double OCM::pdf(const Ptr<OrdinalRegressionData> &dpo, bool logscale) const {
    return pdf(dpo->y(), dpo->x(), logscale);
  }

  double OCM::pdf(const OrdinalData &Y, const Vector &X, bool logscale) const {
    uint y = Y.value();
    return pdf(y, X, logscale);
  }

  double OCM::pdf(uint y, const Vector &X, bool logscale) const {
    if (y >= nlevels()) {
      report_error("ordinal data out of bounds in OrdinalCutpointModel::pdf");
    }
    double btx = predict(X);  // X may or may not contain intercept
    double F1 = link_inv(upper_cutpoint(y) - btx);
    double F0 = link_inv(lower_cutpoint(y) - btx);
    double ans = F1 - F0;
    return logscale ? log(ans) : ans;
  }

  bool OCM::check_cutpoints(const Vector &d) const {
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
                             const Vector &cutpoints) const {
    const std::vector<Ptr<OrdinalRegressionData> > &data(dat());
    int n = data.size();
    double ans = 0;
    for (int i = 0; i < n; ++i) {
      double eta = full_beta.dot(data[i]->x());
      uint y = data[i]->y();
      double F1 = link_inv(compute_upper_cutpoint(y, cutpoints) - eta);
      double F0 = link_inv(compute_lower_cutpoint(y, cutpoints) - eta);
      ans += log(F1 - F0);
    }
    return ans;
  }

  GlmCoefs &OCM::coef() { return ParamPolicy::prm1_ref(); }
  const GlmCoefs &OCM::coef() const { return ParamPolicy::prm1_ref(); }
  Ptr<GlmCoefs> OCM::coef_prm() { return ParamPolicy::prm1(); }
  const Ptr<GlmCoefs> OCM::coef_prm() const { return ParamPolicy::prm1(); }

  Ptr<VectorParams> OCM::Cutpoints_prm() { return ParamPolicy::prm2(); }
  const Ptr<VectorParams> OCM::Cutpoints_prm() const {
    return ParamPolicy::prm2();
  }

  void OCM::set_cutpoint_vector(const Vector &cutpoints) {
    check_cutpoints(cutpoints);
    Cutpoints_prm()->set(cutpoints);
  }

  void OCM::set_cutpoint(int index, double value) {
    Cutpoints_prm()->set_element(value, index);
  }
  
  double OCM::upper_cutpoint(int y) const {
    return compute_upper_cutpoint(y, cutpoint_vector());
  }

  double OCM::lower_cutpoint(int y) const {
    return compute_lower_cutpoint(y, cutpoint_vector());
  }
  
  Ptr<OrdinalRegressionData> OCM::sim(RNG &rng) {
    if (!simulation_key_) {
      simulation_key_ = new FixedSizeIntCatKey(nlevels());
    }

    Vector x(xdim());
    x[0] = 1;
    for (int i = 1; i < x.size(); ++i) {
      x[i] = rnorm_mt(rng);
    }

    double eta = predict(x) + simulate_latent_variable(rng);
    int y = -1;
    for (uint m = 0; m < nlevels(); ++m) {
      if (eta >= (lower_cutpoint(m))) {
        y = m;
        break;
      }
    }
    if (y == -1) {
      report_error("Simulation error in OrdinalCutpointModel::sim().");
    }
    NEW(OrdinalData, yp)(y, simulation_key_);
    NEW(OrdinalRegressionData, ans)(yp, new VectorData(x));
    return ans;
  }

  
  //---------------------------------------------------------------------------
  double OCM::full_loglike(const Vector &beta,
                           const Vector &cutpoints, 
                           Vector &beta_gradient,
                           Vector &cutpoint_gradient,
                           Matrix &beta_Hessian,
                           Matrix &cutpoint_Hessian,
                           Matrix &cross_Hessian,
                           uint nderiv,
                           bool use_beta_derivs,
                           bool use_cutpoint_derivs) const {
    double ans = 0;
    const Selector &inclusion(coef().inc());
    int nvars = inclusion.nvars();
    Vector included_beta = inclusion.select(beta);

    //--------------------------------------------------
    // Resize and initialize derivatives.
    if (use_beta_derivs && nderiv > 0) {
      
      beta_gradient.resize(nvars);
      beta_gradient = 0.0;
      if (nderiv > 1) {
        beta_Hessian.resize(nvars, nvars);
        beta_Hessian = 0.0;
      }
    }
    if (use_cutpoint_derivs && nderiv > 0) {
      int ncutpoints = cutpoints.size();
      cutpoint_gradient.resize(ncutpoints);
      cutpoint_gradient = 0.0;
      if (nderiv > 1) {
        cutpoint_Hessian.resize(ncutpoints, ncutpoints);
        cutpoint_Hessian = 0.0;
      }
    }
    if (use_cutpoint_derivs && use_beta_derivs && nderiv > 1) {
      cross_Hessian.resize(nvars, cutpoints.size());
      cross_Hessian = 0.0;
    }

    const std::vector<Ptr<OrdinalRegressionData>> &data(dat());
    for (uint i = 0; i < data.size(); ++i) {
      const OrdinalRegressionData &data_point(*data[i]);
      int y = data_point.y();
      Vector x = inclusion.select(data_point.x());
      
      double btx = included_beta.dot(x);
      // d1 and d0 are the upper and lower quantiles of the standardized
      // distribution corresponding to y.
      double d1 = compute_upper_cutpoint(y, cutpoints) - btx;
      double d0 = compute_lower_cutpoint(y, cutpoints) - btx;

      // Upper and lower function values at the shifted cutpoints.
      double F1 = link_inv(d1);
      double F0 = link_inv(d0);

      double prob = F1 - F0;
      ans += log(prob);

      if (nderiv > 0) {
        // Handle first derivatives.
        
        // There are two "fake" cutpoints at -infinity and zero before the
        // first cutpoint in the vector.  Thus 'upper' and 'lower' are
        // determined by:
        int upper_cutpoint_index = y + 1 - 2;
        int lower_cutpoint_index = y - 2;
        
        // Derivatives of F1 and F0 with respect to their function arguments at
        // the shifted cutpoints.
        double f1 = dlink_inv(d1) / prob;
        double f0 = dlink_inv(d0) / prob;
        double df = f1 - f0;

        if (use_beta_derivs) {
          // The chain rule introduces an extra factor of (-x).
          beta_gradient -= df * x;
        }
        
        if (use_cutpoint_derivs) {
          // The chain rule multiplies f1 by 1 if the upper cutpoint is in the
          // cutpoint vector, and 0 otherwise.  The increment goes in the slot
          // corresponding to the upper cutpoint.  The chain rule multiplies f0
          // by 1 if the lower cutpoint is in the cutpoint vector, and 0
          // otherwise.  
          // 
          // Another way to say all this is that d logp / d cutpoints = f1 *
          // e(y+1) - f0 * e(y), where e(y) is a vector of all 0's except a 1 in
          // slot y - 2.  If y < 2 then e(y) is a vector of all 0's.
          if (upper_cutpoint_index >= 0
              && upper_cutpoint_index < cutpoints.size()) {
            cutpoint_gradient[upper_cutpoint_index] += f1;
          }
          if (lower_cutpoint_index >= 0
              && lower_cutpoint_index < cutpoint_gradient.size()) {
            cutpoint_gradient[lower_cutpoint_index] -= f0;
          }
        }

        if (nderiv > 1) {
          // Handle second derivatives
          
          // The beta Hessian is the derivative of the beta gradient.
          // Derivatives by the quotient rule.
          double df1 = ddlink_inv(d1) / prob;
          double df0 = ddlink_inv(d0) / prob;

          double quotient_rule_coefficient = (df1 - df0) - square(f1 - f0);

          if (use_beta_derivs) {
            // [(F1 - F0) * (df1 - df0) * x * x'] - [(f1 - f0)^2 * x * x']
            //  --------------------------------------------------
            //       (F1 - F0)^2
            //
            // note that fx and dfx are already divided by (F1 - F0), so that
            // factor disappears.
            beta_Hessian.add_outer(x, x, quotient_rule_coefficient);
          }

          bool has_upper = upper_cutpoint_index >= 0
              && upper_cutpoint_index < cutpoints.size();
          bool has_lower = lower_cutpoint_index >= 0
              && lower_cutpoint_index < cutpoints.size();

          if (use_cutpoint_derivs) {
            if (has_upper) {
              cutpoint_Hessian(upper_cutpoint_index, upper_cutpoint_index) +=
                  df1 - square(f1);
            }
            if (has_lower) {
              cutpoint_Hessian(lower_cutpoint_index, lower_cutpoint_index) -=
                  df0 + square(f0);
            }
            if (has_upper && has_lower) {
              cutpoint_Hessian(lower_cutpoint_index, upper_cutpoint_index)
                  += f0 * f1;
              cutpoint_Hessian(upper_cutpoint_index, lower_cutpoint_index)
                  += f0 * f1;
            }
          }

          if (use_cutpoint_derivs && use_beta_derivs) {
            // Cross hessian has beta along rows, and cutpoints along columns.
            // It is not symmetric.  It occurs along with its transpose in the
            // full Hessian.
            if (has_upper) {
              cross_Hessian.col(upper_cutpoint_index).axpy(
                  x, -df1 + f1 * (f1 - f0));
            }
            if (has_lower) {
              cross_Hessian.col(lower_cutpoint_index).axpy(
                  x, df0 - f0 * (f1 - f0));
            }
          }
        } // nderiv > 1
      }  // nderiv > 0
    }  // for 
    return ans;
  }
  
  //---------------------------------------------------------------------------
  double OCM::Loglike(const Vector &beta_delta, Vector &g, Matrix &Hessian,
                      uint nderiv) const {
    // model is parameterized so that Pr(y = m) = F(delta(m + 1)|eta) -
    // F(delta(m)|eta) if you draw a picture of F with cutpoints
    // delta, the area corresponding to the event Y=m lies to the
    // RIGHT of delta(m)

    int beta_dim = inc().nvars();
    Vector beta(ConstVectorView(beta_delta, 0, beta_dim));
    Vector cutpoints(ConstVectorView(beta_delta, beta_dim));

    Vector beta_gradient, cutpoint_gradient;
    Matrix beta_Hessian, cutpoint_Hessian, cross_Hessian;
    if (nderiv > 0) {
      beta_gradient.resize(beta_dim);
      cutpoint_gradient.resize(cutpoints.size());
      if (nderiv > 1) {
        beta_Hessian.resize(beta.size(), beta.size());
        cutpoint_Hessian.resize(cutpoints.size(), cutpoints.size());
        cross_Hessian.resize(beta.size(), cutpoints.size());
      }
    }
    double ans = full_loglike(
        beta, cutpoints, beta_gradient, cutpoint_gradient, beta_Hessian,
        cutpoint_Hessian, cross_Hessian, nderiv, nderiv > 0, nderiv > 0);

    if (nderiv > 0) {
      g = concat(beta_gradient, cutpoint_gradient);
      if (nderiv > 1) {
        Hessian = unpartition(
            beta_Hessian, cross_Hessian, cutpoint_Hessian);
      }
    }
    return ans;
  }

  //======================================================================
  Vector OCM::CDF(const Vector &x) const {
    double eta = predict(x);
    Vector ans(nlevels());
    ans[0] = link_inv(-eta);
    for (uint i = 1; i < nlevels() - 1; ++i) {
      ans[i] = link_inv(upper_cutpoint(i) - eta);
    }
    ans[nlevels() - 1] = 1;
    return ans;
  }

  void OCM::initialize_params() {
    if (dat().size() > 0) {
      mle();
    } else {
      initialize_params(Vector(nlevels(), 0.0));
    }
  }
  
  void OCM::initialize_params(const Vector &counts) {
    if (counts.size() != nlevels()) {
      report_error("Vector of counts did not align with the number of "
                   "factor levels.");
    }
      
    Vector hist = counts;
    // Add a unit information prior to ensure each element is positive.
    hist += 1.0 / hist.size();
    
    hist.normalize_prob();
    double sum = hist[0];
    double b0 = link(sum);
    
    uint I = 0;  // chaged from I=2 when adopting zero index vectors
    Vector cutpoint_vector = this->cutpoint_vector();
    for (uint i = 1; i < cutpoint_vector.size(); ++i) {
      sum += hist[i];
      cutpoint_vector(I++) = link(sum) - b0;
    }
    set_cutpoint_vector(cutpoint_vector);
    Vector b(Beta());
    b = 0;
    b[0] = b0;
    set_Beta(b);
  }

  namespace {
    typedef OrdinalCutpointBetaLogLikelihood OCBLL;
    typedef OrdinalCutpointLogLikelihood OCLL;
  }  // namespace

  OCLL OCM::cutpoint_log_likelihood() const { return OCLL(this); }

  OCBLL OCM::beta_log_likelihood() const { return OCBLL(this); }

  OCBLL::OrdinalCutpointBetaLogLikelihood(const OCM *m) : m_(m) {}

  double OCBLL::operator()(const Vector &beta) const {
    return m_->log_likelihood(beta, m_->cutpoint_vector());
  }

  OCLL::OrdinalCutpointLogLikelihood(const OCM *m) : m_(m) {}

  double OCLL::operator()(const Vector &delta) const {
    bool ok = m_->check_cutpoints(delta);
    if (!ok) return BOOM::negative_infinity();
    const Vector &beta(m_->Beta());
    return m_->log_likelihood(beta, delta);
  }

  double OrdinalLogitModel::ddlink_inv(double eta) const {
    double exp_minus_eta = exp(-eta);
    if (std::isfinite(exp_minus_eta)) {
      double F = link_inv(eta);
      double f = dlink_inv(eta);
      return exp_minus_eta * F * (2 * f - F);
    } else {
      // if exp_minus_eta is not finite then eta must be negative.  Various
      // arguments (e.g. L'Hoptial's rule) show the answer in this case to be
      // zero.
      return 0;
    }
  }

  double OrdinalProbitModel::ddlink_inv(double eta) const {
    if (std::isfinite(eta)) {
      return -eta * dnorm(eta);
    } else {
      return 0;
    }
  }
  
}  // namespace BOOM
