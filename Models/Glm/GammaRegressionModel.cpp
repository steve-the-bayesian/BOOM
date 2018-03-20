// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#include "Models/Glm/GammaRegressionModel.hpp"
#include "distributions.hpp"

#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {

  GammaRegressionModelBase::GammaRegressionModelBase(int xdim)
      : ParamPolicy(new UnivParams(1.0), new GlmCoefs(xdim)) {}

  GammaRegressionModelBase::GammaRegressionModelBase(double shape_parameter,
                                                     const Vector &coefficients)
      : ParamPolicy(new UnivParams(shape_parameter),
                    new GlmCoefs(coefficients)) {}

  GammaRegressionModelBase::GammaRegressionModelBase(
      const Ptr<UnivParams> &alpha, const Ptr<GlmCoefs> &coefficients)
      : ParamPolicy(alpha, coefficients) {}

  void GammaRegressionModelBase::set_shape_parameter(double alpha) {
    shape_prm()->set(alpha);
  }

  namespace {
    template <class VECTOR>
    double gamma_expected_value(const VECTOR &x, const GlmCoefs &beta) {
      return exp(beta.predict(x));
    }
  }  // namespace

  double GammaRegressionModelBase::expected_value(const Vector &x) const {
    return gamma_expected_value(x, coef());
  }
  double GammaRegressionModelBase::expected_value(const VectorView &x) const {
    return gamma_expected_value(x, coef());
  }
  double GammaRegressionModelBase::expected_value(
      const ConstVectorView &x) const {
    return gamma_expected_value(x, coef());
  }

  double GammaRegressionModelBase::pdf(const Data *dp, bool logscale) const {
    const RegressionData *data = dynamic_cast<const RegressionData *>(dp);
    double eta = coef().predict(data->x());
    double alpha = shape_parameter();
    return dgamma(data->y(), alpha, alpha / exp(eta), logscale);
  }

  double GammaRegressionModelBase::sim(const Vector &x, RNG &rng) const {
    double a = shape_parameter();
    double mu = expected_value(x);
    return rgamma_mt(rng, a, a / mu);
  }

  //======================================================================

  GammaRegressionModel::GammaRegressionModel(int xdim)
      : GammaRegressionModelBase(xdim) {}
  GammaRegressionModel::GammaRegressionModel(double shape_parameter,
                                             const Vector &coefficients)
      : GammaRegressionModelBase(shape_parameter, coefficients) {}
  GammaRegressionModel::GammaRegressionModel(const Ptr<UnivParams> &alpha,
                                             const Ptr<GlmCoefs> &coefficients)
      : GammaRegressionModelBase(alpha, coefficients) {}

  GammaRegressionModel *GammaRegressionModel::clone() const {
    return new GammaRegressionModel(*this);
  }

  namespace {
    void initialize_log_likelihood_computation(const Vector &alpha_beta,
                                               Vector &gradient,
                                               Matrix &Hessian, uint nd,
                                               double &digamma_alpha,
                                               double &trigamma_alpha) {
      double alpha = alpha_beta[0];
      if (nd > 0) {
        gradient.resize(alpha_beta.size());
        gradient = 0;
        digamma_alpha = digamma(alpha);
        if (nd > 1) {
          Hessian.resize(alpha_beta.size(), alpha_beta.size());
          Hessian = 0;
          trigamma_alpha = trigamma(alpha);
        }
      }
    }

    double increment_loglike(Vector &gradient, Matrix &Hessian, int nd,
                             const ConstVectorView &x, double sumy,
                             double sumlogy, double n, double eta, double mu,
                             double alpha, double log_alpha,
                             double lgamma_alpha, double digamma_alpha,
                             double trigamma_alpha) {
      double ans = 0;
      if (mu <= 0) {
        // mu = exp(eta), so it cannot mathematically be less than zero.
        ans = negative_infinity();
        // We should adjust the gradient and hessian here, if nd > 0.  However,
        // we can't really do that without knowing beta.
      }
      if (alpha <= 0) {
        ans = negative_infinity();
        if (nd > 0) {
          gradient[0] = -alpha;
          if (nd > 1) {
            Hessian = 0.0;
            Hessian.diag() = 1.0;
          }
        }
      }
      if (!std::isfinite(ans)) {
        return ans;
      }
      ans = n * (alpha * (log_alpha - eta) - lgamma_alpha) +
            (alpha - 1) * sumlogy - alpha * sumy / mu;
      if (nd > 0) {
        // The derivatives of log likelihood are with respect to its shape
        // parameter alpha, and the vector of regression coefficients beta, with
        // alpha in front.
        gradient[0] += n * (1 + log_alpha) - n * eta - n * digamma_alpha +
                       sumlogy - sumy / mu;
        VectorView(gradient, 1).axpy(x, alpha * ((sumy / mu) - n));
        if (nd > 1) {
          Hessian(0, 0) += n * ((1.0 / alpha) - trigamma_alpha);
          VectorView(Hessian.row(0), 1).axpy(x, -n + sumy / mu);
          // The cross hessian affects both rows and columns.  Only
          // the first row is handled here.  Need to reflect the
          // matrix prior to returning.
          SpdMatrix beta_hessian = outer(x);
          beta_hessian *= -alpha * sumy / mu;
          SubMatrix(Hessian, 1, x.size(), 1, x.size()) += beta_hessian;
        }
      }
      return ans;
    }
  }  // namespace

  // The likelihood for a gamma(a, b) observation is
  // (b^a / Gamma (a)) y^{a-1} exp{-b * y}.
  // Reparameterizing to b = a/mu gives...
  //
  // L = [a^a / (mu^a Gamma(a))] y^{a-1} exp{- a * y / mu}
  //  So on the log scale this gives
  //
  //  \ell = a * log(a) - a * log(mu) - lgamma(a) + (a-1) * log(y) - a * y/mu
  double GammaRegressionModel::Loglike(const Vector &alpha_beta,
                                       Vector &gradient, Matrix &Hessian,
                                       uint nd) const {
    double ans = 0;
    const std::vector<Ptr<RegressionData> > &data(dat());
    double alpha = alpha_beta[0];
    ConstVectorView beta(alpha_beta, 1);
    bool all_coefficients_included = (xdim() == beta.size());
    const Selector &inc(coef().inc());
    double log_alpha = log(alpha);
    double lgamma_alpha = lgamma(alpha);
    double digamma_alpha, trigamma_alpha;
    initialize_log_likelihood_computation(alpha_beta, gradient, Hessian, nd,
                                          digamma_alpha, trigamma_alpha);
    for (size_t i = 0; i < data.size(); ++i) {
      const Vector &x(data[i]->x());
      Vector reduced_x;
      if (!all_coefficients_included) {
        reduced_x = inc.select(x);
      }
      ConstVectorView predictors(all_coefficients_included ? x : reduced_x);
      double eta = beta.dot(predictors);
      double mu = exp(eta);
      double y = data[i]->y();
      ans += increment_loglike(gradient, Hessian, nd, predictors, y, log(y), 1,
                               eta, mu, alpha, log_alpha, lgamma_alpha,
                               digamma_alpha, trigamma_alpha);
      if (!std::isfinite(ans)) {
        return ans;
      }
    }
    if (nd > 1) {
      Hessian.col(0) = Hessian.row(0);
    }
    return ans;
  }

  //======================================================================

  namespace {
    typedef GammaRegressionModelConditionalSuf GRMCS;
    typedef GammaRegressionConditionalSuf GCSUF;
  }  // namespace

  GCSUF::GammaRegressionConditionalSuf() : xdim_(-1), nrow_(0) {}

  GCSUF *GCSUF::clone() const { return new GCSUF(*this); }

  void GCSUF::Update(const RegressionData &data) {
    Ptr<GammaSuf> suf = get(data.Xptr());
    suf->update_raw(data.y());
  }

  void GCSUF::clear() { suf_.clear(); }

  Vector GCSUF::vectorize(bool minimal) const {
    Vector ans;
    for (const auto &el : suf_) {
      ans.concat(el.first->value());
      ans.concat(el.second->vectorize(minimal));
    }
    return ans;
  }

  Vector::const_iterator GCSUF::unvectorize(Vector::const_iterator &it,
                                            bool minimal) {
    if (nrow_ < 0 || xdim_ < 0) {
      report_error("Must call set_dimensions() before calling unvectorize().");
    }
    for (int i = 0; i < nrow_; ++i) {
      Vector v(it, it + xdim_);
      it += xdim_;
      NEW(VectorData, xdata)(v);
      NEW(GammaSuf, ysuf)();
      it = ysuf->unvectorize(it);
      suf_[xdata] = ysuf;
    }
    return it;
  }

  Vector::const_iterator GCSUF::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  std::ostream &GCSUF::print(std::ostream &out) const {
    for (const auto &el : suf_) {
      out << *el.first << " " << *el.second << std::endl;
    }
    return out;
  }

  void GCSUF::set_dimensions(int nrow, int xdim) {
    xdim_ = xdim;
    nrow_ = nrow;
  }

  void GCSUF::combine(const Ptr<GCSUF> &rhs) { combine(*rhs); }

  void GCSUF::combine(const GCSUF &rhs) {
    for (const auto &el : rhs.suf_) {
      if (!suf_[el.first]) {
        suf_[el.first->clone()] = el.second->clone();
        ++nrow_;
      } else {
        suf_[el.first]->combine(el.second);
      }
    }
  }

  GCSUF *GCSUF::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  void GCSUF::increment(double n, double sum, double sumlog,
                        const Ptr<VectorData> &predictors) {
    Ptr<GammaSuf> suf = get(predictors);
    suf->increment(n, sum, sumlog);
  }

  Ptr<GammaSuf> GCSUF::get(const Ptr<VectorData> &predictors) {
    if (xdim_ < 0) {
      xdim_ = predictors->dim();
    } else {
      if (predictors->dim() != xdim_) {
        report_error("Predictor dimension does not match.");
      }
    }
    Ptr<GammaSuf> suf = suf_[predictors];
    if (!suf) {
      suf.reset(new GammaSuf);
      suf_[predictors] = suf;
      ++nrow_;
    }
    return suf;
  }

  //======================================================================
  GRMCS::GammaRegressionModelConditionalSuf(int xdim)
      : GammaRegressionModelBase(xdim),
        DataPolicy(new GammaRegressionConditionalSuf) {}

  GRMCS::GammaRegressionModelConditionalSuf(double shape_parameter,
                                            const Vector &coefficients)
      : GammaRegressionModelBase(shape_parameter, coefficients),
        DataPolicy(new GammaRegressionConditionalSuf) {}

  GRMCS::GammaRegressionModelConditionalSuf(const Ptr<UnivParams> &alpha,
                                            const Ptr<GlmCoefs> &coefficients)
      : GammaRegressionModelBase(alpha, coefficients),
        DataPolicy(new GammaRegressionConditionalSuf) {}

  GRMCS *GRMCS::clone() const { return new GRMCS(*this); }

  double GRMCS::Loglike(const Vector &alpha_beta, Vector &gradient,
                        Matrix &Hessian, uint nd) const {
    double ans = 0;
    double alpha = alpha_beta[0];
    double log_alpha = log(alpha);
    double lgamma_alpha = lgamma(alpha);
    ConstVectorView beta(alpha_beta, 1);
    bool all_coefficients_included = (xdim() == beta.size());
    const Selector &inc(coef().inc());

    double digamma_alpha, trigamma_alpha;
    initialize_log_likelihood_computation(alpha_beta, gradient, Hessian, nd,
                                          digamma_alpha, trigamma_alpha);
    for (const auto &el : suf()->map()) {
      const Vector &x(el.first->value());
      Vector reduced_x;
      if (!all_coefficients_included) {
        reduced_x = inc.select(x);
      }
      ConstVectorView predictors(all_coefficients_included ? x : reduced_x);

      double eta = beta.dot(predictors);
      double mu = exp(eta);
      double sumy = el.second->sum();
      double sumlogy = el.second->sumlog();
      double n = el.second->n();
      ans += increment_loglike(gradient, Hessian, nd, predictors, sumy, sumlogy,
                               n, eta, mu, alpha, log_alpha, lgamma_alpha,
                               digamma_alpha, trigamma_alpha);
    }
    if (nd > 1) {
      // Reflect the first row into the first column before returning.
      Hessian.col(0) = Hessian.row(0);
    }
    return ans;
  }

  void GRMCS::increment_sufficient_statistics(
      double n, double sum, double sumlog, const Ptr<VectorData> &predictors) {
    suf()->increment(n, sum, sumlog, predictors);
  }

}  // namespace BOOM
