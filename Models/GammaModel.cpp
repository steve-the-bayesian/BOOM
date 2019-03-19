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
#include "Models/GammaModel.hpp"
#include <cmath>
#include <limits>
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  GammaSuf::GammaSuf() : sum_(0), sumlog_(0), n_(0) {}

  GammaSuf *GammaSuf::clone() const { return new GammaSuf(*this); }

  void GammaSuf::set(double sum, double sumlog, double n) {
    // Check for impossible values.
    if (n > 0) {
      if (sum <= 0.0) {
        report_error(
            "GammaSuf cannot have a negative sum if "
            "it has a positive sample size");
      }
      // There is no minimum value that sumlog can achieve, because
      // any individual observation might be arbitrarily close to
      // zero, driving the sum of logs close to negative infinity.
      //
      // The sum of logs is maximized if each observation is the same
      // size.
      double ybar = sum / n;
      if (sumlog > n * log(ybar)) {
        report_error(
            "GammaSuf was set with an impossibly large value "
            "of sumlog.");
      }
    } else if (n < 0) {
      report_error("GammaSuf set to have a negative sample size.");
    } else {
      if (std::fabs(sum) > std::numeric_limits<double>::epsilon() ||
          std::fabs(sumlog) > std::numeric_limits<double>::epsilon()) {
        report_error("All elements of GammaSuf must be zero if n == 0.");
      }
    }
    sum_ = sum;
    sumlog_ = sumlog;
    n_ = n;
  }

  void GammaSuf::clear() { sum_ = sumlog_ = n_ = 0; }

  void GammaSuf::Update(const DoubleData &dat) {
    double x = dat.value();
    update_raw(x);
  }

  void GammaSuf::update_raw(double x) {
    ++n_;
    sum_ += x;
    sumlog_ += log(x);
  }

  void GammaSuf::increment(double n, double sum, double sumlog) {
    n_ += n;
    sum_ += sum;
    sumlog_ += sumlog;
  }

  void GammaSuf::add_mixture_data(double y, double prob) {
    n_ += prob;
    sum_ += prob * y;
    sumlog_ += prob * log(y);
  }

  double GammaSuf::sum() const { return sum_; }
  double GammaSuf::sumlog() const { return sumlog_; }
  double GammaSuf::n() const { return n_; }
  std::ostream &GammaSuf::display(std::ostream &out) const {
    out << "gamma::sum    = " << sum_ << endl
        << "gamma::sumlog = " << sumlog_ << endl
        << "gamma::n      = " << n_ << endl;
    return out;
  }

  void GammaSuf::combine(const Ptr<GammaSuf> &s) {
    sum_ += s->sum_;
    sumlog_ += s->sumlog_;
    n_ += s->n_;
  }

  void GammaSuf::combine(const GammaSuf &s) {
    sum_ += s.sum_;
    sumlog_ += s.sumlog_;
    n_ += s.n_;
  }

  GammaSuf *GammaSuf::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  Vector GammaSuf::vectorize(bool) const {
    Vector ans(3);
    ans[0] = sum_;
    ans[1] = sumlog_;
    ans[2] = n_;
    return ans;
  }

  Vector::const_iterator GammaSuf::unvectorize(Vector::const_iterator &v,
                                               bool) {
    sum_ = *v;
    ++v;
    sumlog_ = *v;
    ++v;
    n_ = *v;
    ++v;
    return v;
  }

  Vector::const_iterator GammaSuf::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  std::ostream &GammaSuf::print(std::ostream &out) const {
    return out << n_ << " " << sum_ << " " << sumlog_;
  }
  //======================================================================
  GammaModelBase::GammaModelBase() : DataPolicy(new GammaSuf()) {}

  double GammaModelBase::mean() const { return alpha() / beta(); }

  double GammaModelBase::variance() const { return alpha() / square(beta()); }

  double GammaModelBase::pdf(const Ptr<Data> &dp, bool logscale) const {
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double GammaModelBase::pdf(const Data *dp, bool logscale) const {
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double GammaModelBase::Logp(double x, double &g, double &h, uint nd) const {
    double a = alpha();
    double b = beta();
    double ans = dgamma(x, a, b, true);
    if (nd > 0) {
      g = (a - 1) / x - b;
    }
    if (nd > 1) {
      h = -(a - 1) / square(x);
    }
    return ans;
  }

  double GammaModelBase::sim(RNG &rng) const {
    return rgamma_mt(rng, alpha(), beta());
  }

  void GammaModelBase::add_mixture_data(const Ptr<Data> &dp, double prob) {
    double y = DAT(dp)->value();
    suf()->add_mixture_data(y, prob);
  }

  double GammaModelBase::logp_reciprocal(double sigsq, double *gradient,
                                         double *hessian) const {
    double a = alpha();
    double b = beta();
    if (a <= 0 || b <= 0 || sigsq <= 0) {
      return negative_infinity();
    }
    double log_sigsq = log(sigsq);
    double ans = dgamma(1.0 / sigsq, a, b, true) - 2 * log_sigsq;
    if (gradient != nullptr) {
      double sig4 = sigsq * sigsq;
      *gradient = -(a + 1) / sigsq + (b / sig4);
      if (hessian != nullptr) {
        *hessian = ((a + 1) / sig4) - 2 * b / (sig4 * sigsq);
      }
    }
    return ans;
  }

  //======================================================================

  GammaModel::GammaModel(double a, double b)
      : GammaModelBase(), ParamPolicy(new UnivParams(a), new UnivParams(b)) {
    if (a <= 0 || b <= 0) {
      report_error(
          "Both parameters must be positive in the "
          "GammaModel constructor.");
    }
  }

  GammaModel::GammaModel(double shape, double mean, int)
      : GammaModelBase(),
        ParamPolicy(new UnivParams(shape), new UnivParams(shape / mean)) {
    if (shape <= 0 || mean <= 0) {
      report_error(
          "Both parameters must be positive in the "
          "GammaModel constructor.");
    }
  }

  GammaModel *GammaModel::clone() const { return new GammaModel(*this); }

  Ptr<UnivParams> GammaModel::Alpha_prm() { return ParamPolicy::prm1(); }

  Ptr<UnivParams> GammaModel::Beta_prm() { return ParamPolicy::prm2(); }

  const Ptr<UnivParams> GammaModel::Alpha_prm() const {
    return ParamPolicy::prm1();
  }

  const Ptr<UnivParams> GammaModel::Beta_prm() const {
    return ParamPolicy::prm2();
  }

  double GammaModel::alpha() const { return ParamPolicy::prm1_ref().value(); }

  double GammaModel::beta() const { return ParamPolicy::prm2_ref().value(); }

  void GammaModel::set_alpha(double a) {
    if (a <= 0) {
      ostringstream err;
      err << "The 'a' parameter must be positive in GammaModel::set_alpha()."
          << endl
          << "Called with a = " << a << endl;
      report_error(err.str());
    }
    ParamPolicy::prm1_ref().set(a);
  }

  void GammaModel::set_beta(double b) {
    if (b <= 0) {
      ostringstream err;
      err << "The 'b' parameter must be positive in GammaModel::set_beta()."
          << endl
          << "Called with b = " << b << endl;
      report_error(err.str());
    }
    ParamPolicy::prm2_ref().set(b);
  }

  void GammaModel::set_shape_and_scale(double a, double b) {
    set_alpha(a);
    set_beta(b);
  }

  void GammaModel::set_shape_and_mean(double a, double mean) {
    set_shape_and_scale(a, a / mean);
  }

  void GammaModel::set_mean_and_scale(double mean, double b) {
    set_shape_and_scale(mean * b, b);
  }

  double GammaModel::mean() const { return alpha() / beta(); }

  inline double bad_gamma_loglike(double a, double b, Vector *g, Matrix *h) {
    if (g != nullptr) {
      (*g)[0] = (a <= 0) ? -(a + 1) : 0;
      (*g)[1] = (b <= 0) ? -(b + 1) : 0;
    }
    if (h != nullptr) {
      h->set_diag(-1);
    }
    return negative_infinity();
  }

  double GammaModel::Loglike(const Vector &shape_scale, Vector &gradient,
                             Matrix &hessian,
                             uint number_of_derivatives) const {
    return loglikelihood(shape_scale,
                         number_of_derivatives > 0 ? &gradient : nullptr,
                         number_of_derivatives > 1 ? &hessian : nullptr);
  }

  double GammaModel::loglikelihood(double shape, double scale) const {
    return loglikelihood({shape, scale}, nullptr, nullptr);
  }

  double GammaModel::loglikelihood(const Vector &ab, Vector *gradient,
                                   Matrix *hessian) const {
    if (ab.size() != 2) {
      report_error("GammaModel::loglikelihood expects an argument of length 2");
    }
    double a = ab[0];
    double b = ab[1];
    if (a <= 0 || b <= 0) {
      return bad_gamma_loglike(a, b, gradient, hessian);
    }

    double n = suf()->n();
    double sum = suf()->sum();
    double sumlog = suf()->sumlog();
    double logb = log(b);
    double ans = n * (a * logb - lgamma(a)) + (a - 1) * sumlog - b * sum;

    if (gradient != nullptr) {
      if (gradient->size() != 2) {
        report_error(
            "GammaModel::loglikelihood expects a gradient vector "
            "of length 2");
      }
      (*gradient)[0] = n * (logb - digamma(a)) + sumlog;
      (*gradient)[1] = n * a / b - sum;

      if (hessian != nullptr) {
        if (hessian->nrow() != 2 || hessian->ncol() != 2) {
          report_error(
              "GammaModel::loglikelihood expects a 2 x 2 "
              "Hessian matrix");
        }
        (*hessian)(0, 0) = -n * trigamma(a);
        (*hessian)(1, 0) = (*hessian)(0, 1) = n / b;
        (*hessian)(1, 1) = -n * a / square(b);
      }
    }
    return ans;
  }

  void GammaModel::mle() {
    // Good starting values for the MLE are available from the method of
    // moments.
    double n = suf()->n();
    double sum = suf()->sum();
    double sumlog = suf()->sumlog();

    double ybar = n > 0 ? sum / n : 0;  // arithmetic mean
    double geometric_mean = exp(n > 0 ? sumlog / n : 0);
    double sum_of_squares = 0;
    for (uint i = 0; i < dat().size(); ++i) {
      sum_of_squares += pow(dat()[i]->value() - ybar, 2);
    }
    if ((sum_of_squares > 0) && (n > 1)) {
      double sample_variance = sum_of_squares / (n - 1);

      // method of moments estimates
      double b = ybar / sample_variance;

      // one step newton refinement:
      // a = ybar * b;
      // b - exp(psi(ybar*b)) / geometric_mean = 0
      double tmp = exp(digamma(ybar * b)) / geometric_mean;
      double f = b - tmp;
      double g = 1 - tmp * trigamma(ybar * b) * ybar;

      b -= f / g;
      set_shape_and_scale(ybar * b, b);
    }
    NumOptModel::mle();
  }

}  // namespace BOOM
