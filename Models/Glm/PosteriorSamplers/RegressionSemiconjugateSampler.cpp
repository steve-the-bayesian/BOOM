// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#include "Models/Glm/PosteriorSamplers/RegressionSemiconjugateSampler.hpp"
#include "distributions.hpp"
#include "numopt.hpp"

namespace BOOM {
  namespace {
    using RSS = RegressionSemiconjugateSampler;
  }

  RSS::RegressionSemiconjugateSampler(
      RegressionModel *model, const Ptr<MvnBase> &coefficient_prior,
      const Ptr<GammaModelBase> &residual_precision_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        beta_prior_(coefficient_prior),
        siginv_prior_(residual_precision_prior),
        beta_sampler_(model_, beta_prior_, seeding_rng),
        sigsq_sampler_(siginv_prior_) {}

  void RSS::set_sigma_max(double sigma_max) {
    sigsq_sampler_.set_sigma_max(sigma_max);
  }

  void RSS::draw() {
    draw_beta_given_sigma();
    draw_sigma_given_beta();
  }

  double RSS::logpri() const {
    return beta_sampler_.logpri() + sigsq_sampler_.log_prior(model_->sigsq());
  }

  void RSS::draw_beta_given_sigma() {
    beta_sampler_.sample_regression_coefficients(rng(), model_, *beta_prior_);
  }

  void RSS::draw_sigma_given_beta() {
    const RegSuf &suf(*model_->suf());
    double sigsq =
        sigsq_sampler_.draw(rng(), suf.n(), suf.relative_sse(model_->coef()));
    model_->set_sigsq(sigsq);
  }

  void RSS::find_posterior_mode(double epsilon) {
    SpdMatrix precision =
        beta_prior_->siginv() + model_->suf()->xtx() / model_->sigsq();
    Vector unscaled_mean = beta_prior_->siginv() * beta_prior_->mu() +
                           model_->suf()->xty() / model_->sigsq();
    Vector beta = precision.solve(unscaled_mean);
    double sigsq =
        model_->suf()->relative_sse(GlmCoefs(beta)) / model_->suf()->n();
    model_->set_Beta(beta);
    model_->set_sigsq(sigsq);

    Vector parameters = model_->vectorize_params();
    auto target_fun = [this](const Vector &parameters, Vector &gradient,
                             Matrix &Hessian, uint nd) {
      return this->model_->Loglike(parameters, gradient, Hessian, nd) +
             this->log_prior(parameters, gradient, Hessian, nd);
    };
    auto fun = [target_fun](const Vector &x) {
      Vector g;
      Matrix h;
      return target_fun(x, g, h, 0);
    };
    auto dfun = [target_fun](const Vector &x, Vector &g) {
      Matrix h;
      return target_fun(x, g, h, 1);
    };
    auto d2fun = [target_fun](const Vector &x, Vector &g, Matrix &H) {
      return target_fun(x, g, H, 2);
    };
    Vector gradient(parameters.size());
    Matrix Hessian(parameters.size(), parameters.size());
    double max_function_value = 0;
    std::string error_message;
    bool ok = max_nd2_careful(parameters, gradient, Hessian, max_function_value,
                              fun, dfun, d2fun, 1e-5, error_message);
    if (ok) {
      model_->unvectorize_params(parameters);
    } else {
      ostringstream err;
      err << "An exception was thrown while locating the posterior mode in "
             "RegressionSemiconjugateSampler."
          << std::endl
          << "The error message was: " << std::endl
          << error_message;
      report_error(err.str());
    }
  }

  double RSS::log_prior_density(const ConstVectorView &parameters) const {
    double sigsq = parameters.back();
    const ConstVectorView beta(parameters, 0, parameters.size() - 1);
    double ans = beta_prior_->logp(beta);
    ans += siginv_prior_->logp_reciprocal(sigsq);
    return ans;
  }

  double RSS::increment_log_prior_gradient(const ConstVectorView &parameters,
                                           VectorView gradient) const {
    Vector beta(parameters);
    double sigsq = beta.back();
    beta.pop_back();
    Vector beta_gradient(beta.size());
    double sigsq_deriv;
    double ans = beta_prior_->dlogp(beta, beta_gradient) +
                 siginv_prior_->logp_reciprocal(sigsq, &sigsq_deriv);
    gradient = concat(beta_gradient, sigsq_deriv);
    return ans;
  }

  double RSS::log_prior(const Vector &parameters, Vector &gradient,
                        Matrix &Hessian, uint nd) const {
    Vector beta(parameters);
    double sigsq = beta.back();
    beta.pop_back();
    Vector beta_gradient(beta.size());
    Matrix beta_hessian(beta.size(), beta.size());

    double sigsq_derivative;
    double sigsq_second_derivative;
    double ans = beta_prior_->Logp(beta, beta_gradient, beta_hessian, nd) +
                 siginv_prior_->logp_reciprocal(
                     sigsq, nd > 0 ? &sigsq_derivative : nullptr,
                     nd > 1 ? &sigsq_second_derivative : nullptr);

    if (nd > 0) {
      gradient = concat(beta_gradient, sigsq_derivative);
      if (nd > 1) {
        // NOTE: There is a pretty strong assumption here that the prior models
        // do not share parameters.  If beta_prior_ is of class MvnGivenSigma,
        // for example, then some cross-Hessian terms will be nonzero.
        Hessian = unpartition(beta_hessian, Vector(beta.size(), 0.0),
                              sigsq_second_derivative);
      }
    }
    return ans;
  }

}  // namespace BOOM
