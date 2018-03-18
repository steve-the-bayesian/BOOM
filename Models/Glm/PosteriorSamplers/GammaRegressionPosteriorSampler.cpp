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

#include "Models/Glm/PosteriorSamplers/GammaRegressionPosteriorSampler.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "Samplers/MH_Proposals.hpp"
#include "TargetFun/TargetFun.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "numopt.hpp"

namespace BOOM {
  namespace {
    typedef GammaRegressionPosteriorSampler GRPS;

    class GammaRegressionLogPosterior {
     public:
      explicit GammaRegressionLogPosterior(
          const GammaRegressionPosteriorSampler *sampler)
          : sampler_(sampler) {}

      double operator()(const Vector &theta) const {
        return sampler_->log_posterior(theta, gradient_, Hessian_, 0);
      }

      double operator()(const Vector &theta, Vector &gradient) const {
        return sampler_->log_posterior(theta, gradient, Hessian_, 1);
      }

      double operator()(const Vector &theta, Vector &gradient,
                        Matrix &Hessian) const {
        return sampler_->log_posterior(theta, gradient, Hessian, 2);
      }

     private:
      const GammaRegressionPosteriorSampler *sampler_;
      mutable Vector gradient_;
      mutable Matrix Hessian_;
    };

  }  // namespace

  GRPS::GammaRegressionPosteriorSampler(
      GammaRegressionModelBase *model, const Ptr<MvnBase> &coefficient_prior,
      const Ptr<DiffDoubleModel> &shape_parameter_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        coefficient_prior_(coefficient_prior),
        shape_parameter_prior_(shape_parameter_prior),
        epsilon_(1e-5) {}

  void GRPS::reset_shape_parameter_prior(
      const Ptr<DiffDoubleModel> &shape_parameter_prior) {
    shape_parameter_prior_ = shape_parameter_prior;
    mh_sampler_ = nullptr;
  }

  void GRPS::draw() {
    if (!mh_sampler_) {
      find_posterior_mode(epsilon_);
    }
    Vector log_alpha_beta = model_->vectorize_params();
    log_alpha_beta[0] = log(log_alpha_beta[0]);
    log_alpha_beta = mh_sampler_->draw(log_alpha_beta);
    if (mh_sampler_->last_draw_was_accepted()) {
      log_alpha_beta[0] = exp(log_alpha_beta[0]);
      model_->unvectorize_params(log_alpha_beta);
    }
  }

  double GRPS::logpri() const {
    double ans = shape_parameter_prior_->logp(model_->shape_parameter());
    ans += coefficient_prior_->logp(model_->Beta());
    return ans;
  }

  void GRPS::find_posterior_mode(double epsilon) {
    GammaRegressionLogPosterior target(this);
    Vector log_alpha_beta = model_->vectorize_params();
    log_alpha_beta[0] = log(model_->shape_parameter());
    int dim = log_alpha_beta.size();
    Vector gradient(dim);
    Matrix Hessian(dim, dim);
    double log_posterior = 0;
    std::string error_message = "";
    bool ok = max_nd2_careful(log_alpha_beta, gradient, Hessian, log_posterior,
                              Target(target), dTarget(target), d2Target(target),
                              epsilon, error_message);
    if (!ok) {
      std::ostringstream err;
      err << "Trouble finding posterior mode.  Error message:" << std::endl
          << error_message;
      report_error(err.str());
    }
    mh_sampler_.reset(new MetropolisHastings(
        target, new MvtIndepProposal(log_alpha_beta, -Hessian, 3)));
    log_alpha_beta[0] = exp(log_alpha_beta[0]);
    model_->unvectorize_params(log_alpha_beta);
  }

  double GRPS::log_posterior(const Vector &log_alpha_beta, Vector &gradient,
                             Matrix &Hessian, uint nd) const {
    int dim = log_alpha_beta.size();
    Vector alpha_beta = log_alpha_beta;
    double log_alpha = log_alpha_beta[0];
    double alpha = exp(log_alpha);
    alpha_beta[0] = alpha;
    double ans = model_->Loglike(alpha_beta, gradient, Hessian, nd);
    // gradient is d loglike / dalpha_beta

    // Derivatives of the prior on beta, with respect to beta.
    Vector beta_prior_gradient;
    Matrix beta_prior_hessian;
    ans +=
        coefficient_prior_->Logp(ConstVectorView(log_alpha_beta, 1),
                                 beta_prior_gradient, beta_prior_hessian, nd);
    if (nd > 0) {
      VectorView(gradient, 1) += beta_prior_gradient;
      if (nd > 1) {
        SubMatrix(Hessian, 1, dim - 1, 1, dim - 1) += beta_prior_hessian;
      }
    }

    // Derivatives of the prior on alpha, with respect to alpha.
    double d1, d2;
    ans += shape_parameter_prior_->Logp(alpha, d1, d2, nd);
    if (nd > 0) {
      gradient[0] += d1;
      if (nd > 1) {
        Hessian(0, 0) += d2;
      }
    }

    // The Jacobian matrix is the identity, with alpha in element (0,
    // 0), and 1's along the diagonal.
    if (nd > 0) {
      gradient[0] *= alpha;
      if (nd > 1) {
        // The Hessian is (J * H * J^T) + (J2 * g).  The JHJ bit
        // scales the first row and column.  The J2g bit will be
        // handled below.
        Hessian.row(0) *= alpha;
        Hessian.col(0) *= alpha;

        // The only component of the gradient that has a nonzero
        // derivative in the Jacobian is the first bit, and the only
        // component with which it has a nonzero derivative is the
        // first bit, so we just need to add in the original gradient
        // * d^alpha /da^2.  dalpha / da = alpha, so the second
        // derivative is still alpha.  When the gradient was adjusted
        // above, gradient[0] became the original gradient[0] * alpha,
        // which is what we need to add to the (0, 0) element of the
        // Hessian.
        Hessian(0, 0) += gradient[0];
      }
    }

    // Add in the jacobian term, which is just an extra factor of
    // alpha.  Because this is log_posterior, we have to take its log.
    ans += log(alpha);

    if (nd > 0) {
      gradient[0] += 1.0;
      // No adjustment needed for Hessian.
    }

    return ans;
  }

}  // namespace BOOM
