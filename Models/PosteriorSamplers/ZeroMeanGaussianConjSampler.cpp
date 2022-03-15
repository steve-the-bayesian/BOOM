// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2008 Steven L. Scott

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

#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/GammaModel.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  typedef ZeroMeanGaussianConjSampler ZGS;

  ZGS::ZeroMeanGaussianConjSampler(ZeroMeanGaussianModel *model,
                                   const Ptr<GammaModelBase> &precision_prior,
                                   RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        precision_prior_(precision_prior),
        variance_sampler_(precision_prior) {}

  ZGS::ZeroMeanGaussianConjSampler(ZeroMeanGaussianModel *model, double df,
                                   double sigma_guess, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        precision_prior_(new ChisqModel(df, sigma_guess)),
        variance_sampler_(precision_prior_) {}

  ZGS *ZGS::clone() const { return new ZGS(*this); }

  ZGS *ZGS::clone_to_new_host(Model *new_host) const {
    ZGS *ans = new ZGS(
        dynamic_cast<ZeroMeanGaussianModel *>(new_host),
        precision_prior_->clone(),
        rng());
    ans->set_sigma_upper_limit(variance_sampler_.sigma_max());
    return ans;
  }

  void ZGS::draw() {
    model_->set_sigsq(variance_sampler_.draw(rng(), model_->suf()->n(),
                                             model_->suf()->sumsq()));
  }

  double ZGS::logpri() const {
    return variance_sampler_.log_prior(model_->sigsq());
  }

  double ZGS::sigma_prior_guess() const {
    return variance_sampler_.sigma_prior_guess();
  }

  double ZGS::sigma_prior_sample_size() const {
    return variance_sampler_.sigma_prior_sample_size();
  }

  void ZGS::set_sigma_upper_limit(double sigma_upper_limit) {
    variance_sampler_.set_sigma_max(sigma_upper_limit);
  }

  // The logic of the posterior mode here is as follows.  The prior is a Gamma
  // on 1 / sigsq, but the parameter we really care about (i.e. the parameter as
  // expressed in the likelihood function for ZeroMeanGaussianModel) is sigsq.
  // This introduces a Jacobian term that needs to be taken account of in the
  // optimization.  The mode of the gamma distribution is (a-1)/b, but the mode
  // of the inverse gamma distribution is b/(a+1).
  //
  // The deciding factor is that the prior on sigsq is a Gamma model on 1/sigsq
  // and not an inverse Gamma model on sigsq.  For this result to agree with
  // numerical optimizers we need to do the optimization with respect to
  // 1/sigsq.
  void ZGS::find_posterior_mode(double) {
    double sigsq_mode = variance_sampler_.posterior_mode(
        model_->suf()->n(), model_->suf()->sumsq());
    model_->set_sigsq(sigsq_mode);
  }

  double ZGS::increment_log_prior_gradient(const ConstVectorView &parameters,
                                           VectorView gradient) const {
    if (parameters.size() != 1 || gradient.size() != 1) {
      report_error(
          "Wrong size arguments passed to "
          "ZeroMeanGaussianConjSampler::increment_log_prior_gradient.");
    }
    return log_prior(parameters[0], &gradient[0], nullptr);
  }

  double ZGS::log_prior_density(const ConstVectorView &parameters) const {
    if (parameters.size() != 1) {
      report_error(
          "Wrong size parameters passed to "
          "ZeroMeanGaussianConjSampler::log_prior_density.");
    }
    return log_prior(parameters[0], nullptr, nullptr);
  }

  double ZGS::log_prior(double sigsq, double *d1, double *d2) const {
    if (sigsq <= 0.0) {
      return negative_infinity();
    }
    double a = precision_prior_->alpha();
    double b = precision_prior_->beta();
    // The log prior is the gamma density plus a jacobian term:
    // log(abs(d(siginv) / d(sigsq))).
    if (d1) {
      double sig4 = sigsq * sigsq;
      *d1 += -(a + 1) / sigsq + b / sig4;
      if (d2) {
        double sig6 = sigsq * sig4;
        *d2 += (a + 1) / sig4 - 2 * b / sig6;
      }
    }
    return dgamma(1 / sigsq, a, b, true) - 2 * log(sigsq);
  }

  double ZGS::log_posterior(double sigsq, double &d1, double &d2,
                            uint nd) const {
    // The log likelihood is already parameterized with respect to
    // sigma^2, so derivatives are easy.
    double logp = model_->log_likelihood(sigsq, nd > 0 ? &d1 : nullptr,
                                         nd > 1 ? &d2 : nullptr);
    return logp +
           log_prior(sigsq, nd > 0 ? &d1 : nullptr, nd > 1 ? &d2 : nullptr);
  }

}  // namespace BOOM
