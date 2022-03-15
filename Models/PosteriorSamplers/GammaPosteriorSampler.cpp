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

#include "Models/PosteriorSamplers/GammaPosteriorSampler.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {

  // A private namespace for defining functors used just for implementation.
  namespace {
    // A functor for evaluating the log posterior of the mean
    // parameter in the mean/alpha parameterization.
    class GammaMeanAlphaLogPosterior {
     public:
      GammaMeanAlphaLogPosterior(const GammaModel *model,
                                 const DoubleModel *mean_prior)
          : model_(model), mean_prior_(mean_prior) {}

      double operator()(double mean) const {
        if (mean <= 0) {
          return negative_infinity();
        }
        double a = model_->alpha();
        double b = a / mean;
        double ans = mean_prior_->logp(mean);
        ans += model_->loglikelihood(a, b);
        return ans;
      }

     private:
      const GammaModel *model_;
      const DoubleModel *mean_prior_;
    };

    // A functor for evaluating the log posterior of the mean
    // parameter in the mean/beta parameterization.
    class GammaMeanBetaLogPosterior {
     public:
      GammaMeanBetaLogPosterior(const GammaModel *model,
                                const DoubleModel *mean_prior)
          : model_(model), mean_prior_(mean_prior) {}

      double operator()(double mean) const {
        if (mean <= 0.0) {
          return negative_infinity();
        }
        double ans = mean_prior_->logp(mean);
        double b = model_->beta();
        double a = mean * b;
        ans += model_->loglikelihood(a, b);
        return ans;
      }

     private:
      const GammaModel *model_;
      const DoubleModel *mean_prior_;
    };

    // A functor for evaluating the log posterior of the alpha
    // parameter in the mean/alpha parameterization.
    class GammaAlphaLogPosterior {
     public:
      GammaAlphaLogPosterior(const GammaModel *model,
                             const DoubleModel *alpha_prior)
          : model_(model), alpha_prior_(alpha_prior) {}

      double operator()(double alpha) const {
        if (alpha <= 0) {
          return negative_infinity();
        }
        double mean = model_->alpha() / model_->beta();
        double beta = alpha / mean;
        double ans = alpha_prior_->logp(alpha);
        ans += model_->loglikelihood(alpha, beta);
        return ans;
      }

     private:
      const GammaModel *model_;
      const DoubleModel *alpha_prior_;
    };

    // A functor for evaluating the log posterior of the beta
    // parameter in the mean/beta parameterization.
    class GammaBetaLogPosterior {
     public:
      GammaBetaLogPosterior(const GammaModel *model,
                            const DoubleModel *beta_prior)
          : model_(model), beta_prior_(beta_prior) {}

      double operator()(double beta) const {
        if (beta <= 0.0) {
          return negative_infinity();
        }
        double ans = beta_prior_->logp(beta);
        double mean = model_->alpha() / model_->beta();
        double a = mean * beta;
        ans += model_->loglikelihood(a, beta);
        return ans;
      }

     private:
      const GammaModel *model_;
      const DoubleModel *beta_prior_;
    };

  }  // namespace

  //======================================================================
  GammaPosteriorSampler::GammaPosteriorSampler(
      GammaModel *model, const Ptr<DoubleModel> &mean_prior,
      const Ptr<DoubleModel> &alpha_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        mean_prior_(mean_prior),
        alpha_prior_(alpha_prior),
        mean_sampler_(GammaMeanAlphaLogPosterior(model_, mean_prior_.get()),
                      true, 1.0, &seeding_rng),
        alpha_sampler_(GammaAlphaLogPosterior(model_, alpha_prior_.get()), true,
                       1.0, &seeding_rng) {
    mean_sampler_.set_lower_limit(0);
    alpha_sampler_.set_lower_limit(0);
  }

  GammaPosteriorSampler *GammaPosteriorSampler::clone_to_new_host(
      Model *new_host) const {
    return new GammaPosteriorSampler(
        dynamic_cast<GammaModel *>(new_host),
        mean_prior_->clone(),
        alpha_prior_->clone(),
        rng());
  }


  void GammaPosteriorSampler::draw() {
    // Draw alpha with the mean fixed.
    double mean = model_->mean();
    double alpha = alpha_sampler_.draw(model_->alpha());
    model_->set_shape_and_mean(alpha, mean);

    // Then draw the mean with alpha fixed.
    mean = mean_sampler_.draw(mean);
    model_->set_shape_and_mean(alpha, mean);
  }

  double GammaPosteriorSampler::logpri() const {
    double a = model_->alpha();
    double mean = a / model_->beta();
    return mean_prior_->logp(mean) + alpha_prior_->logp(a);
  }

  //======================================================================

  GammaPosteriorSamplerBeta::GammaPosteriorSamplerBeta(
      GammaModel *model, const Ptr<DoubleModel> &mean_prior,
      const Ptr<DoubleModel> &beta_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        mean_prior_(mean_prior),
        beta_prior_(beta_prior),
        mean_sampler_(GammaMeanBetaLogPosterior(model, mean_prior.get()), false,
                      1.0, &rng()),
        beta_sampler_(GammaBetaLogPosterior(model, beta_prior.get()), false,
                      1.0, &rng()) {}

  GammaPosteriorSamplerBeta *GammaPosteriorSamplerBeta::clone_to_new_host(
      Model *new_host) const {
    return new GammaPosteriorSamplerBeta(
        dynamic_cast<GammaModel *>(new_host),
        mean_prior_->clone(),
        beta_prior_->clone(),
        rng());
  }

  void GammaPosteriorSamplerBeta::draw() {
    // Draw beta given mean.
    double mean = model_->mean();
    double beta = beta_sampler_.draw(model_->beta());
    model_->set_mean_and_scale(mean, beta);

    // Draw mean given beta.
    mean = mean_sampler_.draw(mean);
    model_->set_mean_and_scale(mean, beta);
  }

  double GammaPosteriorSamplerBeta::logpri() const {
    double beta = model_->beta();
    double mean = model_->alpha() / beta;
    if (mean <= 0 || beta <= 0) {
      return negative_infinity();
    }
    return mean_prior_->logp(mean) + beta_prior_->logp(beta);
  }

}  // namespace BOOM
