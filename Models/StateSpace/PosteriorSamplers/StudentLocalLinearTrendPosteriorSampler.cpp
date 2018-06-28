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

#include "Models/StateSpace/PosteriorSamplers/StudentLocalLinearTrendPosteriorSampler.hpp"
#include "Samplers/ScalarSliceSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"
#include "distributions/trun_gamma.hpp"

namespace BOOM {

  namespace {
    // A local namespace for minor implementation details.
    class NuPosteriorFast {
     public:
      NuPosteriorFast(const BOOM::DoubleModel *nu_prior,
                      const BOOM::GammaSuf *suf)
          : nu_prior_(nu_prior), suf_(suf) {}

      // Returns the un-normalized log posterior evaulated at nu.
      double operator()(double nu) const {
        double n = suf_->n();
        double sum = suf_->sum();
        double sumlog = suf_->sumlog();
        double nu2 = nu / 2.0;

        double ans = nu_prior_->logp(nu);
        ans += n * (nu2 * log(nu2) - lgamma(nu2));
        ans += (nu2 - 1) * sumlog;
        ans -= nu2 * sum;
        return ans;
      }
     private:
      const BOOM::DoubleModel *nu_prior_;
      const BOOM::GammaSuf *suf_;
    };

    class NuPosteriorRobust {
     public:
      // Args:
      //   nu_prior_: A prior distribution for the tail thickness parameter of
      //     the student T distribution.
      //   residuals: A set of zero-mean data drawn from the T_nu(0, sigma)
      //     distribution.
      //    sigma: The scatter parameter of the T distribution describing
      //      'residuals'.
      NuPosteriorRobust(const BOOM::DoubleModel *nu_prior,
                        const Vector &residuals,
                        double sigma)
          : nu_prior_(nu_prior), residuals_(residuals), sigma_(sigma) {}

      // Return the un-normalized log posterior, conditional on mu and sigma
      // evaluated at nu.
      double operator()(double nu) const {
        double ans = nu_prior_->logp(nu);
        if (!std::isfinite(ans)) {
          return ans;
        }
        for (double r : residuals_) {
          ans += dstudent(r, 0, sigma_, nu, true);
        }
        return ans;
      }

     private:
      const BOOM::DoubleModel *nu_prior_;
      const BOOM::Vector &residuals_;
      double sigma_;
    };
    
    typedef StudentLocalLinearTrendPosteriorSampler SLLTPS;

  }  // namespace

  SLLTPS::StudentLocalLinearTrendPosteriorSampler(
      StudentLocalLinearTrendStateModel *model,
      const Ptr<GammaModelBase> &sigsq_level_prior,
      const Ptr<DoubleModel> &nu_level_prior,
      const Ptr<GammaModelBase> &sigsq_slope_prior,
      const Ptr<DoubleModel> &nu_slope_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        sigsq_level_prior_(sigsq_level_prior),
        nu_level_prior_(nu_level_prior),
        sigsq_slope_prior_(sigsq_slope_prior),
        nu_slope_prior_(nu_slope_prior),
        sigsq_level_sampler_(sigsq_level_prior_),
        sigsq_slope_sampler_(sigsq_slope_prior_) {}

  double SLLTPS::logpri() const {
    return sigsq_level_sampler_.log_prior(model_->sigsq_level()) +
           nu_level_prior_->logp(model_->nu_level()) +
           sigsq_slope_sampler_.log_prior(model_->sigsq_slope()) +
           nu_slope_prior_->logp(1.0 / model_->nu_slope());
  }

  void SLLTPS::draw() {
    draw_sigsq_level();
    draw_nu_level();
    draw_sigsq_slope();
    draw_nu_slope();
  }

  void SLLTPS::set_sigma_level_upper_limit(double upper_limit) {
    sigsq_level_sampler_.set_sigma_max(upper_limit);
  }

  void SLLTPS::set_sigma_slope_upper_limit(double upper_limit) {
    sigsq_slope_sampler_.set_sigma_max(upper_limit);
  }

  void SLLTPS::draw_sigsq_level() {
    const WeightedGaussianSuf &suf(model_->sigma_level_complete_data_suf());
    double sigsq = sigsq_level_sampler_.draw(rng(), suf.n(), suf.sumsq());
    model_->set_sigsq_level(sigsq);
  }

  void SLLTPS::draw_sigsq_slope() {
    const WeightedGaussianSuf &suf(model_->sigma_slope_complete_data_suf());
    double sigsq = sigsq_slope_sampler_.draw(rng(), suf.n(), suf.sumsq());
    model_->set_sigsq_slope(sigsq);
  }

  void SLLTPS::draw_nu_level() {
    std::function<double(double)> logpost;
    if (model_->nu_level() > 20) {
      logpost = NuPosteriorRobust(nu_level_prior_.get(),
                                  model_->level_residuals(),
                                  model_->sigma_level());
    } else {
      logpost = NuPosteriorFast(nu_level_prior_.get(),
                                &model_->nu_level_complete_data_suf());
    }

    ScalarSliceSampler sampler(logpost, true);
    sampler.set_lower_limit(0.0);
    double nu = sampler.draw(model_->nu_level());
    model_->set_nu_level(nu);
  }

  void SLLTPS::draw_nu_slope() {
    NuPosteriorFast logpost(nu_slope_prior_.get(),
                            &model_->nu_slope_complete_data_suf());
    ScalarSliceSampler sampler(logpost, true);
    sampler.set_lower_limit(0.0);
    double nu = sampler.draw(model_->nu_slope());
    model_->set_nu_slope(nu);
  }

}  // namespace BOOM
