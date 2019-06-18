// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

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

#include "Models/Glm/PosteriorSamplers/BinomialProbitTimSampler.hpp"
#include "numopt.hpp"

namespace BOOM {

  namespace {
    // A functor class for evaluating log posterior.
    class LogPosterior {
     public:
      LogPosterior(BinomialProbitModel *model, const Ptr<MvnBase> &prior)
          : model_(model), prior_(prior) {}

      double operator()(const Vector &beta) const {
        double ans = prior_->logp_given_inclusion(beta, nullptr, nullptr,
                                                  model_->coef().inc(), true);
        if (std::isfinite(ans)) {
          ans += model_->log_likelihood(beta, nullptr, nullptr, false);
        }
        return ans;
      }

      double operator()(const Vector &beta, Vector &gradient) const {
        double ans = prior_->logp_given_inclusion(beta, &gradient, nullptr,
                                                  model_->coef().inc(), true);
        if (std::isfinite(ans)) {
          ans += model_->log_likelihood(beta, &gradient, nullptr, false);
        }
        return ans;
      }

      double operator()(const Vector &beta, Vector &gradient,
                        Matrix &hessian) const {
        double ans = prior_->logp_given_inclusion(beta, &gradient, &hessian,
                                                  model_->coef().inc(), true);
        if (std::isfinite(ans)) {
          ans += model_->log_likelihood(beta, &gradient, &hessian, false);
        }
        return ans;
      }

     private:
      BinomialProbitModel *model_;
      Ptr<MvnBase> prior_;
    };

  }  // namespace

  BinomialProbitTimSampler::BinomialProbitTimSampler(BinomialProbitModel *model,
                                                     const Ptr<MvnBase> &prior,
                                                     double proposal_df,
                                                     RNG &rng)
      : PosteriorSampler(rng),
        model_(model),
        prior_(prior),
        proposal_df_(proposal_df) {}

  double BinomialProbitTimSampler::logpri() const {
    return prior_->logp_given_inclusion(model_->included_coefficients(),
                                        nullptr, nullptr, model_->coef().inc(),
                                        false);
  }

  void BinomialProbitTimSampler::draw() {
    const Selector &included_coefficients(model_->inc());
    if (included_coefficients.nvars() == 0) {
      return;
    }
    auto it = samplers_.find(included_coefficients);
    if (it == samplers_.end()) {
      LogPosterior log_posterior(model_, prior_);
      TIM sampler(log_posterior, log_posterior, log_posterior, proposal_df_,
                  &rng());
      sampler.locate_mode(model_->included_coefficients());
      sampler.fix_mode(true);
      samplers_.emplace(included_coefficients, sampler);
      it = samplers_.find(included_coefficients);
    }
    Vector beta = it->second.draw(model_->included_coefficients());
    model_->set_included_coefficients(beta);
  }

}  // namespace BOOM
