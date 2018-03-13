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

#include "Models/Glm/PosteriorSamplers/TRegressionSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    class TRegressionLogPosterior {
     public:
      TRegressionLogPosterior(TRegressionModel *model,
                              const Ptr<DoubleModel> &nu_prior)
          : model_(model), nu_prior_(nu_prior) {}

      double operator()(double nu) const {
        double ans = nu_prior_->logp(nu);
        if (ans <= negative_infinity()) {
          return ans;
        }
        return ans +
               model_->log_likelihood(model_->Beta(), model_->sigsq(), nu);
      }

     private:
      TRegressionModel *model_;
      Ptr<DoubleModel> nu_prior_;
    };

    class TRegressionCompleteDataLogPosterior {
     public:
      TRegressionCompleteDataLogPosterior(
          const Ptr<ScaledChisqModel> &complete_data_model,
          const Ptr<DoubleModel> &prior)
          : complete_data_model_(complete_data_model), prior_(prior) {}

      double operator()(double nu) const {
        if (nu <= 0.0) {
          return negative_infinity();
        }
        double ans = prior_->logp(nu);
        if (ans <= negative_infinity()) {
          return ans;
        }
        return ans + complete_data_model_->log_likelihood(nu);
      }

     private:
      Ptr<ScaledChisqModel> complete_data_model_;
      Ptr<DoubleModel> prior_;
    };

  }  // namespace

  TRegressionSampler::TRegressionSampler(
      TRegressionModel *model, const Ptr<MvnBase> &coefficient_prior,
      const Ptr<GammaModelBase> &siginv_prior, const Ptr<DoubleModel> &nu_prior,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        coefficient_prior_(coefficient_prior),
        siginv_prior_(siginv_prior),
        nu_prior_(nu_prior),
        weight_model_(new ScaledChisqModel(model_->nu())),
        complete_data_sufficient_statistics_(model_->xdim()),
        sigsq_sampler_(siginv_prior_),
        nu_observed_data_sampler_(TRegressionLogPosterior(model_, nu_prior_),
                                  false, 1.0, &rng()),
        nu_complete_data_sampler_(
            TRegressionCompleteDataLogPosterior(weight_model_, nu_prior_),
            false, 1.0, &rng()),
        latent_data_is_fixed_(false) {
    nu_observed_data_sampler_.set_lower_limit(0.0);
    nu_complete_data_sampler_.set_lower_limit(0.0);
  }

  void TRegressionSampler::draw() {
    impute_latent_data();
    draw_beta_full_conditional();
    draw_sigsq_full_conditional();
    draw_nu_given_observed_data();
  }

  double TRegressionSampler::logpri() const {
    double ans = nu_prior_->logp(model_->nu());
    ans += sigsq_sampler_.log_prior(model_->sigsq());
    ans += coefficient_prior_->logp(model_->Beta());
    return ans;
  }

  void TRegressionSampler::impute_latent_data() {
    if (!latent_data_is_fixed_) {
      complete_data_sufficient_statistics_.clear();
      weight_model_->suf()->clear();
      const std::vector<Ptr<RegressionData> > &data(model_->dat());
      for (int i = 0; i < data.size(); ++i) {
        double mu = model_->predict(data[i]->x());
        double residual = data[i]->y() - mu;
        double weight = data_imputer_.impute(rng(), residual, model_->sigma(),
                                             model_->nu());
        weight_model_->suf()->update_raw(weight);
        complete_data_sufficient_statistics_.add_data(data[i]->x(),
                                                      data[i]->y(), weight);
      }
    }
  }

  // Y ~ N(X * beta, sigma^2 * W^{-1}), where W is diagonal.
  // p(Y|.)  \propto \exp( -0.5 * (Y - Xb)^T W (Y - XB) / sigsq)
  // p(beta|.-Y) \propto \exp( -0.5 (beta - b)^T Ominv (beta - b))
  //
  // p(beta|Y, .) \propto exp(beta_tilde, V)
  //    where V^{-1} = Ominv + X'WX/sigsq
  //    beta_tilde = V * (Ominv * b + X'Wy/sigsq)
  void TRegressionSampler::draw_beta_full_conditional() {
    SpdMatrix Vinv =
        coefficient_prior_->siginv() +
        complete_data_sufficient_statistics_.xtx() / model_->sigsq();
    Vector Vinv_beta_tilde =
        coefficient_prior_->siginv() * coefficient_prior_->mu() +
        complete_data_sufficient_statistics_.xty() / model_->sigsq();
    Vector beta = rmvn_suf_mt(rng(), Vinv, Vinv_beta_tilde);
    model_->set_Beta(beta);
  }

  // p(1/sigsq) \propto (1/sigsq)^(df/2 - 1) * exp(-ss/2  * ss)
  // p(y | sigsq, w) \propto
  //      \prod_i (w[i]/sigsq)^1/2 exp( -0.5 (y[i] - mu[i])^2 * w[i] / sigsq)
  void TRegressionSampler::draw_sigsq_full_conditional() {
    double sigsq = sigsq_sampler_.draw(
        rng(), complete_data_sufficient_statistics_.n(),
        complete_data_sufficient_statistics_.weighted_sum_of_squared_errors(
            model_->Beta()));
    model_->set_sigsq(sigsq);
  }

  void TRegressionSampler::draw_nu_given_complete_data() {
    double nu = nu_complete_data_sampler_.draw(model_->nu());
    model_->set_nu(nu);
  }

  void TRegressionSampler::draw_nu_given_observed_data() {
    double nu = nu_observed_data_sampler_.draw(model_->nu());
    model_->set_nu(nu);
  }

  void TRegressionSampler::set_sigma_upper_limit(double max_sigma) {
    sigsq_sampler_.set_sigma_max(max_sigma);
  }

  void TRegressionSampler::clear_complete_data_sufficient_statistics() {
    complete_data_sufficient_statistics_.clear();
    weight_model_->clear_data();
  }

  void TRegressionSampler::update_complete_data_sufficient_statistics(
      double y, const Vector &x, double weight) {
    complete_data_sufficient_statistics_.add_data(x, y, weight);
    weight_model_->suf()->update_raw(weight);
  }

  void TRegressionSampler::fix_latent_data(bool fixed) {
    latent_data_is_fixed_ = fixed;
  }

}  // namespace BOOM
