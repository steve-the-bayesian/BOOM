/*
  Copyright (C) 2005-2022 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/Mixtures/PosteriorSamplers/BetaBinomialMixturePosteriorSampler.hpp"
#include "distributions.hpp"
#include "TargetFun/MultinomialLogitTransform.hpp"
#include "TargetFun/LogitTransform.hpp"
#include "TargetFun/LogTransform.hpp"
#include "stats/logit.hpp"

namespace BOOM {

  BetaBinomialMixturePosteriorSampler::BetaBinomialMixturePosteriorSampler(
      BetaBinomialMixtureModel *model, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model)
  {}

  double BetaBinomialMixturePosteriorSampler::logpri() const {
    double ans = model_->mixing_distribution()->logpri();
    for (int s = 0; s < model_->number_of_mixture_components(); ++s) {
      ans += model_->mixture_component(s)->logpri();
    }
    return ans;
  }

  void BetaBinomialMixturePosteriorSampler::draw() {
    model_->impute_latent_data(rng());
    model_->mixing_distribution()->sample_posterior();
    for (int s = 0; s < model_->number_of_mixture_components(); ++s) {
      model_->mixture_component(s)->sample_posterior();
    }
  }

  //===========================================================================
  namespace {
    using Direct = BetaBinomialMixtureDirectPosteriorSampler;
  }  // namespace

  Direct::BetaBinomialMixtureDirectPosteriorSampler(
      BetaBinomialMixtureModel *model,
      const Ptr<DirichletModel> &mixing_weight_prior,
      const std::vector<Ptr<BetaModel>> &component_mean_priors,
      const std::vector<Ptr<DoubleModel>> &sample_size_priors,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        mixing_weight_prior_(mixing_weight_prior),
        component_mean_priors_(component_mean_priors),
        sample_size_priors_(sample_size_priors),
        sampler_([this](const Vector &theta){
                   return this->log_posterior(theta);},
          1.0, false, &rng())
  {}

  double Direct::logpri() const {
    double ans = mixing_weight_prior_->logp(model_->mixing_weights());
    for (int s = 0; s < model_->number_of_mixture_components(); ++s) {
      ans += component_mean_priors_[s]->logp(
          model_->mixture_component(s)->mean());
      ans += sample_size_priors_[s]->logp(
          model_->mixture_component(s)->prior_sample_size());
    }
    return ans;
  }

  void Direct::draw() {
    Vector theta = sampler_.draw(pack_theta());
    Vector mixing_weights, component_means, sample_sizes;
    unpack_theta(theta, mixing_weights, component_means, sample_sizes);
    model_->mixing_distribution()->set_pi(mixing_weights);
    for (size_t s = 0; s < component_means.size(); ++s) {
      double a = component_means[s] * sample_sizes[s];
      double b = sample_sizes[s] - a;
      model_->mixture_component(s)->set_a(a);
      model_->mixture_component(s)->set_b(b);
    }
  }

  // Args:
  //   theta: Vector of packed and transformed model parameters.  If there are K
  //     components then the first K-1 elements of theta are the multinomial
  //     logit transformation of the mixing weights.  The next 2K elements are
  //     logit(mean) and log(sample_size) for models 1, 2, 3, ..., K.
  double Direct::log_posterior(const Vector &theta) const {
    //----------------------------------------------------------------------
    // Unpack the vector and transform the parameters to their original scale.
    Vector mixing_weights, component_means, sample_sizes;
    unpack_theta(theta, mixing_weights, component_means, sample_sizes);

    //----------------------------------------------------------------------
    // Evaluate the priors are on probs and sample sizes.
    double ans = mixing_weight_prior_->logp(mixing_weights);
    if (!std::isfinite(ans)) return ans;

    for (int i = 0; i < component_means.size(); ++i) {
      ans += component_mean_priors_[i]->logp(component_means[i]);
      ans += sample_size_priors_[i]->logp(sample_sizes[i]);
    }
    if (!std::isfinite(ans)) return ans;

    //----------------------------------------------------------------------
    // The log likelihood needs mixture_weights, and the (a,b) matrix.
    Matrix ab(component_means.size(), 2);
    ab.col(0) = component_means * sample_sizes;
    ab.col(1) = sample_sizes - ab.col(0);
    ans += model_->log_likelihood(mixing_weights, ab);
    if (!std::isfinite(ans)) return ans;

    //----------------------------------------------------------------------
    // Add the log Jacobians from the various transformations.
    MultinomialLogitJacobian mlogit_jacobian;
    ans += mlogit_jacobian.logdet(ConstVectorView(mixing_weights, 1));

    LogTransformJacobian log_jacobian;
    ans += log_jacobian.logdet(sample_sizes);

    LogitTransformJacobian logit_jacobian;
    ans += logit_jacobian.logdet(component_means);

    return ans;
  }

  Vector Direct::pack_theta() const {
    MultinomialLogitTransform mlogit;
    Vector ans = mlogit.to_logits(model_->mixing_weights());
    for (int s = 0; s < model_->number_of_mixture_components(); ++s) {
      ans.push_back(logit(model_->mixture_component(s)->mean()));
      ans.push_back(log(model_->mixture_component(s)->prior_sample_size()));
    }
    return ans;
  }

  void Direct::unpack_theta(
      const Vector &theta,
      Vector &mixing_weights,
      Vector &component_means,
      Vector &sample_sizes) const {
    int num_clusters = (theta.size() + 1) / 3;
    ConstVectorView logits(theta, 0, num_clusters - 1);
    ConstVectorView logit_probs(theta.data() + num_clusters - 1, num_clusters, 2);
    ConstVectorView log_sample_sizes(theta.data() + num_clusters, num_clusters, 2);

    MultinomialLogitTransform mlogit;
    bool truncate = false;
    mixing_weights = mlogit.to_probs(logits, truncate);

    component_means = logit_inv(logit_probs);
    sample_sizes = exp(log_sample_sizes);
  }

}  // namespace BOOM
