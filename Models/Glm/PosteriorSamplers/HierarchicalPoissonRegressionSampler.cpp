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

#include "Models/Glm/PosteriorSamplers/HierarchicalPoissonRegressionSampler.hpp"
#include "distributions.hpp"

namespace {
  //  Some global constants that can be used to control the sub-steps
  //  of the sampling algorithm.
  const bool draw_beta = true;
  const bool redraw_alpha_and_sigma = true;
}  // namespace

namespace BOOM {

  typedef HierarchicalPoissonRegressionPosteriorSampler HPRS;
  HPRS::HierarchicalPoissonRegressionPosteriorSampler(
      HierarchicalPoissonRegressionModel *model, const Ptr<MvnBase> &mu_prior,
      int nthreads, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        mu_prior_(mu_prior),
        zero_mean_random_effect_model_(new ZeroMeanMvnModel(model->xdim())),
        nthreads_(nthreads) {
    if (!model) {
      report_error(
          "NULL model passed to "
          "HierarchicalPoissonRegressionPosteriorSampler "
          "constructor.");
    }
    if (model->xdim() != mu_prior->dim()) {
      report_error(
          "The model and mu_prior arguments do not conform in "
          "HierarchicalPoissonRegressionPosteriorSampler "
          "constructor.");
    }
  }

  void HPRS::draw() {
    check_data_model_samplers();
    impute_latent_data();
    draw_mu_and_sigma_given_beta();
    if (redraw_alpha_and_sigma) {
      compute_zero_mean_sufficient_statistics();
      draw_mu_given_zero_mean_sufficient_statistics();
      draw_sigma_given_zero_mean_sufficient_statistics();
    }
  }

  // It is not possible for a HierarchicalPoissonRegressionModel to
  // remove a model once it has been added.  Therefore,
  // check_data_model_samplers only needs to check that the size of
  // the vector of samplers matches the number of data_models in
  // model_.  If there are too few samplers, more will be added.
  void HPRS::check_data_model_samplers() {
    int nmodels = model_->number_of_groups();
    while (data_model_samplers_.size() < nmodels) {
      PoissonRegressionModel *data_model =
          model_->data_model(data_model_samplers_.size());
      Ptr<MvnModel> data_parent_model(model_->data_parent_model());
      NEW(PoissonRegressionAuxMixSampler, sampler)
      (data_model, data_parent_model, nthreads_);
      data_model_samplers_.push_back(sampler);
    }
  }

  void HPRS::impute_latent_data() {
    MvnModel *data_parent_model = model_->data_parent_model();
    data_parent_model->clear_data();
    for (int i = 0; i < data_model_samplers_.size(); ++i) {
      if (draw_beta) {
        //        cerr << "drawing beta" << endl;
        data_model_samplers_[i]->draw();
      } else {
        //        cerr << "drawing latent data, but not changing beta" << endl;
        const Vector beta = model_->data_model(i)->Beta();
        data_model_samplers_[i]->draw();
        model_->data_model(i)->set_Beta(beta);
      }
      Ptr<VectorData> beta = model_->data_model(i)->coef_prm();
      data_parent_model->add_data(beta);
    }
  }

  void HPRS::compute_zero_mean_sufficient_statistics() {
    zero_mean_random_effect_model_->clear_data();
    Vector mu(model_->data_parent_model()->mu());
    Vector alpha(mu.size());

    xtx_ = mu_prior_->siginv();
    xtu_ = mu_prior_->siginv() * mu_prior_->mu();

    for (int i = 0; i < model_->number_of_groups(); ++i) {
      alpha = model_->data_model(i)->Beta() - mu;
      zero_mean_random_effect_model_->suf()->update_raw(alpha);
      const WeightedRegSuf &local_suf(
          data_model_samplers_[i]->complete_data_sufficient_statistics());
      xtx_ += local_suf.xtx();
      xtu_ += local_suf.xty() - local_suf.xtx() * alpha;
    }
  }

  void HPRS::draw_mu_given_zero_mean_sufficient_statistics() {
    Vector mu = rmvn_suf_mt(rng(), xtx_, xtu_);
    model_->data_parent_model()->set_mu(mu);
  }

  ZeroMeanMvnModel *HPRS::zero_mean_random_effect_model() {
    return zero_mean_random_effect_model_.get();
  }

  const ZeroMeanMvnModel *HPRS::zero_mean_random_effect_model() const {
    return zero_mean_random_effect_model_.get();
  }

  //======================================================================

  typedef HierarchicalPoissonRegressionConjugatePosteriorSampler HPCRS;
  HPCRS::HierarchicalPoissonRegressionConjugatePosteriorSampler(
      HierarchicalPoissonRegressionModel *model, const Ptr<MvnBase> &mu_prior,
      const Ptr<WishartModel> &siginv_wishart_prior, int nthreads)
      : HierarchicalPoissonRegressionPosteriorSampler(model, mu_prior,
                                                      nthreads),
        siginv_prior_(siginv_wishart_prior),
        zero_mean_sigma_sampler_(new ZeroMeanMvnConjSampler(
            zero_mean_random_effect_model(), siginv_wishart_prior)),
        sigma_given_beta_sampler_(new MvnVarSampler(model->data_parent_model(),
                                                    siginv_wishart_prior)),
        mu_given_beta_sampler_(
            new MvnMeanSampler(model->data_parent_model(), mu_prior)) {}

  double HPCRS::logpri() const {
    return mu_prior()->logp(data_parent_model()->mu()) +
           siginv_prior_->logp(data_parent_model()->siginv());
  }

  void HPCRS::draw_mu_and_sigma_given_beta() {
    // Sufficient statistics for mu and sigma given beta were
    // accumulated during impute_latent_data().
    sigma_given_beta_sampler_->draw();
    mu_given_beta_sampler_->draw();
  }

  void HPCRS::draw_sigma_given_zero_mean_sufficient_statistics() {
    zero_mean_sigma_sampler_->draw();
    data_parent_model()->set_siginv(zero_mean_random_effect_model()->siginv());
  }

  //======================================================================

  typedef HierarchicalPoissonRegressionIndependencePosteriorSampler HPRIPS;
  HPRIPS::HierarchicalPoissonRegressionIndependencePosteriorSampler(
      HierarchicalPoissonRegressionModel *model, const Ptr<MvnBase> &mu_prior,
      const std::vector<Ptr<GammaModelBase> > &siginv_priors,
      const Vector &upper_sigma_truncation_point, int nthreads)
      : HierarchicalPoissonRegressionPosteriorSampler(model, mu_prior,
                                                      nthreads),
        siginv_priors_(siginv_priors),
        mu_given_beta_sampler_(
            new MvnMeanSampler(model->data_parent_model(), mu_prior)),
        sigma_given_beta_sampler_(new MvnIndependentVarianceSampler(
            model->data_parent_model(), siginv_priors,
            upper_sigma_truncation_point)),
        zero_mean_sigma_sampler_(new ZeroMeanMvnCompositeIndependenceSampler(
            zero_mean_random_effect_model(), siginv_priors,
            upper_sigma_truncation_point)) {
    if (model->xdim() != siginv_priors.size()) {
      report_error(
          "model and 'siginv_priors' arguments do not conform in "
          "HierarchicalPoissonRegressionIndependencePosteriorSampler "
          "constructor.");
    }
    if (model->xdim() != upper_sigma_truncation_point.size()) {
      report_error(
          "model snad upper_sigma_truncation_point arguments do not "
          "conform in "
          "HierarchicalPoissonRegressionIndependencePosteriorSampler "
          "constructor.");
    }
  }

  double HPRIPS::logpri() const {
    const SpdMatrix &siginv(data_parent_model()->siginv());
    double ans = 0;
    for (int i = 0; i < nrow(siginv); ++i) {
      ans += siginv_priors_[i]->logp(siginv(i, i));
    }
    return ans;
  }

  void HPRIPS::draw_mu_and_sigma_given_beta() {
    mu_given_beta_sampler_->draw();
    sigma_given_beta_sampler_->draw();
  }

  void HPRIPS::draw_sigma_given_zero_mean_sufficient_statistics() {
    zero_mean_sigma_sampler_->draw();
    data_parent_model()->set_siginv(zero_mean_random_effect_model()->siginv());
  }

}  // namespace BOOM
