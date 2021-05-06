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

#include "Models/PosteriorSamplers/IndependentMvnConjSampler.hpp"
#include "Models/ChisqModel.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "distributions/trun_gamma.hpp"

namespace BOOM {

  IndependentMvnConjSampler::IndependentMvnConjSampler(
      IndependentMvnModel *model, const Vector &mean_guess,
      const Vector &mean_sample_size, const Vector &sd_guess,
      const Vector &sd_sample_size, const Vector &sigma_upper_limit,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        mean_prior_guess_(mean_guess),
        mean_prior_sample_size_(mean_sample_size),
        prior_ss_(sd_guess * sd_guess * sd_sample_size),
        prior_df_(sd_sample_size) {
    check_sizes(sigma_upper_limit);
    int dim = mean_guess.size();
    for (int i = 0; i < dim; ++i) {
      GenericGaussianVarianceSampler sigsq_sampler(
          new ChisqModel(sd_sample_size[i], sd_guess[i]), sigma_upper_limit[i]);
      sigsq_samplers_.push_back(sigsq_sampler);
    }
  }

  IndependentMvnConjSampler::IndependentMvnConjSampler(
      IndependentMvnModel *model, double mean_guess, double mean_sample_size,
      double sd_guess, double sd_sample_size, double sigma_upper_limit,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        mean_prior_guess_(model->dim(), mean_guess),
        mean_prior_sample_size_(model->dim(), mean_sample_size),
        prior_ss_(model->dim(), sd_guess * sd_guess * sd_sample_size),
        prior_df_(model->dim(), sd_sample_size) {
    const Ptr<ChisqModel> &siginv_prior(
        new ChisqModel(sd_sample_size, sd_guess));
    for (int i = 0; i < model_->dim(); ++i) {
      GenericGaussianVarianceSampler sigsq_sampler(siginv_prior,
                                                   sigma_upper_limit);
      sigsq_samplers_.push_back(sigsq_sampler);
    }
  }

  IndependentMvnConjSampler *IndependentMvnConjSampler::clone_to_new_host(
      Model *new_host) const {
    Vector sigma_upper_limit;
    for (const auto &sampler : sigsq_samplers_) {
      sigma_upper_limit.push_back(sampler.sigma_max());
    }
    return new IndependentMvnConjSampler(
        dynamic_cast<IndependentMvnModel *>(new_host),
        mean_prior_guess_,
        mean_prior_sample_size_,
        sqrt(prior_ss_ / prior_df_),
        prior_df_,
        sigma_upper_limit,
        rng());
  }

  double IndependentMvnConjSampler::logpri() const {
    int dim = model_->dim();
    const Vector &mu(model_->mu());
    const Vector &sigsq(model_->sigsq());
    double ans = 0;
    for (int i = 0; i < dim; ++i) {
      ans += sigsq_samplers_[i].log_prior(sigsq[i]);
      ans += dnorm(mu[i], mean_prior_guess_[i],
                   sqrt(sigsq[i] / mean_prior_sample_size_[i]), true);
    }
    return ans;
  }

  void IndependentMvnConjSampler::check_sizes(const Vector &sigma_upper_limit) {
    check_vector_size(mean_prior_guess_, "mean_prior_guess_");
    check_vector_size(mean_prior_sample_size_, "mean_prior_sample_size_");
    check_vector_size(prior_ss_, "prior_ss_");
    check_vector_size(prior_df_, "prior_df_");
    check_vector_size(sigma_upper_limit, "sigma_upper_limit");
  }

  void IndependentMvnConjSampler::check_vector_size(const Vector &v,
                                                    const char *vector_name) {
    if (v.size() != model_->dim()) {
      ostringstream err;
      err << "One of the elements of IndependentMvnConjSampler does not "
          << "match the model dimension" << endl
          << vector_name << endl
          << v << endl;
      report_error(err.str());
    }
  }

  void IndependentMvnConjSampler::draw() {
    int dim = model_->dim();
    const IndependentMvnSuf &suf(*(model_->suf()));
    Vector mu(dim);
    Vector sigsq(dim);
    for (int i = 0; i < dim; ++i) {
      double n = suf.n();
      double ybar = suf.ybar(i);
      double v = suf.sample_var(i);

      double kappa = mean_prior_sample_size_[i];
      double mu0 = mean_prior_guess_[i];

      double mu_hat = (n * ybar + kappa * mu0) / (n + kappa);
      double ss = (n - 1) * v + n * kappa * pow(ybar - mu0, 2) / (n + kappa);
      sigsq[i] = sigsq_samplers_[i].draw(rng(), n, ss);
      v = sigsq[i] / (n + kappa);
      mu[i] = rnorm_mt(rng(), mu_hat, sqrt(v));
    }
    model_->set_mu(mu);
    model_->set_sigsq(sigsq);
  }
}  // namespace BOOM
