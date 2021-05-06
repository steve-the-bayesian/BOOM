// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2014 Steven L. Scott

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

#include "Models/Glm/PosteriorSamplers/PoissonRegressionAuxMixSampler.hpp"
#include "Models/Glm/PosteriorSamplers/poisson_mixture_approximation_table.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef PoissonRegressionAuxMixSampler PRAMS;
    typedef LatentDataSampler<PoissonRegressionDataImputer> Parent;
  }  // namespace

  PoissonRegressionDataImputer::PoissonRegressionDataImputer(
      WeightedRegSuf &global_suf, std::mutex &global_suf_mutex,
      const GlmCoefs *coefficients, RNG *rng, RNG &seeding_rng)
      : SufstatImputeWorker<PoissonRegressionData, WeightedRegSuf>(
            global_suf, global_suf_mutex, rng, seeding_rng),
        coefficients_(coefficients),
        imputer_(new PoissonDataImputer) {}

  // The latent variable scheme imagines the event times of y[i]
  // events from a Poisson process that occur in the interval [0, 1].
  // The maximum of these events, denoted tau[i], is marginally
  // Gamma(y[i],1)/lambda[i], which means u[i] = -log(tau[i]) =
  // log(lambda[i]) + NegLogGamma(y[i], 1).  Given a draw of u[i] we
  // need to express NegLogGamma() as a mixture of normals.
  //
  // The conditional distribution of tau[i], given it is the largest
  // of y[i] events from a Poisson process, is the same as the maximum
  // of y[i] uniform random variables.  That is,
  //
  //       tau[i] | y[i] ~ Beta(y[i], 1)
  //
  // If y[i] = 0 then tau[i] = 0 as well, since no events occurred in
  // [0,1].  It is also necessary to account for the unused portion of
  // [0,1], which is done by sampling the first event after the end of
  // the interval.  The terminal event kappa[i] is marginally
  // exponential, and conditionally truncated exponential with support
  // above 1 - tau[i].
  void PoissonRegressionDataImputer::impute_latent_data_point(
      const PoissonRegressionData &dp, WeightedRegSuf *complete_data_suf,
      RNG &rng) {
    const Vector &x(dp.x());
    double eta = coefficients_->predict(x);
    int y = dp.y();
    double exposure = dp.exposure();
    double internal_neglog_final_event_time;
    double internal_mu;
    double internal_weight;
    double neglog_final_interarrival_time;
    double external_mu;
    double external_weight;
    imputer_->impute(rng, y, exposure, eta, &internal_neglog_final_event_time,
                     &internal_mu, &internal_weight,
                     &neglog_final_interarrival_time, &external_mu,
                     &external_weight);
    if (y > 0) {
      complete_data_suf->add_data(
          x, internal_neglog_final_event_time - internal_mu, internal_weight);
    }
    complete_data_suf->add_data(x, neglog_final_interarrival_time - external_mu,
                                external_weight);
  }

  //======================================================================

  PRAMS::PoissonRegressionAuxMixSampler(PoissonRegressionModel *model,
                                        const Ptr<MvnBase> &prior,
                                        int number_of_imputation_workers,
                                        RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        prior_(prior),
        complete_data_suf_(model_->xdim()),
        first_pass_through_data_(true) {
    set_number_of_workers(number_of_imputation_workers);
  }

  PoissonRegressionAuxMixSampler * PRAMS::clone_to_new_host(
      Model *new_host) const {
    return new PoissonRegressionAuxMixSampler(
        dynamic_cast<PoissonRegressionModel *>(new_host),
        prior_->clone(),
        1,
        rng());
  }

  double PRAMS::logpri() const { return prior_->logp(model_->Beta()); }

  void PRAMS::draw() {
    impute_latent_data();
    draw_beta_given_complete_data();
  }

  void PRAMS::impute_latent_data() {
    Parent::impute_latent_data();
    if (first_pass_through_data_) {
      first_pass_through_data_ = false;
      if (desired_number_of_workers_ > 1) {
        set_number_of_workers(desired_number_of_workers_);
      }
    }
  }

  void PRAMS::draw_beta_given_complete_data() {
    SpdMatrix ivar = prior_->siginv() + complete_data_suf_.xtx();
    Vector ivar_mu = prior_->siginv() * prior_->mu() + complete_data_suf_.xty();
    Vector beta = rmvn_suf_mt(rng(), ivar, ivar_mu);
    model_->set_Beta(beta);
  }

  const WeightedRegSuf &PRAMS::complete_data_sufficient_statistics() const {
    return complete_data_suf_;
  }

  Ptr<PoissonRegressionDataImputer> PRAMS::create_worker(std::mutex &m) {
    return new PoissonRegressionDataImputer(
        complete_data_suf_, m, model_->coef_prm().get(), nullptr, rng());
  }

  void PRAMS::set_number_of_workers(int n) {
    desired_number_of_workers_ = n;
    if (first_pass_through_data_) {
      n = 1;
    }
    Parent::set_number_of_workers(n);
  }

  void PRAMS::clear_latent_data() { complete_data_suf_.clear(); }

  void PRAMS::clear_complete_data_sufficient_statistics() {
    complete_data_suf_.clear();
  }

  void PRAMS::update_complete_data_sufficient_statistics(
      double precision_weighted_sum, double total_precision,
      const Vector &predictors) {
    complete_data_suf_.add_data(
        predictors, precision_weighted_sum / total_precision, total_precision);
  }

  void PRAMS::assign_data_to_workers() {
    BOOM::assign_data_to_workers(model_->dat(), workers());
  }

}  // namespace BOOM
