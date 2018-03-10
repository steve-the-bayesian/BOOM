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

#include "Models/Glm/PosteriorSamplers/NonconjugateRegressionSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef NonconjugateRegressionSampler NRS;
  }

  NRS::NonconjugateRegressionSampler(
      RegressionModel *model, const Ptr<LocationScaleVectorModel> &beta_prior,
      const Ptr<GammaModelBase> &residual_precision_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        beta_prior_(beta_prior),
        residual_precision_prior_(residual_precision_prior),
        residual_variance_sampler_(residual_precision_prior_),
        mh_proposal_(new MvtIndepProposal(Vector(model->xdim()),
                                          SpdMatrix(model->xdim(), 1.0), 3.0)),
        mh_sampler_(beta_log_posterior_callback(), mh_proposal_, &rng()),
        slice_sampler_(beta_log_posterior_callback(), 1.0, false, &rng()) {}

  void NRS::set_slice_sampler_limits(const Vector &lower, const Vector &upper) {
    slice_sampler_.set_limits(lower, upper);
  }

  void NRS::draw() {
    draw_coefficients();
    draw_sigsq();
  }

  void NRS::draw_coefficients() {
    SamplingMethod method = select_sampling_method();
    switch (method) {
      case (METROPOLIS):
        draw_using_mh();
        break;
      case (SLICE):
        draw_using_slice();
        break;
      default:
        report_error("Unknown sampling method.");
    }
  }

  NRS::SamplingMethod NRS::select_sampling_method() {
    int number_of_trials;
    double probability =
        move_accounting_.acceptance_ratio("MH", number_of_trials);
    if (number_of_trials < 100) {
      probability = .5;
    }
    double u = runif_mt(rng());
    return u < probability ? METROPOLIS : SLICE;
  }

  double NRS::logpri() const {
    return beta_prior_->logp(model_->Beta()) +
           residual_precision_prior_->logp(1.0 / model_->sigsq());
  }

  void NRS::draw_sigsq() {
    double sigsq = residual_variance_sampler_.draw(
        rng(), model_->suf()->n(), model_->suf()->relative_sse(model_->coef()));
    model_->set_sigsq(sigsq);
  }

  void NRS::draw_using_mh() {
    MoveTimer timer = move_accounting_.start_time("MH");
    refresh_mh_proposal_distribution();
    Vector beta = mh_sampler_.draw(model_->Beta());
    if (mh_sampler_.last_draw_was_accepted()) {
      move_accounting_.record_acceptance("MH");
    } else {
      move_accounting_.record_rejection("MH");
    }
    model_->set_Beta(beta);
  }

  void NRS::refresh_mh_proposal_distribution() {
    double sigsq = model_->sigsq();
    SpdMatrix posterior_precision =
        beta_prior_->siginv() + model_->suf()->xtx() / sigsq;
    Vector posterior_mean = beta_prior_->siginv() * beta_prior_->mu() +
                            model_->suf()->xty() / sigsq;
    posterior_mean = posterior_precision.solve(posterior_mean);
    mh_proposal_->set_mu(posterior_mean);
    mh_proposal_->set_ivar(posterior_precision);
  }

  void NRS::draw_using_slice() {
    MoveTimer timer = move_accounting_.start_time("slice");
    Vector beta = slice_sampler_.draw(model_->Beta());
    model_->set_Beta(beta);
    move_accounting_.record_acceptance("slice");
  }

  std::function<double(const Vector &)> NRS::beta_log_posterior_callback() {
    return [this](const Vector &beta) {
      return this->model_->log_likelihood(beta, model_->sigsq()) +
             this->beta_prior_->logp(beta);
    };
  }

}  // namespace BOOM
