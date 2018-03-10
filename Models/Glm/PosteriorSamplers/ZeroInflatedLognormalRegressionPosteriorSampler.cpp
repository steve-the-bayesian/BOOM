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

#include "Models/Glm/PosteriorSamplers/ZeroInflatedLognormalRegressionPosteriorSampler.hpp"
#include <functional>
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef ZeroInflatedLognormalRegressionPosteriorSampler ZILRP;
  }  // namespace

  ZILRP::ZeroInflatedLognormalRegressionPosteriorSampler(
      ZeroInflatedLognormalRegressionModel *model,
      const Ptr<VariableSelectionPrior> &regression_spike,
      const Ptr<MvnGivenScalarSigmaBase> &regression_slab,
      const Ptr<GammaModelBase> &siginv_prior,
      const Ptr<VariableSelectionPrior> &logit_spike,
      const Ptr<MvnBase> &logit_slab, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        regression_model_(new RegressionModel(
            model->regression_coefficient_ptr(), model->sigsq_prm())),
        nonzero_(new BinomialLogitModel(model->logit_coefficient_ptr())),
        regression_spike_(regression_spike),
        regression_slab_(regression_slab),
        siginv_prior_(siginv_prior),
        logit_spike_(logit_spike),
        logit_slab_(logit_slab),
        regression_sampler_(new BregVsSampler(regression_model_.get(),
                                              regression_slab_, siginv_prior_,
                                              regression_spike_, seeding_rng)),
        logit_sampler_(new BinomialLogitCompositeSpikeSlabSampler(
            nonzero_.get(), logit_slab_, logit_spike_,
            3,    // clt_threshold
            3.0,  // t degrees of freedom
            10,   // max_tim_chunk_size
            1,    // max_rwm_chunk_size
            1.0,  // rwm_variance_scale_factor
            seeding_rng)),
        data_is_current_(false),
        check_data_(true),
        posterior_mode_found_(false) {
    regression_model_->set_method(regression_sampler_);
    nonzero_->set_method(logit_sampler_);
  }

  void ZILRP::allow_model_selection(bool do_model_selection) {
    if (do_model_selection) {
      regression_sampler_->allow_model_selection();
    } else {
      regression_sampler_->suppress_model_selection();
    }
    logit_sampler_->allow_model_selection(do_model_selection);
  }

  void ZILRP::draw() {
    ensure_latent_data();
    regression_model_->sample_posterior();
    nonzero_->sample_posterior();
  }

  double ZILRP::logpri() const {
    return regression_sampler_->logpri() + logit_sampler_->logpri();
  }

  void ZILRP::ensure_latent_data() {
    if (!check_data_) {
      return;
    }
    std::function<void(void)> observer = [this]() {
      this->invalidate_latent_data();
    };

    if (!data_is_current_) {
      regression_model_->clear_data();
      regression_model_->suf()->combine(model_->suf());

      nonzero_->clear_data();
      model_->add_observer(observer);
      for (int i = 0; i < model_->dat().size(); ++i) {
        Ptr<RegressionData> data_point = model_->dat()[i];
        if (observed_data_.count(data_point) == 0) {
          data_point->add_observer(observer);
          observed_data_.insert(data_point);
        }
        NEW(BinomialRegressionData, nonzero_data)
        (data_point->y() > model_->zero_threshold(), 1, data_point->Xptr());
        nonzero_->add_data(nonzero_data);
      }
    }
    data_is_current_ = true;
  }

  double ZILRP::regression_sample_size() const {
    return regression_model_->suf()->n();
  }

  double ZILRP::logit_sample_size() const {
    double ans = 0;
    for (const auto &dp : nonzero_->dat()) {
      ans += dp->n();
    }
    return ans;
  }

  void ZILRP::find_posterior_mode(double) {
    posterior_mode_found_ = false;
    regression_sampler_->find_posterior_mode();
    logit_sampler_->find_posterior_mode();
    posterior_mode_found_ = logit_sampler_->posterior_mode_found();
  }

}  // namespace BOOM
