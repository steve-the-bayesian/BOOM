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
#ifndef BOOM_ZERO_INFLATED_LOGNORMAL_REGRESSION_SAMPLER_HPP_
#define BOOM_ZERO_INFLATED_LOGNORMAL_REGRESSION_SAMPLER_HPP_

#include <set>

#include "Models/GammaModel.hpp"
#include "Models/Glm/BinomialLogitModel.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialLogitCompositeSpikeSlabSampler.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/Glm/ZeroInflatedLognormalRegression.hpp"
#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  class ZeroInflatedLognormalRegressionPosteriorSampler
      : public PosteriorSampler {
   public:
    ZeroInflatedLognormalRegressionPosteriorSampler(
        ZeroInflatedLognormalRegressionModel *model,
        const Ptr<VariableSelectionPrior> &regression_spike,
        const Ptr<MvnGivenScalarSigmaBase> &regression_slab,
        const Ptr<GammaModelBase> &siginv_prior,
        const Ptr<VariableSelectionPrior> &logit_spike,
        const Ptr<MvnBase> &logit_slab,
        RNG &seeding_rng = BOOM::GlobalRng::rng);

    // Model selection is on by default.  allow_model_selection(false)
    // turns it off.  allow_model_selection(true) turns it back on
    // again.  This function just affects whether the sampler can
    // change the included / excluded status of individual
    // coefficients.  It does not (for example) force currently
    // excluded coefficients to be included.
    void allow_model_selection(bool tf);

    void draw() override;
    double logpri() const override;

    // An observer method to be called when the data in *model
    // changes.
    void invalidate_latent_data() { data_is_current_ = false; }

    void set_regression_slab(
        const Ptr<MvnGivenScalarSigmaBase> &regression_slab) {
      regression_slab_ = regression_slab;
      regression_sampler_->set_slab(regression_slab_);
    }
    void set_regression_spike(
        const Ptr<VariableSelectionPrior> &regression_spike) {
      regression_spike_ = regression_spike;
      regression_sampler_->set_spike(regression_spike_);
    }

    void set_logit_slab(const Ptr<MvnBase> &logit_slab) {
      logit_slab_ = logit_slab;
      logit_sampler_->set_slab(logit_slab_);
    }

    void set_logit_spike(const Ptr<VariableSelectionPrior> &logit_spike) {
      logit_spike_ = logit_spike;
      logit_sampler_->set_spike(logit_spike_);
    }

    // Access to the underlying models is allowed so that you can do
    // clever data augmentation.
    Ptr<RegressionModel> regression_model() { return regression_model_; }

    Ptr<BinomialLogitModel> logit_model() { return nonzero_; }

    // Normally the sampler will check to see if the data in the model
    // has not changed.  If it has, the data associated with the
    // component models is removed and regenerated.
    //
    // One might wish turn off data checking as a minor speed
    // optimization, or as part of an "expert interface" data
    // augmentation.
    void prevent_data_checking() { check_data_ = false; }

    void find_posterior_mode(double epsilon = 1e-5) override;
    bool can_find_posterior_mode() const override { return true; }

    bool posterior_mode_found() const { return posterior_mode_found_; }

    // Check that the data being managed by the component models
    // (regression_model_ and nonzero_) corresponds to the data
    // in model_.
    void ensure_latent_data();

    double regression_sample_size() const;
    double logit_sample_size() const;

   private:
    ZeroInflatedLognormalRegressionModel *model_;
    Ptr<RegressionModel> regression_model_;
    Ptr<BinomialLogitModel> nonzero_;
    Ptr<VariableSelectionPrior> regression_spike_;
    Ptr<MvnGivenScalarSigmaBase> regression_slab_;
    Ptr<GammaModelBase> siginv_prior_;
    Ptr<VariableSelectionPrior> logit_spike_;
    Ptr<MvnBase> logit_slab_;

    Ptr<BregVsSampler> regression_sampler_;
    Ptr<BinomialLogitCompositeSpikeSlabSampler> logit_sampler_;

    bool data_is_current_;
    bool check_data_;
    std::set<Ptr<RegressionData>> observed_data_;

    bool posterior_mode_found_;
  };

}  // namespace BOOM

#endif  //  BOOM_ZERO_INFLATED_LOGNORMAL_REGRESSION_SAMPLER_HPP_
