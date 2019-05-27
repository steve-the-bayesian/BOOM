/*
  Copyright (C) 2018 Steven L. Scott

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

#include "Models/StateSpace/PosteriorSamplers/SharedLocalLevelPosteriorSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using SLLPS = SharedLocalLevelPosteriorSampler;
  }  // namespace

  SLLPS::SharedLocalLevelPosteriorSampler(
      SharedLocalLevelStateModel *model,
      const std::vector<Ptr<MvnBase>> &slabs,
      const std::vector<Ptr<VariableSelectionPrior>> &spikes,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        slabs_(slabs),
        spikes_(spikes)
  {
    // Check that spikes and slabs are the right size.
    if (spikes.size() != model_->nseries()) {
      report_error("Number of spike priors does not match number of series.");
    }
    for (int i = 0; i < spikes.size(); ++i) {
      if (spikes[i]->potential_nvars() != model_->state_dimension()) {
        report_error("At least one spike prior expects the wrong state size.");
      }
    }
    
    if (slabs.size() != model_->nseries()) {
      report_error("Number of slab priors does not match number of series.");
    }
    for (int i = 0; i < slabs.size(); ++i) {
      if (slabs[i]->dim() != model_->state_dimension()) {
        report_error("At least one slab prior expects the wrong state size.");
      }
    }

    // Use the spikes to enforce the constraint on the coefficients.
    Matrix coefficients = model_->coefficient_model()->Beta().transpose();
    for (int i = 0; i < spikes_.size(); ++i) {
      Selector inclusion_indicator(model_->state_dimension(), true);
      for (int j = i + 1; j < model_->state_dimension(); ++j) {
        spikes_[i]->set_prior_inclusion_probability(j, 0.0);
        coefficients(i, j) = 0.0;
        inclusion_indicator.drop(j);
      }
      inclusion_indicators_.push_back(inclusion_indicator);
    }
    model_->coefficient_model()->set_Beta(coefficients.transpose());

    // Set the innovation variances to 1, for identifiability.
    for (int i = 0; i < model_->state_dimension(); ++i) {
      model_->innovation_model(i)->set_sigsq(1.0);
    }
    
    // Build the samplers.
    for (int i = 0; i < spikes_.size(); ++i) {
      samplers_.push_back(SpikeSlabSampler(nullptr, slabs_[i], spikes_[i]));
    }
  }

  //===========================================================================
  double SLLPS::logpri() const {
    double ans = 0;
    const Matrix &transposed_coefficients(
        model_->coefficient_model()->Beta());
    
    for (int i = 0; i < inclusion_indicators_.size(); ++i) {
      ans += spikes_[i]->logp(inclusion_indicators_[i]);
      if (!std::isfinite(ans)) {
        return ans;
      }
      ans += dmvn(
          inclusion_indicators_[i].select(transposed_coefficients.col(i)),
          inclusion_indicators_[i].select(slabs_[i]->mu()),
          inclusion_indicators_[i].select(slabs_[i]->siginv()),
          true);
    }
    return ans;
  }

  //===========================================================================
  void SLLPS::draw() {
    Matrix coefficients = model_->coefficient_model()->Beta().transpose();
    WeightedRegSuf suf(model_->number_of_factors());
    const MvRegSuf &mvsuf(*model_->coefficient_model()->suf());
    for (int i = 0; i < slabs_.size(); ++i) {
      suf.reset(mvsuf.xtx(),
                mvsuf.xty().col(i),
                mvsuf.yty()(i, i),
                mvsuf.n(),
                mvsuf.n(),
                0.0);
      
      samplers_[i].draw_inclusion_indicators(
          rng(), inclusion_indicators_[i], suf);
      Vector row = coefficients.row(i);
      samplers_[i].draw_coefficients_given_inclusion(
          rng(), row, inclusion_indicators_[i], suf, 1.0);
      coefficients.row(i) = row;
    }
    model_->coefficient_model()->set_Beta(coefficients.transpose());
  }

  //===========================================================================
  void SharedLocalLevelPosteriorSampler::limit_model_selection(int max_flips) {
    for (int i = 0; i < samplers_.size(); ++i) {
      samplers_[i].limit_model_selection(max_flips);
    }
  }
  
}  // namespace BOOM
