// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#include "Models/Mixtures/PosteriorSamplers/MvnMetaAnalysisDPMPriorSampler.hpp"
#include "Models/Mixtures/PosteriorSamplers/DirichletProcessMvnCollapsedGibbsSampler.hpp"
#include "Models/PosteriorSamplers/MvnMeanSampler.hpp"

namespace BOOM {

  MvnMetaAnalysisDPMPriorSampler::MvnMetaAnalysisDPMPriorSampler(
      MvnMetaAnalysisDPMPriorModel *model,
      const Ptr<MvnGivenSigma> &mean_base_measure,
      const Ptr<WishartModel> &precision_base_measure, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        mean_base_measure_(mean_base_measure),
        precision_base_measure_(precision_base_measure) {
    DirichletProcessMvnModel *prior = model_->prior_model();
    // Initialize data for prior to the noisy copies.
    for (int i = 0; i < model_->number_of_groups(); ++i) {
      MvnModel *data_model = model_->data_model(i);
      Ptr<VectorData> group_mean = new VectorData(data_model->suf()->ybar());
      prior->add_data(group_mean);
    }
    // Set up sampler for prior.
    prior->clear_methods();
    NEW(DirichletProcessMvnCollapsedGibbsSampler, prior_sampler)
    (prior, mean_base_measure_, precision_base_measure_, rng());
    prior->set_method(prior_sampler);
  }

  double MvnMetaAnalysisDPMPriorSampler::logpri() const {
    return model_->prior_model()->log_likelihood();
  }

  void MvnMetaAnalysisDPMPriorSampler::draw() {
    DirichletProcessMvnModel *prior = model_->prior_model();
    // Update DPM prior given group means.
    prior->sample_posterior();
    // Update group means given DPM prior.
    prior->clear_data();
    for (int i = 0; i < model_->number_of_groups(); ++i) {
      MvnModel *data_model = model_->data_model(i);
      // Get group parameters.
      const Vector current_group_mean(data_model->mu());
      const int current_cluster = prior->cluster_indicators(i);
      const MvnModel &cluster_model = prior->cluster(current_cluster);
      // Draw new group_mean.
      data_model->clear_methods();
      NEW(MvnMeanSampler, data_model_sampler)
      (data_model, cluster_model.mu(), cluster_model.Sigma(), rng());
      data_model->set_method(data_model_sampler);
      data_model->sample_posterior();
      const Vector new_group_mean(data_model->mu());
      Ptr<VectorData> new_group_mean_ptr = new VectorData(new_group_mean);
      prior->add_data(new_group_mean_ptr);
      // Add and remove new data point from cluster sufstats.
      prior->update_cluster(current_group_mean, new_group_mean,
                            current_cluster);
    }
  }

}  // namespace BOOM
