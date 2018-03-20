// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#include "Models/Mixtures/PosteriorSamplers/DirichletProcessCollapsedGibbsSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef DirichletProcessCollapsedGibbsSampler DPCGS;
  }

  DPCGS::DirichletProcessCollapsedGibbsSampler(
      ConjugateDirichletProcessMixtureModel *model, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), model_(model) {
    cluster_membership_probabilities_.reserve(20);
  }

  void DPCGS::draw() {
    collapsed_gibbs_update();
    // TODO:  Add a Jain-Neal split-merge update.
    // conjugate_split_merge_update();
  }

  double DPCGS::logpri() const { return model_->base_distribution()->logpri(); }

  void DPCGS::collapsed_gibbs_update() {
    draw_marginal_cluster_membership_indicators();
    draw_parameters_given_cluster_membership();
  }

  void DPCGS::draw_parameters_given_cluster_membership() {
    for (int i = 0; i < model_->number_of_components(); ++i) {
      model_->base_distribution()->draw_model_parameters(*model_->component(i));
    }
  }

  void DPCGS::draw_marginal_cluster_membership_indicators() {
    const std::vector<Ptr<Data>> &data(model_->dat());
    for (int i = 0; i < data.size(); ++i) {
      Ptr<Data> dp = data[i];
      model_->remove_data_from_cluster(dp);
      const Vector &prob = marginal_cluster_membership_probabilities(dp);
      int cluster_number = rmulti_mt(rng(), prob);
      model_->assign_data_to_cluster(dp, cluster_number, rng());
    }
  }

  const Vector &DPCGS::marginal_cluster_membership_probabilities(
      const Ptr<Data> &dp) {
    cluster_membership_probabilities_.resize(model_->number_of_components() +
                                             1);
    int sample_size = model_->dat().size();
    double log_normalizing_constant =
        log(sample_size - 1 + model_->concentration_parameter());
    for (int c = 0; c < model_->number_of_components(); ++c) {
      cluster_membership_probabilities_[c] =
          log(model_->cluster_count(c)) - log_normalizing_constant +
          model_->log_marginal_density(dp, c);
    }
    cluster_membership_probabilities_.back() =
        model_->log_concentration_parameter() - log_normalizing_constant +
        model_->log_marginal_density(dp, -1);
    cluster_membership_probabilities_.normalize_logprob();
    return cluster_membership_probabilities_;
  }

}  // namespace BOOM
