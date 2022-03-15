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

#include "Models/Mixtures/PosteriorSamplers/DirichletProcessMvnCollapsedGibbsSampler.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "math/special_functions.hpp"

namespace BOOM {
  namespace {
    typedef DirichletProcessMvnCollapsedGibbsSampler DPMCGS;
  }  // namespace

  DPMCGS::DirichletProcessMvnCollapsedGibbsSampler(
      DirichletProcessMvnModel *model,
      const Ptr<MvnGivenSigma> &mean_base_measure,
      const Ptr<WishartModel> &precision_base_measure, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        mean_base_measure_(mean_base_measure),
        precision_base_measure_(precision_base_measure),
        prior_(mean_base_measure_.get(), precision_base_measure_.get()),
        posterior_(mean_base_measure_.get(), precision_base_measure_.get()) {}

  double DPMCGS::logpri() const {
    report_error(
        "Calling logpri for a Dirichlet process mixture really "
        "doesn't make a lot of sense");
    return 0;
  }

  void DPMCGS::draw() {
    // The cluster membership indicators should be drawn before the parameters.
    // The cluster membership indicators are drawn from their marginal
    // distribution, integrating over the parameters.  The parameters are then
    // drawn from their conditional distribution given the indicators.
    // Reversing the order leads to an incoherent joint distribution between
    // parameters and states.
    draw_cluster_membership_indicators();
    draw_parameters();
  }

  void DPMCGS::draw_cluster_membership_indicators() {
    const std::vector<Ptr<VectorData> > &data(model_->dat());
    model_->initialize_cluster_membership_probabilities();
    if (model_->cluster_indicators().empty()) {
      // If this is the first time we've been down this code path then
      // cluster_indicators_ will be empty.  Fill it with -1's which
      // are the signal that a data point is currently unassigned.
      model_->initialize_cluster_indicators(data.size());
      for (int i = 0; i < data.size(); ++i) {
        // Then assign each data point to cluster 0.
        assign_data_to_cluster(data[i]->value(), 0);
        model_->set_cluster_indicator(i, 0);
      }
    }

    for (int i = 0; i < data.size(); ++i) {
      const Vector &y(data[i]->value());
      remove_data_from_cluster(y, model_->cluster_indicators(i));
      model_->set_cluster_indicator(i, -1);
      Vector prob = cluster_membership_probability(y);
      int cluster_number = rmulti_mt(rng(), prob);
      model_->assign_data_to_cluster(y, cluster_number);
      model_->set_cluster_indicator(i, cluster_number);
    }

    model_->initialize_cluster_membership_probabilities();
    for (int i = 0; i < data.size(); ++i) {
      Vector prob = cluster_membership_probability(data[i]->value());
      model_->set_cluster_membership_probabilities(i, prob);
    }
  }

  void DPMCGS::draw_parameters() {
    for (int i = 0; i < model_->number_of_clusters(); ++i) {
      posterior_.compute_mvn_posterior(*model_->cluster(i).suf());
      SpdMatrix Siginv = rWish_mt(rng(), posterior_.variance_sample_size(),
                                  posterior_.sum_of_squares().inv());
      Vector mu = rmvn_ivar_mt(rng(), posterior_.mean(),
                               posterior_.mean_sample_size() * Siginv);
      model_->set_component_params(i, mu, Siginv);
    }
  }

  Vector DPMCGS::cluster_membership_probability(const Vector &y) {
    Vector ans(model_->number_of_clusters() + 1);
    int n = model_->dat().size();
    // NOTE: there is an optimization opportunity here, because we
    // will compute log(n-1+alpha) for every data point,
    // number_of_clusters+1 times, every MCMC iteration.  Currently
    // both alpha and n are fixed, so this could be precomputed.
    for (int i = 0; i < model_->number_of_clusters(); ++i) {
      const MvnSuf &suf(*model_->cluster(i).suf());
      ans[i] = log(suf.n()) - log(n - 1 + model_->alpha()) +
               log_marginal_density(y, suf);
    }
    ans.back() = log(model_->alpha()) - log(n - 1 + model_->alpha()) +
                 log_marginal_density(y, empty_suf_);

    ans.normalize_logprob();
    return ans;
  }

  // For the math, see Murphy (Machine Learning: A probabilistic
  // perspective) page 161 (eq: 5.29).  We omit any factors that only
  // depend on the 'sample size' (which is always 1), or other
  // quantities that will be the same for all elements of the cluster
  // membership probability vector.  We also rely on heavy
  // cancellation in ratios of the multivariate gamma function.
  double DPMCGS::log_marginal_density(const Vector &y,
                                      const MvnSuf &suf) const {
    prior_.compute_mvn_posterior(suf);
    MvnSuf posterior_suf(suf);
    posterior_suf.update_raw(y);
    posterior_.compute_mvn_posterior(posterior_suf);
    int dim = prior_.mean().size();
    double ans =
        0.5 * dim *
            log(prior_.mean_sample_size() / posterior_.mean_sample_size()) +
        0.5 * prior_.variance_sample_size() * prior_.sum_of_squares().logdet() -
        0.5 * posterior_.variance_sample_size() *
            posterior_.sum_of_squares().logdet() +
        lmultigamma_ratio(prior_.variance_sample_size() / 2.0, 1, dim);
    return ans;
  }

  void DPMCGS::assign_data_to_cluster(const Vector &y, int cluster) {
    model_->assign_data_to_cluster(y, cluster);
  }

  void DPMCGS::remove_data_from_cluster(const Vector &y, int cluster) {
    bool empty = (model_->cluster(cluster).suf()->n() == 1);
    model_->remove_data_from_cluster(y, cluster);
    // If this is the last data point in the cluster, then removing it will make
    // the cluster empty.  Decrement cluster indicators for clusters numbered
    // larger than 'cluster' to account for the fact that 'cluster' no longer
    // exists.
    if (empty) {
      for (int i = 0; i < model_->cluster_indicators().size(); ++i) {
        const int current_indicator = model_->cluster_indicators(i);
        if (current_indicator >= cluster) {
          model_->set_cluster_indicator(i, current_indicator - 1);
        }
      }
    }
  }

}  // namespace BOOM
