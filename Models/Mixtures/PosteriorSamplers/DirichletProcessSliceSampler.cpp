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

#include "Models/Mixtures/PosteriorSamplers/DirichletProcessSliceSampler.hpp"
#include "cpputil/lse.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef DirichletProcessSliceSampler DPSS;
    typedef DirichletProcessMixtureModel DPMM;
    typedef DirichletProcessMixtureComponent DpMixtureComponent;
    //    const bool print_mcmc_details = false;
  }  // namespace

  DPSS::DirichletProcessSliceSampler(DirichletProcessMixtureModel *model,
                                     int initial_clusters, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        mixing_weight_importance_ratio_(.2),
        log_mixing_weight_importance_ratio_(
            log(mixing_weight_importance_ratio_)),
        max_clusters_(model_->number_of_observations(), initial_clusters),
        global_max_clusters_(initial_clusters),
        first_time_(true),
        split_merge_strategy_(nullptr) {}

  //----------------------------------------------------------------------
  void DPSS::draw() {
    if (first_time_) {
      randomly_allocate_data_to_clusters();
      first_time_ = false;
    }
    draw_parameters_given_mixture_indicators();
    draw_stick_fractions_given_mixture_indicators();
    draw_slice_variables_given_mixture_indicators();
    draw_mixture_indicators();
    shuffle_order();
    split_merge_move();
  }

  //----------------------------------------------------------------------
  double DPSS::logpri() const { return model_->base_distribution()->logpri(); }
  //----------------------------------------------------------------------
  double DPSS::mixing_weight_importance(int cluster) const {
    return exp(log_mixing_weight_importance(cluster));
  }
  //----------------------------------------------------------------------
  double DPSS::log_mixing_weight_importance(int cluster) const {
    return (cluster)*log_mixing_weight_importance_ratio_;
  }
  //----------------------------------------------------------------------
  void DPSS::set_mixing_weight_importance_ratio(double value) {
    mixing_weight_importance_ratio_ = value;
    log_mixing_weight_importance_ratio_ = log(value);
  }
  //----------------------------------------------------------------------
  void DPSS::draw_parameters_given_mixture_indicators() {
    for (int i = 0; i < model_->number_of_components(); ++i) {
      model_->base_distribution()->draw_model_parameters(*model_->component(i));
    }
  }
  //----------------------------------------------------------------------
  void DPSS::draw_stick_fractions_given_mixture_indicators() {
    int nc = model_->number_of_components();
    if (nc == 0) {
      model_->set_stick_fractions(Vector(0));
      return;
    }
    Vector cluster_counts(nc);
    Vector cumulative_cluster_counts(nc);
    cluster_counts[0] = cumulative_cluster_counts[0] = model_->cluster_count(0);
    for (int i = 1; i < nc; ++i) {
      cluster_counts[i] = model_->cluster_count(i);
      cumulative_cluster_counts[i] =
          cumulative_cluster_counts[i - 1] + cluster_counts[i];
    }
    Vector stick_fractions(nc);
    double sample_size = cumulative_cluster_counts.back();
    for (int i = 0; i < nc; ++i) {
      double a = 1 + cluster_counts[i];
      double b = model_->concentration_parameter() + sample_size -
                 cumulative_cluster_counts[i];
      stick_fractions[i] = rbeta_mt(rng(), a, b);
    }
    model_->set_stick_fractions(stick_fractions);
  }
  //----------------------------------------------------------------------
  void DPSS::draw_slice_variables_given_mixture_indicators() {
    int nobs = model_->number_of_observations();
    max_clusters_.resize(nobs);
    global_max_clusters_ = 0;
    for (int i = 0; i < nobs; ++i) {
      double slice = runif_mt(
          rng(), 0, mixing_weight_importance(model_->cluster_indicator(i)));
      max_clusters_[i] = find_max_number_of_clusters(slice);
      global_max_clusters_ =
          std::max<int>(global_max_clusters_, max_clusters_[i]);
    }
  }

  //----------------------------------------------------------------------
  // The draw of the mixture indicators is conditional on the slice variables.
  // The probability mass is over indices k such that
  //
  //  I(xi[k] > u[i])  * (mixing_weight[k] / xi[k]) * pdf(data[i] | cluster k)
  //
  // The indicator function is the interesting part of this expression.  xi[k] =
  // r^k for some ratio r (given by mixing_weight_importance_ratio_).
  // Satisfying r^k > u implies k * log(r) > log(u).  Dividing both sides by the
  // (negative) value log(r) gives k < log(u) / log(r).
  void DPSS::draw_mixture_indicators() {
    while (model_->number_of_components() < global_max_clusters_) {
      model_->add_empty_cluster(rng());
    }
    for (int i = 0; i < model_->number_of_components(); ++i) {
      model_->component(i)->clear_data();
    }
    const std::vector<Ptr<Data>> &data(model_->dat());
    int sample_size = data.size();
    Vector mixing_weights = model_->mixing_weights();
    Vector log_mixing_weights = log(mixing_weights);
    Vector workspace;
    for (int i = 0; i < sample_size; ++i) {
      Ptr<Data> data_point = data[i];
      workspace.resize(max_clusters_[i]);
      for (int c = 0; c < max_clusters_[i]; ++c) {
        workspace[c] = log_mixing_weights[c] +
                       model_->component(c)->pdf(data_point.get(), true) -
                       log_mixing_weight_importance(c);
      }
      workspace.normalize_logprob();
      int new_mixture_indicator = rmulti_mt(rng(), workspace);
      model_->assign_data_to_cluster(data_point, new_mixture_indicator, rng());
    }
    model_->remove_all_empty_clusters();
  }
  //----------------------------------------------------------------------
  int DPSS::find_max_number_of_clusters(double slice_variable) const {
    int ans =
        lround(ceil(log(slice_variable) / log_mixing_weight_importance_ratio_));
    if (ans <= 0) {
      report_error("Found an impossible value for max_number_of_clusters.");
    }
    return ans;
  }

  //----------------------------------------------------------------------
  void DPSS::shuffle_order() {
    MoveTimer timer = move_accounting_.start_time("Shuffle");
    int number_of_components = model_->number_of_components();
    if (number_of_components <= 1) return;
    int component_index = runif_mt(rng(), 0, number_of_components - 1);
    int destination = component_index + (runif_mt(rng()) < .5 ? -1 : 1);
    if (destination < 0 || destination >= number_of_components) {
      move_accounting_.record_rejection("Shuffle");
      return;
    }

    // Construct the MH acceptance ratio, which depends entirely on mixing
    // weights, because everything else cancels.
    Vector original_mixing_weights = model_->mixing_weights();
    original_mixing_weights.pop_back();
    Vector shuffled_mixing_weights = original_mixing_weights;
    shuffled_mixing_weights.shift_element(component_index, destination);

    double concentration = model_->concentration_parameter();
    double log_MH_alpha =
        DPMM::dstick(shuffled_mixing_weights, concentration, true) -
        DPMM::dstick(original_mixing_weights, concentration, true);

    // Make the MH decision.
    double logu = log(runif_mt(rng()));
    if (logu < log_MH_alpha) {
      model_->shift_cluster(component_index, destination);
      move_accounting_.record_acceptance("Shuffle");
    } else {
      move_accounting_.record_rejection("Shuffle");
    }
  }

  //----------------------------------------------------------------------
  void DPSS::set_split_merge_strategy(SplitMerge::ProposalStrategy *strategy) {
    split_merge_strategy_.reset(strategy);
  }

  // TODO: only do split_merge with some probability, because it is
  // expensive.
  void DPSS::split_merge_move() {
    if (!split_merge_strategy_) return;
    int n = model_->number_of_observations();
    if (n <= 1) {
      // The model has either zero or one observations.  There is nothing to be
      // done here.
      return;
    }
    int first_index = random_int_mt(rng(), 0, n - 1);
    int second_index = first_index;
    while (second_index == first_index) {
      second_index = random_int_mt(rng(), 0, n - 1);
    }
    if (model_->cluster_indicator(first_index) !=
        model_->cluster_indicator(second_index)) {
      attempt_merge_move(first_index, second_index);
    } else {
      attempt_split_move(first_index, second_index);
    }
    // The order of the mixture components may have changed as a result of the
    // split or merge, so refresh the stick fractions to preserve model
    // invariants.
    draw_stick_fractions_given_mixture_indicators();
  }

  //----------------------------------------------------------------------
  void DPSS::attempt_merge_move(int data_index_1, int data_index_2) {
    // if (print_mcmc_details) {
    //   std::cout << "Attempting to merge clusters "
    //             << model_->cluster_indicator(data_index_1) << " and "
    //             << model_->cluster_indicator(data_index_2) << "." << std::endl;
    // }
    MoveTimer timer = move_accounting_.start_time("Merge");
    SplitMerge::Proposal proposal =
        split_merge_strategy_->propose_merge(data_index_1, data_index_2, rng());
    double log_MH_alpha = log_MH_probability(proposal);
    double logu = log(runif_mt(rng(), 0, 1));
    if (logu < log_MH_alpha) {
      model_->accept_split_merge_proposal(proposal);
      move_accounting_.record_acceptance("Merge");
      // if (print_mcmc_details) {
      //   std::cout << "Merge successful with log alpha = " << log_MH_alpha << "."
      //             << std::endl;
      // }
    } else {
      // Proposal failed, leave things as they are.
      move_accounting_.record_rejection("Merge");
      // if (print_mcmc_details) {
      //   std::cout << "Merge failed with log alpha = " << log_MH_alpha << "."
      //             << std::endl;
      // }
    }
  }
  //----------------------------------------------------------------------
  void DPSS::attempt_split_move(int data_index_1, int data_index_2) {
    // if (print_mcmc_details) {
    //   std::cout << "Attempting to split cluster "
    //             << model_->cluster_indicator(data_index_1) << std::endl;
    // }
    MoveTimer time = move_accounting_.start_time("Split");
    SplitMerge::Proposal proposal =
        split_merge_strategy_->propose_split(data_index_1, data_index_2, rng());
    double log_MH_alpha = log_MH_probability(proposal);
    double logu = log(runif_mt(rng(), 0, 1));
    if (logu < log_MH_alpha) {
      model_->accept_split_merge_proposal(proposal);
      move_accounting_.record_acceptance("Split");
      // if (print_mcmc_details) {
      //   std::cout << "Split was successful with log_alpha = " << log_MH_alpha
      //             << "." << std::endl;
      // }
    } else {
      move_accounting_.record_rejection("Split");
      // if (print_mcmc_details) {
      //   std::cout << "Split failed with log_alpha = " << log_MH_alpha << "."
      //             << std::endl;
      // }
    }
  }
  //----------------------------------------------------------------------
  // Compute the MH acceptance probability for the proposal.
  double DPSS::log_MH_probability(const SplitMerge::Proposal &proposal) const {
    double log_likelihood_ratio = proposal.split1()->log_likelihood() +
                                  proposal.split2()->log_likelihood() -
                                  proposal.merged()->log_likelihood();

    double log_prior_ratio =
        model_->base_distribution()->log_prior_density(*proposal.split1()) +
        model_->base_distribution()->log_prior_density(*proposal.split2()) -
        model_->base_distribution()->log_prior_density(*proposal.merged()) -
        model_->base_distribution()->log_prior_density(*proposal.empty());

    double log_allocation_probability_ratio =
        proposal.split1()->number_of_observations() *
            log(proposal.split1_mixing_weight()) +
        proposal.split2()->number_of_observations() *
            log(proposal.split2_mixing_weight()) -
        proposal.merged()->number_of_observations() *
            log(proposal.merged_mixing_weight());

    double alpha = model_->concentration_parameter();
    double log_mixing_weight_prior_ratio =
        DPMM::dstick(proposal.split_mixing_weights(), alpha, true) -
        DPMM::dstick(proposal.merged_mixing_weights(), alpha, true);

    double log_target_density_ratio = log_likelihood_ratio + log_prior_ratio +
                                      log_allocation_probability_ratio +
                                      log_mixing_weight_prior_ratio;
    // if (print_mcmc_details) {
    //   std::cout << "positive numbers favor splits" << endl
    //             << "   log likelihood ratio:       " << log_likelihood_ratio
    //             << std::endl
    //             << "   log_prior_ratio:            " << log_prior_ratio
    //             << std::endl
    //             << "   log_prior_allocation_ratio: "
    //             << log_allocation_probability_ratio << std::endl
    //             << "   log stick ratio:            "
    //             << log_mixing_weight_prior_ratio << std::endl;
    // }
    double log_MH_ratio = log_target_density_ratio -
                          proposal.log_split_to_merge_probability_ratio();

    return log_MH_ratio * (proposal.is_merge() ? -1.0 : 1.0);
  }

  void DPSS::randomly_allocate_data_to_clusters() {
    while (model_->number_of_components() < global_max_clusters_) {
      model_->add_empty_cluster(rng());
    }
    for (int i = 0; i < model_->number_of_components(); ++i) {
      model_->component(i)->clear_data();
    }
    const std::vector<Ptr<Data>> &data(model_->dat());
    for (int i = 0; i < data.size(); ++i) {
      int cluster = random_int_mt(rng(), 0, global_max_clusters_ - 1);
      model_->assign_data_to_cluster(data[i], cluster, rng());
    }
  }

}  // namespace BOOM
