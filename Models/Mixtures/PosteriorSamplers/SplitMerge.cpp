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

#include "Models/Mixtures/PosteriorSamplers/SplitMerge.hpp"
#include "cpputil/lse.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/shift_element.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace SplitMerge {

    namespace {
      typedef SingleObservationSplitStrategy SOSS;
      typedef DirichletProcessMixtureComponent DpMixtureComponent;
      using std::endl;
    }  // namespace

    Proposal::Proposal(ProposalType type, int data_index_1, int data_index_2)
        : type_(type),
          data_index_1_(data_index_1),
          data_index_2_(data_index_2),
          log_split_to_merge_probability_ratio_(
              std::numeric_limits<double>::quiet_NaN()) {}

    void Proposal::check() {
      if (split_mixing_weights_.empty() || merged_mixing_weights_.empty()) {
        report_error("Mixing weights were not set.");
      }
      if (split_mixing_weights_.size() != merged_mixing_weights_.size()) {
        report_error("Mixing weights were set incorrectly.");
      }
      if (std::isnan(log_split_to_merge_probability_ratio_)) {
        report_error("Proposal density was not set.");
      }
      if (!merged_ || !empty_ || !split1_ || !split2_) {
        report_error("Mixture components were not set.");
      }
      const double numerical_fudge_factor = 1e-10;
      if (fabs(merged_mixing_weight() + empty_mixing_weight() -
               split1_mixing_weight() - split2_mixing_weight()) >
          numerical_fudge_factor) {
        report_error("Mixing weights must sum to the same number.");
      }
      if (fabs(merged_mixing_weights_.sum() - split_mixing_weights_.sum()) >
          numerical_fudge_factor) {
        report_error(
            "Mixing weight vectors differ in positions other "
            "than split and merge.  Have they gotten misaligned?");
      }
    }

    void Proposal::set_components(const Ptr<DpMixtureComponent> &merged,
                                  const Ptr<DpMixtureComponent> &empty,
                                  const Ptr<DpMixtureComponent> &split1,
                                  const Ptr<DpMixtureComponent> &split2) {
      if (merged->number_of_observations() !=
          split1->number_of_observations() + split2->number_of_observations()) {
        report_error(
            "All data must be allocated before setting the "
            "proposed mixture components.");
      }
      if (empty->number_of_observations() != 0) {
        report_error("The empty component was not empty.");
      }
      if (merged->mixture_component_index() < 0 ||
          empty->mixture_component_index() < 0 ||
          split1->mixture_component_index() < 0 ||
          split2->mixture_component_index() < 0) {
        report_error(
            "Mixture component index was not set for one of the "
            "components in a SplitMerge::Proposal.");
      }
      if (split2->mixture_component_index() ==
          split1->mixture_component_index()) {
        report_error("split1 and split2 must have distinct positions.");
      } else if (split2->mixture_component_index() >
                 split1->mixture_component_index()) {
        if (split1->mixture_component_index() !=
            merged->mixture_component_index()) {
          report_error(
              "If split2 comes after split1 then the indices for "
              "split1 and merged should be the same.");
        }
      } else {
        // In this branch split2 comes before split1.  When merged becomes
        // split1 it gets shifted one to the right.
        if (split1->mixture_component_index() !=
            merged->mixture_component_index() + 1) {
          report_error("split1 and merged indices are misaligned.");
        }
      }

      merged_ = merged;
      empty_ = empty;
      split1_ = split1;
      split2_ = split2;
    }

    namespace {
      // Ensure that weights sum to 1 (or less).  If the sum is less than or
      // equal to 1 then return 'weights'.  If the sum is larger than 1 but
      // within rounding error then correct it and move on; otherwise raise an
      // error.
      Vector check_mixing_weights(const Vector &weights) {
        double total = weights.sum();
        if (total > 1.0) {
          // Given the 'if' statement, we know delta is positive.
          double delta = total - 1.0;
          if (delta > 1e-12) {
            report_error("Illegal value for mixing weights.");
          } else {
            return weights * (1 - delta);
          }
        }
        return weights;
      }
    }  // namespace

    void Proposal::set_mixing_weights(const Vector &merged_mixing_weights,
                                      const Vector &split_mixing_weights) {
      if (split_mixing_weights.size() != merged_mixing_weights.size()) {
        report_error(
            "The split mixing weight vector should be the same size "
            "as the merged mixing weight vector.");
      }
      const double numerical_fudge_factor = 1e-10;
      if (fabs(merged_mixing_weights.sum() - split_mixing_weights.sum()) >
          numerical_fudge_factor) {
        report_error(
            "merged_mixing_weights and split_mixing_weights should "
            "sum to the same number.");
      }
      merged_mixing_weights_ = check_mixing_weights(merged_mixing_weights);
      split_mixing_weights_ = check_mixing_weights(split_mixing_weights);
    }

    double Proposal::empty_mixing_weight() const {
      return merged_mixing_weights_.back();
    }
    double Proposal::merged_mixing_weight() const {
      return merged_mixing_weights_[merged_->mixture_component_index()];
    }
    double Proposal::split1_mixing_weight() const {
      return split_mixing_weights_[split1_->mixture_component_index()];
    }
    double Proposal::split2_mixing_weight() const {
      return split_mixing_weights_[split2_->mixture_component_index()];
    }

    //======================================================================
    SOSS::SingleObservationSplitStrategy(DirichletProcessMixtureModel *model,
                                         double annealing_factor)
        : model_(model), annealing_factor_(annealing_factor) {}

    //--------------------------------------------------------------------------
    Proposal SOSS::propose_split(int data_index_1, int data_index_2, RNG &rng) {
      int component_index = model_->cluster_indicator(data_index_1);
      if (component_index != model_->cluster_indicator(data_index_2)) {
        report_error(
            "Both data points must belong to the same cluster "
            "in order to attempt a split move.");
      }

      Proposal proposal(Proposal::Split, data_index_1, data_index_2);

      // Initialize the two mixture components split1 and split2.  The
      // parameters for split1 are equal to the original component.  The
      // parameters for split2 are equal to a draw from the posterior given the
      // single data point at data_index_2.  Each component has its seed
      // observation assigned.  The initialize function removes observations 1
      // and 2 from the data set.
      Ptr<DpMixtureComponent> original_component =
          model_->component(component_index);
      std::set<Ptr<Data>> data_set(original_component->abstract_data_set());
      Ptr<DpMixtureComponent> split1 = initialize_split_proposal(
          original_component, data_set, data_index_1, false, rng);
      Ptr<DpMixtureComponent> split2 = initialize_split_proposal(
          original_component, data_set, data_index_2, true, rng);
      if (split2->mixture_component_index() <=
          split1->mixture_component_index()) {
        // split2 will be inserted at its mixture_index.  If it comes before
        // split1 then split1 will be shifted one unit to the right.
        split1->increment_mixture_component_index();
      }
      Ptr<DpMixtureComponent> empty = split2->clone();
      empty->clear_data();
      empty->set_mixture_component_index(model_->number_of_components());
      model_->base_distribution()->draw_model_parameters(*empty);

      // Allocate observations to components using annealed likelihood, with
      // equal prior mising weights.
      double log_partition_probability = allocate_data_between_split_components(
          split1.get(), split2.get(), data_set, rng);

      proposal.set_components(original_component, empty, split1, split2);

      // Compute the mixing weights for the two components.  The final element
      // of original_mixing_weights corresponds to unseen components in the
      // infinite tail. Replace this element with the mixing weight for the
      // empty cluster.
      //
      // Simulate the value of the mixing weight for the empty cluster.  This is
      // a Gibbs sampling draw that can be considered to have taken place prior
      // to this MH proposal.
      double alpha = model_->concentration_parameter();
      double final_stick_fraction = rbeta_mt(rng, 1, alpha);
      Vector original_mixing_weights = model_->mixing_weights();
      original_mixing_weights.back() *= final_stick_fraction;

      // The mixing weights for split1 and split2 are determined by a beta
      // random variable (epsilon) times the total mixing weight from the
      // original and empty components.
      double total_mixing_weight = original_mixing_weights[component_index] +
                                   original_mixing_weights.back();
      double epsilon = rbeta_mt(rng, split1->number_of_observations(),
                                alpha + split2->number_of_observations() - 1);
      double split1_mixing_weight = total_mixing_weight * epsilon;
      double split2_mixing_weight = total_mixing_weight - split1_mixing_weight;

      // The split_mixing_weights are the same size as the
      // original_mixing_weights, but the final component is non-empty.
      Vector split_mixing_weights = original_mixing_weights;
      split_mixing_weights[component_index] = split1_mixing_weight;
      split_mixing_weights.back() = split2_mixing_weight;
      shift_element(split_mixing_weights, split_mixing_weights.size() - 1,
                    split2->mixture_component_index());
      proposal.set_mixing_weights(original_mixing_weights,
                                  split_mixing_weights);

      // Set the final element, check that everything has been set, and return
      // the proposal.
      proposal.set_log_proposal_density_ratio(split_log_proposal_density_ratio(
          proposal, log_partition_probability, data_index_2));
      proposal.check();
      return proposal;
    }

    //----------------------------------------------------------------------
    Ptr<DpMixtureComponent> SOSS::initialize_split_proposal(
        const Ptr<DpMixtureComponent> &original_component,
        std::set<Ptr<Data>> &original_component_data_set, int data_index,
        bool initialize_parameters, RNG &rng) {
      // TODO: Consider getting the component from the buffer of
      // components held by the model if profiling shows the clone operation to
      // be a significant expense.
      Ptr<DpMixtureComponent> component = original_component->clone();
      component->clear_data();
      Ptr<Data> data_point = model_->dat()[data_index];
      component->add_data(data_point);
      bool found = original_component_data_set.erase(data_point);
      if (!found) {
        report_error(
            "Data element was not part of its "
            "assigned mixture component.");
      }
      if (initialize_parameters) {
        sample_parameters(*component);
        Vector mixing_weights = model_->mixing_weights();
        int candidate_index = rmulti_mt(rng, mixing_weights);
        // The new mixture component should be placed in front of the component
        // currently at candidate_index.
        component->set_mixture_component_index(candidate_index);
      } else {
        // The component is to replace the existing component that currently
        // owns data_index.  If the mixture component index of the other
        // component comes before this one, then it will need to be incremented
        // outside this function.
        component->set_mixture_component_index(
            model_->cluster_indicator(data_index));
      }
      return component;
    }

    //----------------------------------------------------------------------
    void SOSS::sample_parameters(DirichletProcessMixtureComponent &component) {
      int ndraws = model_->conjugate() ? 1 : 100;
      for (int i = 0; i < ndraws; ++i) {
        model_->base_distribution()->draw_model_parameters(component);
      }
    }

    //----------------------------------------------------------------------
    double SOSS::allocate_data_between_split_components(
        DpMixtureComponent *split1, DpMixtureComponent *split2,
        const std::set<Ptr<Data>> &data_set, RNG &rng) const {
      double log_partition_probability = 0;
      // If the annealing_factor is set to zero (which we check for with <= 0)
      // then the user intent is to ignore likelhood and just do uniformly
      // random allocation.
      if (annealing_factor_ <= 0) {
        int data_set_size = 0;
        for (const auto &data_point : data_set) {
          ++data_set_size;
          if (runif_mt(rng) < .5) {
            split1->add_data(data_point);
          } else {
            split2->add_data(data_point);
          }
        }
        log_partition_probability = log(.5) * data_set_size;
      } else {
        for (const auto &data_point : data_set) {
          double logp1 =
              annealing_factor_ * split1->pdf(data_point.get(), true);
          double logp2 =
              annealing_factor_ * split2->pdf(data_point.get(), true);
          double lognc = lse2(logp1, logp2);
          logp1 -= lognc;
          double logu = log(runif_mt(rng));
          // Do the allocation on the log scale.
          if (logu < logp1) {
            split1->add_data(data_point);
            // logp1 has already been normalized.
            log_partition_probability += logp1;
          } else {
            split2->add_data(data_point);
            // Need to normalize logp2.
            log_partition_probability += logp2 - lognc;
          }
        }
      }
      return log_partition_probability;
    }

    //----------------------------------------------------------------------
    Proposal SOSS::propose_merge(int data_index_1, int data_index_2, RNG &rng) {
      int component_index_1 = model_->cluster_indicator(data_index_1);
      int component_index_2 = model_->cluster_indicator(data_index_2);
      if (component_index_1 == component_index_2) {
        report_error(
            "Merge move cannot be attempted with data points "
            "in the same cluster");
      }
      Proposal proposal(Proposal::Merge, data_index_1, data_index_2);

      // Initialize the merged and empty components.  The merged component has
      // the same parameters as split1, and all the data from split1 and
      // split2.
      Ptr<DpMixtureComponent> split1 = model_->component(component_index_1);
      Ptr<DpMixtureComponent> split2 = model_->component(component_index_2);
      Ptr<DpMixtureComponent> merged = split1->clone();
      merged->clear_data();
      merged->combine_data(*split1, false);
      merged->combine_data(*split2, false);
      merged->set_mixture_component_index(component_index_1);
      if (component_index_2 < component_index_1) {
        // If split2 is to the left of split1 then it will be removed as part of
        // the merge, so the index for merged will be one less than that of
        // split1.
        merged->decrement_mixture_component_index();
      }

      // Set the parameters for *empty to a draw from p(theta | observation 2).
      Ptr<DpMixtureComponent> empty = split2->clone();
      empty->clear_data();
      empty->add_data(model_->dat()[data_index_2]);
      sample_parameters(*empty);
      empty->clear_data();
      empty->set_mixture_component_index(model_->number_of_components());
      proposal.set_components(merged, empty, split1, split2);

      // Remove the 'all other components' mixing weight from the end of
      // split_mixing_weights.
      Vector split_mixing_weights = model_->mixing_weights();
      split_mixing_weights.pop_back();

      double split1_mixing_weight = split_mixing_weights[component_index_1];
      double split2_mixing_weight = split_mixing_weights[component_index_2];
      double total_mixing_weight = split1_mixing_weight + split2_mixing_weight;
      double alpha = model_->concentration_parameter();
      double n0 = merged->number_of_observations();
      double empty_mixing_weight_fraction = rbeta_mt(rng, 1, alpha + n0);

      double merged_mixing_weight =
          total_mixing_weight * (1 - empty_mixing_weight_fraction);
      double empty_mixing_weight =
          total_mixing_weight * empty_mixing_weight_fraction;

      Vector merged_mixing_weights = split_mixing_weights;
      merged_mixing_weights[component_index_1] = merged_mixing_weight;
      merged_mixing_weights[component_index_2] = empty_mixing_weight;
      shift_element(merged_mixing_weights, component_index_2,
                    merged_mixing_weights.size() - 1);

      proposal.set_mixing_weights(merged_mixing_weights, split_mixing_weights);

      double log_partition_probability = compute_log_partition_probability(
          split1, split2, data_index_1, data_index_2);
      proposal.set_log_proposal_density_ratio(split_log_proposal_density_ratio(
          proposal, log_partition_probability, data_index_2));
      proposal.check();
      return proposal;
    }

    //----------------------------------------------------------------------
    // Returns the log of q(split | merged) / q(merged | split).
    //
    // Because this fraction appears in the denominator of the MH probability,
    // negative numbers favor a split.
    double SOSS::split_log_proposal_density_ratio(
        const Proposal &proposal, double log_allocation_probability,
        int data_index_2) const {
      // The split proposal has the following steps:
      // 1) randomly choose 2 observations.  This step is identical in both the
      //    split and merge moves, so it will cancel in the ratio of the
      //    proposal densities.
      // 2) Set the parameters of split1 equal to the parameters of the cluster
      //    containing observation 1.
      // 3) Choose a new location for the split2 mixture comopnent.
      double split2_location_log_density = log(
          proposal.merged_mixing_weights()[proposal.split2()
                                               ->mixture_component_index()]);

      // 4) Simulate the parameters of split2 ~ p(theta | observation 2): a
      //    single element model with the base distribution as the prior.
      double split_log_prior =
          model_->base_distribution()->log_prior_density(*proposal.split2());
      const Ptr<Data> &data_2 = model_->dat()[data_index_2];
      double split_log_likelihood = proposal.split2()->pdf(data_2.get(), true);

      // 5) Allocate the remaining observations between the two mixture
      //    components using annealed likelihood with equal prior mixing
      //    weights.
      // 6) Combine the mixing weight for 'merged' with the mixing weight for
      //    the first unpopulated component.  Split this weight among split1
      //    and split2 in proportion to the number allocated to each component.
      double total_weight =
          proposal.split1_mixing_weight() + proposal.split2_mixing_weight();
      double split1_mixing_weight_fraction =
          proposal.split1_mixing_weight() / total_weight;
      double alpha = model_->concentration_parameter();
      double split1_mixing_weight_fraction_density =
          dbeta(split1_mixing_weight_fraction,
                proposal.split1()->number_of_observations(),
                alpha + proposal.split2()->number_of_observations() - 1, true);

      double log_proposal_density_split =
          split_log_prior + split_log_likelihood + log_allocation_probability +
          split1_mixing_weight_fraction_density + split2_location_log_density;
      // if (print_mcmc_details) {
      //   std::cout << "   log split proposal density:  "
      //             << log_proposal_density_split << endl
      //             << "          prior:           " << split_log_prior << endl
      //             << "          likelihood:      " << split_log_likelihood
      //             << endl
      //             << "          allocation       " << log_allocation_probability
      //             << endl
      //             << "          mixing fraction: " << split_log_prior << endl
      //             << "          split2 location: "
      //             << split2_location_log_density << endl;
      // }

      // The merge proposal has the following steps:
      // 1) randomly choose 2 observations.
      // 2) Move the data from component 2 to component 1.  Rename component 1
      //    'merged'.  Do not change its parameters.
      // 3) Move component 2 to one past the final populated mixing component,
      //    and rename this component 'empty'.  Simulate the parameters for this
      //    now 'empty' component from p(theta | observation 2).
      double merged_log_prior =
          model_->base_distribution()->log_prior_density(*proposal.empty());
      double merged_log_likelihood = proposal.empty()->pdf(data_2.get(), true);

      // 4) Sum the mixing weights for component 1 and component 2, and call
      //    this sum w0.  Simulate epsilon ~ Beta(1, alpha + n0), where n0 is
      //    the number of observations in the merged component.  Assign 'merged'
      //    a mixing weight of (1 - epsilon) * w0 and 'empty' a mixing weight of
      //    epsilon * w0.
      double empty_mixing_weight_proportion =
          1 - (proposal.merged_mixing_weight() / total_weight);
      double n0 = proposal.merged()->number_of_observations();
      double empty_mixing_weight_fraction_density =
          dbeta(empty_mixing_weight_proportion, 1, alpha + n0, true);

      double log_proposal_density_merged = merged_log_prior +
                                           merged_log_likelihood +
                                           empty_mixing_weight_fraction_density;
      // if (print_mcmc_details) {
      //   std::cout << "   log merged proposal density: "
      //             << log_proposal_density_merged << endl
      //             << "          prior:            " << merged_log_prior << endl
      //             << "          likelihood:       " << merged_log_likelihood
      //             << endl
      //             << "          mixing fraction:  "
      //             << empty_mixing_weight_fraction_density << endl;
      // }
      return log_proposal_density_split - log_proposal_density_merged;
    }

    //----------------------------------------------------------------------
    double SOSS::compute_log_partition_probability(
        const Ptr<DpMixtureComponent> &split1,
        const Ptr<DpMixtureComponent> &split2, int data_index_1,
        int data_index_2) const {
      return log_allocation_probability(split1, split2, data_index_1) +
             log_allocation_probability(split2, split1, data_index_2);
    }

    //----------------------------------------------------------------------
    double SOSS::log_allocation_probability(
        const Ptr<DpMixtureComponent> &component,
        const Ptr<DpMixtureComponent> &other_component, int data_index) const {
      if (annealing_factor_ > 0) {
        double ans = 0;
        Ptr<Data> seed_data_point = model_->dat()[data_index];
        std::set<Ptr<Data>> data_set = component->abstract_data_set();
        for (const auto &el : data_set) {
          if (el != seed_data_point) {
            double logp1 = annealing_factor_ * component->pdf(el.get(), true);
            double logp2 =
                annealing_factor_ * other_component->pdf(el.get(), true);
            double increment = logp1 - lse2(logp1, logp2);
            ans += increment;
          }
        }
        return ans;
      } else {
        const double log_half = log(.5);
        return log_half * (component->number_of_observations() - 1);
      }
    }

  }  // namespace SplitMerge
}  // namespace BOOM
