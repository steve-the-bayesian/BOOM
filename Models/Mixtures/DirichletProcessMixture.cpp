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

#include "Models/Mixtures/DirichletProcessMixture.hpp"
#include "Models/Mixtures/PosteriorSamplers/SplitMerge.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/shift_element.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef DirichletProcessMixtureModel DPMM;
    typedef ConjugateDirichletProcessMixtureModel CDPMM;
    typedef DirichletProcessMixtureComponent DpMixtureComponent;
    typedef ConjugateDirichletProcessMixtureComponent
        ConjugateDpMixtureComponent;
  }  // namespace

  DPMM::DirichletProcessMixtureModel(
      const Ptr<DirichletProcessMixtureComponent> &mixture_component_prototype,
      const Ptr<HierarchicalPosteriorSampler> &base_distribution,
      const Ptr<UnivParams> &concentration_parameter)
      : mixture_component_prototype_(mixture_component_prototype),
        base_distribution_(base_distribution),
        concentration_parameter_(concentration_parameter),
        mixing_weights_(1, 1.0),
        spare_mixture_component_target_buffer_size_(10) {
    observe_concentration_parameter();
  }

  void DPMM::set_stick_fractions(const Vector &stick_fractions) {
    if (stick_fractions.size() != number_of_components()) {
      report_error("Stick fractions have the wrong dimension.");
    }
    stick_fractions_ = stick_fractions;
    compute_mixing_weights();
  }

  int DPMM::cluster_indicator(int observation) const {
    const Ptr<Data> &data_point(dat()[observation]);
    auto it = cluster_indicators_.find(data_point);
    if (it != cluster_indicators_.end()) {
      // If the observation is currently unassigned then its cluster indicator
      // is the nullptr.  Return -1 in that case.
      return !!it->second ? it->second->mixture_component_index() : -1;
    } else {
      report_error("Cluster indicator could not be found");
      return -2;  // Will never get here
    }
  }

  void DPMM::cluster_indicators(std::vector<int> &indicators) const {
    int sample_size = number_of_observations();
    indicators.resize(sample_size);
    for (int i = 0; i < sample_size; ++i) {
      indicators[i] = cluster_indicator(i);
    }
  }

  void DPMM::add_data(const Ptr<Data> &dp) {
    data_.push_back(dp);
    cluster_indicators_[dp] = nullptr;
  }

  void DPMM::clear_data() {
    data_.clear();
    for (int i = 0; i < mixture_components_.size(); ++i) {
      mixture_components_[i]->clear_data();
    }
    cluster_indicators_.clear();
  }

  void DPMM::combine_data(const Model &other_model, bool just_suf) {
    const DPMM &other(dynamic_cast<const DPMM &>(other_model));
    const std::vector<Ptr<Data>> &other_data(other.dat());
    for (int i = 0; i < other_data.size(); ++i) {
      add_data(other_data[i]);
    }
  }

  void DPMM::accept_split_merge_proposal(const SplitMerge::Proposal &proposal) {
    if (proposal.is_merge()) {
      replace_cluster(
          mixture_components_[proposal.split1()->mixture_component_index()],
          proposal.merged());
      int component_index_2 = proposal.split2()->mixture_component_index();
      mixture_components_[component_index_2]->clear_data();
      remove_empty_cluster(mixture_components_[component_index_2], false);
      // The last element of proposal.merged_mixing_weights() is the mixing
      // weight for an empty cluster.  Get rid of that and put in the collective
      // weight for all unpopulated components.
      mixing_weights_ = proposal.merged_mixing_weights();
      mixing_weights_.back() = 0;
      mixing_weights_.back() = 1.0 - mixing_weights_.sum();
    } else {
      // Accept a split move.
      replace_cluster(
          mixture_components_[proposal.merged()->mixture_component_index()],
          proposal.split1());
      insert_cluster(proposal.split2(),
                     proposal.split2()->mixture_component_index());
      mixing_weights_ = proposal.split_mixing_weights();
      mixing_weights_.push_back(1.0 - mixing_weights_.sum());
    }
    compute_stick_fractions_from_mixing_weights();
  }

  void DPMM::assign_data_to_cluster(const Ptr<Data> &dp, int cluster,
                                    RNG &rng) {
    if (cluster == number_of_components()) {
      add_empty_cluster(rng);
    }
    if (cluster < number_of_components()) {
      mixture_components_[cluster]->add_data(dp);
      cluster_indicators_[dp] = mixture_components_[cluster];
    } else {
      report_error("Invalid cluster index.");
    }
  }

  void DPMM::remove_data_from_cluster(const Ptr<Data> &dp,
                                      bool remove_empty_cluster) {
    Ptr<DirichletProcessMixtureComponent> component = cluster_indicators_[dp];
    if (!!component) {
      component->remove_data(dp);
      if (component->number_of_observations() == 0 && remove_empty_cluster) {
        this->remove_empty_cluster(component, true);
      }
    }
    cluster_indicators_[dp] = nullptr;
  }

  void DPMM::add_empty_cluster(RNG &rng) {
    repopulate_spare_mixture_components();
    Ptr<DirichletProcessMixtureComponent> component =
        spare_mixture_components_.back();
    assign_and_add_mixture_component(component, rng);
    pop_spare_component_stack();
  }

  void DPMM::remove_empty_cluster(const Ptr<DpMixtureComponent> &component,
                                  bool adjust_mixing_weights) {
    if (component->number_of_observations() != 0) {
      report_error("Cluster to be removed is not empty.");
    }
    int which_cluster = component->mixture_component_index();
    if (which_cluster < 0) {
      // The component is not currently assigned.
      return;
    } else if (which_cluster > number_of_components()) {
      report_error("Mixture component index too large.");
    }
    if (mixture_components_[which_cluster] != component) {
      report_error("Mixture components have become misaligned.");
    }
    component->set_mixture_component_index(-1);
    spare_mixture_components_.push_back(component);
    for (int i = which_cluster; i < mixture_components_.size(); ++i) {
      mixture_components_[i]->decrement_mixture_component_index();
    }
    mixture_components_.erase(mixture_components_.begin() + which_cluster);
    if (adjust_mixing_weights) {
      stick_fractions_.erase(stick_fractions_.begin() + which_cluster);
      mixing_weights_.pop_back();
      compute_mixing_weights();
    }
  }

  void DPMM::remove_all_empty_clusters() {
    for (int i = 0; i < mixture_components_.size(); ++i) {
      if (cluster_count(i) == 0) {
        remove_empty_cluster(mixture_components_[i], true);
        --i;
      }
    }
  }

  void DPMM::shift_cluster(int from, int to) {
    shift_element(mixture_components_, from, to);
    for (int i = 0; i < number_of_components(); ++i) {
      mixture_components_[i]->set_mixture_component_index(i);
    }
    mixing_weights_.shift_element(from, to);
    compute_stick_fractions_from_mixing_weights();
  }

  void DPMM::compute_mixing_weights() {
    mixing_weights_.resize(stick_fractions_.size() + 1);
    double fraction_remaining = 1.0;
    for (int i = 0; i < stick_fractions_.size(); ++i) {
      mixing_weights_[i] = stick_fractions_[i] * fraction_remaining;
      fraction_remaining *= (1 - stick_fractions_[i]);
    }
    mixing_weights_.back() = fraction_remaining;
  }

  void DPMM::compute_stick_fractions_from_mixing_weights() {
    stick_fractions_.resize(mixing_weights_.size() - 1);
    stick_fractions_[0] = mixing_weights_[0];
    double probability_remaining = 1.0 - stick_fractions_[0];
    for (int i = 1; i < stick_fractions_.size(); ++i) {
      stick_fractions_[i] = mixing_weights_[i] / probability_remaining;
      probability_remaining -= mixing_weights_[i];
    }
  }

  // Here is the math for the stick breaking distribution.  Let w = w1, w2, ...,
  // wn.  The density factors as p(w) = p(w1) p(w2 | w1) ... p(wn | w1..wn).
  // Each wi is defined as vi * (1 - sum of previous weights), where vi ~
  // Beta(1, alpha).
  //
  // That means p(wi | w1, ..., wi-1) = Beta(wi / previous) / previous, where
  // the extra factor is a Jacobian.
  //
  // There is some nice cancellation that falls out of the beta distribution.
  // Beta(v, 1, alpha) =    Gamma(1 + alpha)
  //                      -------------------  * v^(1-1) * (1-v)^(alpha - 1).
  //                      Gamma(1) Gamma(alpha)
  //
  // Now, Gamma(1 + alpha) = alpha * Gamma(alpha), so the normalizing constant
  // here is just alpha, and the density is just (1-v)^(alpha-1).
  //
  // The Jacobian term is 1/previous, where we can write previous_i =
  // (1-v1)...(1-vi-1).  Thus 1-v1 appears in the denominator of n-1 terms, 1-v2
  // in n-2 terms etc.
  //
  // Putting all this together gives
  //    p(w) = alpha^n \prod_{i=1}^n (1-v_i)^{\alpha + i - 1 - n)}
  //
  // In C's zero-based counting scheme we just replace i-1 with i.
  double DPMM::dstick(const Vector &weights, double alpha, bool logscale) {
    // The amount of probability remaining after subtracting off all previous
    // mixing weights.
    double log_alpha = log(alpha);
    int dim = weights.size();
    double ans = dim * log_alpha;
    double previous_probability = 1.0;
    for (int i = 0; i < dim; ++i) {
      if (previous_probability > 0) {
        double stick_fraction = weights[i] / previous_probability;
        previous_probability -= weights[i];
        ans += (alpha + i - dim) * log(1 - stick_fraction);
      } else {
        // Do some error checking to make sure previous_probability isn't so
        // more negative than can plausibly be attributed to numerical issues.
        if (fabs(previous_probability) > 1e-10) {
          report_error("Vector of weights sums to more than 1.");
        } else {
          // Assume all future weights (and thus all future stick fractions) are
          // zero.  Each zero stick fraction increments ans by zero, so we're
          // done.
          break;
        }
      }
    }
    return logscale ? ans : exp(ans);
  }

  void DPMM::repopulate_spare_mixture_components() {
    if (spare_mixture_components_.empty()) {
      for (int i = 0; i < spare_mixture_component_target_buffer_size(); ++i) {
        Ptr<DirichletProcessMixtureComponent> component =
            mixture_component_prototype_->clone();
        component->clear_data();
        unassign_component_and_add_to_spares(component);
      }
    }
  }

  void DPMM::pop_spare_component_stack() {
    spare_mixture_components_.pop_back();
  }

  void DPMM::unassign_component_and_add_to_spares(
      const Ptr<DirichletProcessMixtureComponent> &component) {
    spare_mixture_components_.push_back(component);
    spare_mixture_components_.back()->set_mixture_component_index(-1);
  }

  void DPMM::assign_and_add_mixture_component(
      const Ptr<DpMixtureComponent> &component, RNG &rng) {
    mixture_components_.push_back(component);
    base_distribution_->draw_model_parameters(*mixture_components_.back());
    mixture_components_.back()->set_mixture_component_index(
        mixture_components_.size() - 1);
    stick_fractions_.push_back(rbeta_mt(rng, 1, concentration_parameter()));
    double remainder = mixing_weights_.back();
    mixing_weights_.back() = remainder * stick_fractions_.back();
    mixing_weights_.push_back(remainder * (1 - stick_fractions_.back()));
  }

  void DPMM::replace_cluster(
      const Ptr<DpMixtureComponent> &component_to_replace,
      const Ptr<DpMixtureComponent> &new_component) {
    int index = component_to_replace->mixture_component_index();

    component_to_replace->set_mixture_component_index(-1);
    component_to_replace->clear_data();

    spare_mixture_components_.push_back(component_to_replace);
    int buffer_size = spare_mixture_component_target_buffer_size_;
    if (spare_mixture_components_.size() > 2 * buffer_size) {
      spare_mixture_components_.erase(
          spare_mixture_components_.begin() + buffer_size,
          spare_mixture_components_.end());
    }

    new_component->set_mixture_component_index(index);
    mixture_components_[index] = new_component;
    std::set<Ptr<Data>> data_set = new_component->abstract_data_set();
    for (const auto &el : data_set) {
      cluster_indicators_[el] = new_component;
    }
  }

  void DPMM::insert_cluster(const Ptr<DpMixtureComponent> &component,
                            int index) {
    mixture_components_.insert(mixture_components_.begin() + index, component);
    std::set<Ptr<Data>> data_set = component->abstract_data_set();
    for (const auto &data_point : data_set) {
      cluster_indicators_[data_point] = component;
    }
    for (int i = index; i < mixture_components_.size(); ++i) {
      mixture_components_[i]->set_mixture_component_index(i);
    }
  }

  void DPMM::observe_concentration_parameter() {
    concentration_parameter_->add_observer(
        this,
        [this]() {
          this->log_concentration_parameter_ = log(this->concentration_parameter());
        });
    concentration_parameter_->set(concentration_parameter());
  }

  //======================================================================
  CDPMM::ConjugateDirichletProcessMixtureModel(
      const Ptr<ConjugateDpMixtureComponent> &mixture_component_prototype,
      const Ptr<ConjugateHierarchicalPosteriorSampler> &base_distribution,
      const Ptr<UnivParams> &concentration_parameter)
      : DPMM(mixture_component_prototype, base_distribution,
             concentration_parameter),
        conjugate_mixture_component_prototype_(mixture_component_prototype),
        conjugate_base_distribution_(base_distribution) {}

  double ConjugateDirichletProcessMixtureModel::log_marginal_density(
      const Ptr<Data> &data_point, int which_component) const {
    if (which_component > 0) {
      return conjugate_base_distribution_->log_marginal_density(
          data_point, component(which_component));
    } else {
      return conjugate_base_distribution_->log_marginal_density(
          data_point, conjugate_mixture_component_prototype_.get());
    }
  }

  void CDPMM::add_empty_cluster(RNG &rng) {
    repopulate_spare_mixture_components();
    Ptr<ConjugateDpMixtureComponent> component =
        spare_conjugate_components_.back();
    conjugate_mixture_components_.push_back(component);
    DPMM::assign_and_add_mixture_component(component, rng);
    pop_spare_component_stack();
  }

  void CDPMM::remove_empty_cluster(const Ptr<DpMixtureComponent> &component,
                                   bool adjust_mixing_weights) {
    int which_cluster = component->mixture_component_index();
    if (conjugate_mixture_components_[which_cluster] != component) {
      report_error("Conjugate mixture components have become misaligned");
    }
    spare_conjugate_components_.push_back(
        conjugate_mixture_components_[which_cluster]);
    conjugate_mixture_components_.erase(conjugate_mixture_components_.begin() +
                                        which_cluster);
    DPMM::remove_empty_cluster(component, adjust_mixing_weights);
  }

  void CDPMM::replace_cluster(
      const Ptr<DpMixtureComponent> &component_to_replace,
      const Ptr<DpMixtureComponent> &new_component) {
    int index = component_to_replace->mixture_component_index();
    conjugate_mixture_components_[index] =
        new_component.dcast<ConjugateDirichletProcessMixtureComponent>();
    DPMM::replace_cluster(component_to_replace, new_component);
  }

  void CDPMM::insert_cluster(const Ptr<DpMixtureComponent> &component,
                             int index) {
    conjugate_mixture_components_.insert(
        conjugate_mixture_components_.begin() + index,
        component.dcast<ConjugateDpMixtureComponent>());
    DPMM::insert_cluster(component, index);
  }

  void CDPMM::shift_cluster(int from, int to) {
    shift_element(conjugate_mixture_components_, from, to);
    DPMM::shift_cluster(from, to);
  }

  void CDPMM::repopulate_spare_mixture_components() {
    if (spare_conjugate_components_.empty()) {
      for (int i = 0; i < spare_mixture_component_target_buffer_size(); ++i) {
        Ptr<ConjugateDirichletProcessMixtureComponent> component =
            conjugate_mixture_component_prototype_->clone();
        component->clear_data();
        unassign_component_and_add_to_spares(component);
        spare_conjugate_components_.push_back(component);
      }
    }
  }

  void CDPMM::pop_spare_component_stack() {
    spare_conjugate_components_.pop_back();
    DPMM::pop_spare_component_stack();
  }

}  // namespace BOOM
