/*
  Copyright (C) 2005-2024 Steven L. Scott

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

#include "numopt/ClassAssigner.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/data_checking.hpp"
#include "stats/FreqDist.hpp"
#include "stats/kl_divergence.hpp"
#include "distributions.hpp"

// #include <iostream>

namespace BOOM {

  ClassAssigner::ClassAssigner()
      : niter_(1000),
        distribution_scale_factor_(1.0),
        initial_temperature_(1.0),
        temperature_(1.0),
        max_tolerable_kl_(.1)
  {}

  bool ClassAssigner::accept_candidate(int candidate,
                                       size_t index,
                                       FrequencyDistribution &empirical_distribution,
                                       RNG &rng) const {
    if (candidate == assignment_[index]) {
      return false;
    }
    const Vector &probs(marginal_posteriors_.row(index));
    int original_value = assignment_[index];

    double pmax = probs.max();

    double original_cost =
        log(pmax / probs[original_value]) / assignment_.size()
        + distribution_scale_factor_ * kl_divergence(
            global_target_, empirical_distribution.relative_frequencies());

    empirical_distribution.add_count(candidate);
    empirical_distribution.remove_count(original_value);
    double candidate_cost =
        log(pmax / probs[candidate]) / assignment_.size()
        + distribution_scale_factor_ * kl_divergence(
            global_target_, empirical_distribution.relative_frequencies());
    empirical_distribution.add_count(original_value);
    empirical_distribution.remove_count(candidate);

    if (candidate_cost < original_cost) {
      return true;
    } else {
      double log_accept_prob = (original_cost - candidate_cost) / temperature_;
      if (log_accept_prob < log(runif_mt(rng))) {
        return true;
      }
    }
    return false;
  }

  void ClassAssigner::simulated_annealing(RNG &rng) {
    temperature_ = initial_temperature_;
    for (int i = 0; i < niter_; ++i) {
      int num_changes = simulated_annealing_step(rng);
      temperature_ *= .9;
      if (num_changes == 0) {
        break;
      }
    }
  }

  Int ClassAssigner::simulated_annealing_step(RNG &rng) {
    Int num_changes = 0;
    for (size_t i = 0; i < assignment_.size(); ++i) {
      const ConstVectorView probs(marginal_posteriors_.row(i));
      int candidate = rmulti_mt(rng, probs);
      if (accept_candidate(candidate, i, empirical_distribution_, rng)) {
        ++num_changes;
        empirical_distribution_.remove_count(assignment_[i]);
        empirical_distribution_.add_count(candidate);
        assignment_[i] = candidate;
      }
    }
    return num_changes;
  }

  // The distance between the empirical distribution function and the global
  // distribution is given by the KL divergence (with the global distribution as
  // the baseline).  The KL is the expected (under the baseline) log likelihood
  // ratio between the two distributions.
  //
  // The per-user cost of an allocation is the average log likelihood ratio of
  // the assigned value to the maximum value.  These two components of cost
  // should be on comparable scales.
  double ClassAssigner::cost_function(const std::vector<int> &assignment) const {
    // Step 1: compute the averge log-likelihood ratio "distance" between the
    //   assignment and the MAP assignment.
    double cost = 0;
    for (size_t i = 0; i < assignment.size(); ++i) {
      ConstVectorView probs(marginal_posteriors_.row(i));
      int max_position = probs.imax();
      if (assignment[i] != max_position) {
        cost += log(probs[max_position] / probs[assignment[i]]);
      }
    }
    cost /= assignment.size();

    // Step 2: Add in the KL divergence between the target distribution and the
    //   empirical distribution of the assignment.  The KL divergence is
    //   multiplied by a scale factor that can be used to trade off between
    //   individual assignments and closeness to the target.
    FrequencyDistribution empirical_distribution(
        assignment, 0, marginal_posteriors_.ncol() - 1);
    cost += distribution_scale_factor_ * kl_divergence(
        global_target_,
        empirical_distribution.relative_frequencies());
    return cost;
  }

  //===========================================================================
  std::vector<int> ClassAssigner::assign(
      const Matrix &marginal_posteriors,
      const Vector &global_target,
      RNG &rng) {
    check_inputs(marginal_posteriors, global_target);
    marginal_posteriors_ = marginal_posteriors;
    global_target_ = global_target;

    assignment_.resize(marginal_posteriors.nrow());
    // It is tempting to initialize the assignment by setting each object to its
    // most likely value (MAP estimation).  However, in applications where this
    // function is necessary, it is likely MAP estimation would set some
    // elements of the empirical distribution to zero, or even that all elements
    // would assigned to a single class.  In this case the KL divergence between
    // the global and empirical distributions would be infinite, which is an
    // undesirable starting value.
    //
    // To avoid an infinite KL divergence after initialization, half the objects
    // are assigned MAP estimates, while the other half are randomly assigned
    // values according to the global target distribution.
    for (Int i = 0; i < marginal_posteriors.nrow(); ++i) {
      double u = runif_mt(rng);
      if (u < .5) {
        assignment_[i] = marginal_posteriors.row(i).imax();
      } else {
        assignment_[i] = rmulti_mt(rng, global_target);
      }
    }

    empirical_distribution_ = FrequencyDistribution(
        assignment_, 0, marginal_posteriors.ncol() - 1);

    // 'do' loop here forces at least one pass through simulated annealing.
    do {
      simulated_annealing(rng);
      distribution_scale_factor_ *= 1.5;
    } while (kl() > max_tolerable_kl_);

    return assignment_;
  }

  //===========================================================================
  double ClassAssigner::kl() const {
    return kl_divergence(
        global_target_,
        empirical_distribution_.relative_frequencies());
  }

  //===========================================================================
  void ClassAssigner::check_inputs(
      const Matrix &marginal_posteriors,
      const Vector &global_target) const {
    if (marginal_posteriors.ncol() != global_target.size()) {
      std::ostringstream err;
      err << "The number of columns in marginal_posteriors ("
          << marginal_posteriors.ncol()
          << ") does not equal the size of the global target distribution ("
          << global_target.size()
          << ").";
      report_error(err.str());
    }

    check_probabilities(global_target);
    for (Int i = 0; i < marginal_posteriors.nrow(); ++i) {
      std::string msg = check_probabilities(
          marginal_posteriors.row(i),
          false,
          global_target.size(),
          1e-6,
          false);
      if (msg.size() != 0) {
        std::ostringstream err;
        err << "Problem in row " << i << " of marginal_posteriors:\n"
            << msg;
        report_error(msg);
      }
    }
  }



}  // namespace BOOM
