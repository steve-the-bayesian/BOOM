/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "Models/Glm/PosteriorSamplers/CorrelationMap.hpp"
#include "distributions.hpp"

namespace BOOM {
  CorrelationMap::CorrelationMap(double threshold)
      : threshold_(threshold), filled_(false) {}

  CorrelationMap::CorrelationMap(double threshold, const RegSuf &suf) {
    set_threshold(threshold);
    fill(suf);
  }

  void CorrelationMap::set_threshold(double threshold) {
    threshold_ = threshold;
    filled_ = false;
  }

  void CorrelationMap::fill(const RegSuf &suf) {
    correlations_.clear();
    SpdMatrix covariance = suf.centered_xtx() / (suf.n() - 1);
    Vector stdev = sqrt(covariance.diag());
    for (auto &s : stdev) if (s <= 0.0) s = 1.0;
    for (int i = 0; i < covariance.nrow(); ++i) {
      for (int j = 0; j < covariance.ncol(); ++j) {
        if (j == i) continue;
        double correlation = fabs(covariance(i, j) / (stdev[i] * stdev[j]));
        if (correlation >= threshold_) {
          correlations_[i].first.push_back(j);
          correlations_[i].second.push_back(correlation);
        }
      }
    }
    filled_ = true;
  }

  int CorrelationMap::propose_swap(RNG &rng, const Selector &included, int index,
                                   double *proposal_weight) const {
    if (!included[index]) {
      report_error("Cannot find a swap partner for a currently excluded variable.");
    }
    auto it = correlations_.find(index);
    if (it == correlations_.end()) {
      *proposal_weight = 0;
      return -1;
    }
    const std::vector<int> &candidate_indices(it->second.first);
    const Vector &abs_correlations(it->second.second);
    // The potential swaps are the subset of candidate_indices that are
    // currently excluded.
    std::vector<int> potential_swaps;
    Vector weights;
    double total = 0;
    for (int i = 0; i < candidate_indices.size(); ++i) {
      if (!included[candidate_indices[i]]) {
        potential_swaps.push_back(candidate_indices[i]);
        weights.push_back(abs_correlations[i]);
        total += weights.back();
      }
    }
    if (total == 0) {
      // The index has no strong correlates.  This should never happen, because
      // no table entry should have been made for this index.
      *proposal_weight = 0;
      return -1;
    }
    weights /= total;
    int which_swap = rmulti_mt(rng, weights);
    *proposal_weight = weights[which_swap];
    return potential_swaps[which_swap];
  }

  double CorrelationMap::proposal_weight(const Selector &included, int index,
                                         int candidate) const {
    if (!included[index]) {
      report_error("Cannot compute the proposal weight for an excluded index.");
    }
    auto it = correlations_.find(index);
    const std::vector<int> &candidate_indices(it->second.first);
    const Vector &abs_correlations(it->second.second);
    // The potential swaps are the subset of candidate_indices that are
    // currently excluded.
    double ans = negative_infinity();
    double total = 0;
    for (int i = 0; i < candidate_indices.size(); ++i) {
      if (!included[candidate_indices[i]]) {
        if (candidate_indices[i] == candidate) {
          ans = abs_correlations[i];
        }
        total += abs_correlations[i];
      }
    }
    if (total == 0) {
      return 0;
    }
    return ans / total;
  }
  
} // namespace BOOM
