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

#include "Samplers/ImportanceResampler.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"
#include "stats/Resampler.hpp"

namespace BOOM {

  ImportanceResampler::ImportanceResampler(
      const std::function<double(const Vector &)> &log_target_density,
      const Ptr<DirectProposal> &proposal)
      : log_target_density_(log_target_density), proposal_(proposal) {}

  std::pair<Matrix, Vector> ImportanceResampler::draw(int number_of_draws,
                                                      RNG &rng) {
    std::vector<Vector> proposal_draws;
    proposal_draws.reserve(number_of_draws);
    Vector importance_weights(number_of_draws);

    double max_log_importance_weight = negative_infinity();
    for (int i = 0; i < number_of_draws; ++i) {
      Vector proposed_draw = proposal_->draw(rng);
      proposal_draws.push_back(proposed_draw);
      importance_weights[i] =
          log_target_density_(proposed_draw) - proposal_->logp(proposed_draw);
      max_log_importance_weight =
          std::max<double>(max_log_importance_weight, importance_weights[i]);
    }
    importance_weights -= max_log_importance_weight;
    importance_weights.normalize_logprob();

    std::vector<int> resampling_counts =
        rmultinom_mt(rng, number_of_draws, importance_weights);

    int number_of_distinct_draws = 0;

    for (int i = 0; i < resampling_counts.size(); ++i) {
      if (resampling_counts[i] > 0) {
        ++number_of_distinct_draws;
      }
    }

    Matrix unique_draws(number_of_distinct_draws, proposal_draws[0].size());
    Vector weight(number_of_distinct_draws);
    int pos = 0;
    for (int i = 0; i < resampling_counts.size(); ++i) {
      if (resampling_counts[i] > 0) {
        unique_draws.row(pos) = proposal_draws[i];
        weight[pos] = resampling_counts[i];
        ++pos;
      }
    }
    return std::make_pair(unique_draws, weight);
  }

  Matrix ImportanceResampler::draw_and_resample(int number_of_draws, RNG &rng) {
    Matrix distinct_draws;
    Vector weights;
    std::tie(distinct_draws, weights) = draw(number_of_draws, rng);
    Resampler resample(weights);
    std::vector<int> sample = resample(number_of_draws);

    Matrix ans(number_of_draws, ncol(distinct_draws));
    for (int i = 0; i < number_of_draws; ++i) {
      ans.row(i) = distinct_draws.row(sample[i]);
    }
    return ans;
  }

}  // namespace BOOM
