/*
  Copyright (C) 2005 Steven L. Scott

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
#include "stats/Resampler.hpp"
#include "LinAlg/Vector.hpp"
#include "distributions.hpp"
#include "cpputil/report_error.hpp"
#include <limits>

namespace BOOM {

  Resampler::Resampler(const Vector &probs, bool normalize)
  {
    setup_cdf(probs, normalize);
  }

  std::vector<int> Resampler::operator()(int number_of_draws, RNG &rng) const {
    if (number_of_draws < 0) {
      number_of_draws = dimension();
    }
    std::vector<int> ans(number_of_draws);
    for (int i = 0; i < number_of_draws; ++i) {
      ans[i] = sample_index(rng);
    }
    return ans;
  }

  int64_t Resampler::sample_index(RNG &rng) const {
    return cdf.lower_bound(runif_mt(rng))->second;
  }
  
  void Resampler::set_probs(const Vector &probs, bool normalize) {
    cdf.clear();
    setup_cdf(probs, normalize);
  }

  void Resampler::setup_cdf(const Vector &probs, bool normalize) {
    weight_vector_size_ = probs.size();
    if (probs.empty()) {
      report_error("Resampling weights cannot be empty.");
    }
    int N = probs.size();
    double normalizing_constant = 1.0;
    if (normalize) {
      normalizing_constant = sum(probs);
      if (normalizing_constant <= 0) {
        report_error("Negative or zero normalizing constant.");
      }
    }
    double cumulative_probability = 0.0;
    cdf.clear();
    for (int i = 0; i < N; ++i) {
      double p0 = probs[i] / normalizing_constant;
      if (p0 < 0) {
        report_error("Negative resamplng weight found.");
      }
      if (p0 > 0) {
        cumulative_probability += p0;
        cdf[cumulative_probability] = i;
      }
    }
    if (cumulative_probability > 1 + 1e-8) {
      std::ostringstream err;
      err << "Weights were not properly normalized.  They sum to "
          << cumulative_probability;
      report_error(err.str());
    }
  }

  int64_t Resampler::dimension() const { return weight_vector_size_; }

}  // namespace BOOM
