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
#ifndef BOOM_RESAMPLER_HPP
#define BOOM_RESAMPLER_HPP
#include "uint.hpp"
#include "LinAlg/Vector.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>
#include "distributions/rng.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  // Efficiently sample with replacement from a discrete distribution.
  //
  // Typical usage:
  // Resampler resample(probs);
  //
  // std::vector<int> index = resample(number_of_draws, rng).
  // (Then use index to subsample things)
  //
  // std::vector<Things> resampled = resample(unique_things, number_of_draws);
  class Resampler {
   public:
    // Args:
    //   probs:  A discrete distribution (all non-negative elements).
    //   normalize: If true then the probs will be divided by its sum
    //     to ensure proper normalization before being used.  If false
    //     then it is assumed that the normalization has already
    //     occurred prior to construction, so sum(probs) == 1 already.
    explicit Resampler(const Vector &probs, bool normalize = true);

    // Resample from a vector of objects.
    // Args:
    //   source: The vector of (likely distinct) object to sample from.  The
    //     length of the vector must match the length of the vector of sampling
    //     weights being used.
    //   number_of_draws: The desired number of draws.  If less than 0
    //     (the default) this will be taken to be source.size().
    //   rng:  The random number generator used to create random subsamples.
    //
    // Returns:
    //   A vector of objects of size number_of_draws.  Draws will be
    //   chosen according to the probabilities passed in at
    //   construction.
    template <class T>
    std::vector<T> operator()(const std::vector<T> &source,
                              int number_of_draws = -1,
                              RNG &rng = GlobalRng::rng) const;

    // Returns a sample, with replacement from [0, ... probs.size() - 1].
    std::vector<int> operator()(int number_of_draws,
                                RNG &rng = GlobalRng::rng) const;

    // Return a randomly chosen index from [0, ..., probs.size() - 1].
    int64_t sample_index(RNG &rng) const;
    
    // Returns the number of categories in the discrete distribution.
    int64_t dimension() const;

    // Reset the Resampler with a new set of probabilities.
    // Equivalent to Resampler that(probs, normalize); swap(*this, that);
    // Args:
    //   probs:  A vector of sampling weights.
    void set_probs(const Vector &probs, bool normalize = true);

   private:
    std::map<double, int64_t>  cdf;

    // If some weights are zero, then the cdf won't include them, so we need to
    // store the weight vector size separately.
    int64_t weight_vector_size_;
    void setup_cdf(const Vector &probs, bool normalize);
  };

  //------------------------------------------------------------
  // This is the primary method of the resampler object.
  template <class T>
  std::vector<T> Resampler::operator()(const std::vector<T> &things,
                                       int number_of_draws,
                                       RNG &rng) const {
    if (cdf.crbegin()->second >= things.size()) {
      report_error("The vector of things to sample is smaller than "
                   "the vector of sampling weights.");
    }
    if (number_of_draws < 0) {
      number_of_draws = dimension();
    }
    std::vector<T> ans;
    ans.reserve(number_of_draws);
    for (int i = 0; i < number_of_draws; ++i) {
      int64_t which_thing = this->sample_index(rng);
      ans.push_back(things[which_thing]);
    }
    return ans;
  }

}  // namespace BOOM
#endif  // BOOM_RESAMPLER_HPP
