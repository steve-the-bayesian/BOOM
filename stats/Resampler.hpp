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
#include "BOOM.hpp"
#include "LinAlg/Vector.hpp"

#include <algorithm>
#include <map>
#include <vector>
#include "distributions/rng.hpp"

namespace BOOM {

  // Efficiently sample with replacement from a discrete distribution.
  //
  // Typical usage:
  // Resampler resample(probs);
  //
  // std::vector<int> index = resample(number_of_draws, rng).
  // (Then use index to subsample things)
  //
  // std::vector<Things> ressampled = resample(unique_things, number_of_draws);
  class Resampler {
   public:
    // Resamples according to an equally weighted distribution of
    // dimension nvals.
    // Equivalent to Resampler(Vector(nvals, 1.0 / nvals), false);
    explicit Resampler(int nvals = 1);  // equally weighted [0..nvals-1]

    // Args:
    //   probs:  A discrete distribution (all non-negative elements).
    //   normalize: If true then the probs will be divided by its sum
    //     to ensure proper normalization before being used.  If false
    //     then it is assumed that the normalization has already
    //     occurred prior to construction, so sum(probs) == 1 already.
    explicit Resampler(const Vector &probs, bool normalize = true);

    // Resample from a vector of objects.
    // Args:
    //   source:  The vector of (likely distinct) object to sample from.
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

    // Returns the number of categories in the discrete distribution.
    int dimension() const;

    // Reset the Resampler with a new set of probabilities.
    // Equivalent to Resampler that(probs, normalize); swap(*this, that);
    void set_probs(const Vector &probs, bool normalize = true);

   private:
    typedef std::map<double, int> CDF;
    CDF cdf;
    void setup_cdf(const Vector &probs, bool normalize);
  };

  //------------------------------------------------------------

  template <class T>
  std::vector<T> Resampler::operator()(const std::vector<T> &things,
                                       int number_of_draws, RNG &rng) const {
    std::vector<int> index = (*this)(number_of_draws);
    std::vector<T> ans;
    if (number_of_draws < 0) {
      number_of_draws = cdf.size();
    }
    ans.reserve(number_of_draws);
    for (int i = 0; i < number_of_draws; ++i) {
      ans.push_back(things[index[i]]);
    }
    return ans;
  }

  //______________________________________________________________________

  template <class T>
  std::vector<T> resample(const std::vector<T> &things, int number_of_draws,
                          const Vector &probs) {
    Vector cdf = cumsum(probs);
    double total = cdf.back();
    if (total < 1.0 || total > 1.0) {
      cdf /= total;
    }

    Vector u(number_of_draws);
    u.randomize();
    std::sort(u.begin(), u.end());

    std::vector<T> ans;
    ans.reserve(number_of_draws);
    int cursor = 0;
    for (int i = 0; i < number_of_draws; ++i) {
      while (u[i] > cdf[cursor]) ++cursor;
      ans.push_back(things[cursor]);
    }
    return (ans);
  }
}  // namespace BOOM
#endif  // BOOM_RESAMPLER_HPP
