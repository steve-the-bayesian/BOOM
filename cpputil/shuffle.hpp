#ifndef BOOM_CPPUTIL_SHUFFLE_HPP_
#define BOOM_CPPUTIL_SHUFFLE_HPP_
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

#include "distributions/rng.hpp"
#include "distributions.hpp"

namespace BOOM {

  // Randomly shuffle the elements of the vector 'things'.  This is a
  // replacement for std::random_shuffle that avoids the C++11 random number
  // engines.
  //
  // Args:
  //   things:  The vector of objects to shuffle.
  //   rng:  The BOOM random number generator to use for the shuffling.
  //
  // Effects:
  //   The elements of 'things' are shuffled to a new random order.
  template <class T> void shuffle(std::vector<T> &things,
                                  RNG &rng = GlobalRng::rng) {
    if (things.empty()) {
      return;
    }
    int upper_limit = things.size() - 1;
    for (int i = upper_limit; i  > 0; --i) {
      int other = random_int_mt(rng, 0, i);
      std::swap(things[i], things[other]);
    }
  }
  
}  // namespace BOOM

#endif // BOOM_CPPUTIL_SHUFFLE_HPP_

