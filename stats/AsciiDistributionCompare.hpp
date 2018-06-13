// Copyright 2018 Google LLC. All Rights Reserved.
#ifndef BOOM_ASCII_DISTRIBUTION_COMPARE_HPP_
#define BOOM_ASCII_DISTRIBUTION_COMPARE_HPP_

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

#include "LinAlg/Vector.hpp"
#include "cpputil/AsciiGraph.hpp"

namespace BOOM {
  // A primitive method of plotting multiple histograms using ASCII characters.
  // Mainly useful for logging error messages while testing.
  class AsciiDistributionCompare {
   public:
    // Args:
    //   x, y:  The vectors whose distributions are to be compared.
    //   xbuckets: The number of horizontal buckets (e.g. histogram bars) to use
    //     in estimating the distributions of x and y.
    //   ybuckets: The number of vertical buckets to use for drawing histogram
    //     bars.  Generally xbuckets should be larger than ybuckets because
    //     screens have more horizontal than vertical space.
    AsciiDistributionCompare(const Vector &x, const Vector &y,
                             int xbuckets = 80, int ybuckets = 30);


    // Compare a set of draws to a known true value.
    // Args:
    //   draws:  The simulated values.
    //   truth:  The true value around which 'draws' should be centered.
    //   xbuckets: The number of horizontal buckets (e.g. histogram bars) to use
    //     in estimating the distribution of 'draws'.
    //   ybuckets: The number of vertical buckets to use for drawing histogram
    //     bars.  Generally xbuckets should be larger than ybuckets because
    //     screens have more horizontal than vertical space.
    //
    // Effect:
    //   Produces a rough empirical distribution of 'draws' plotted on an ASCII
    //   graph, with a vertical line representing 'truth'.
    AsciiDistributionCompare(const Vector &draws, double truth,
                             int xbuckets = 80, int ybuckets = 30);
    
    std::string print() const { return graph_.print(); }

   private:
    AsciiGraph graph_;
  };

  inline std::ostream &operator<<(std::ostream &out,
                                  const AsciiDistributionCompare &cmp) {
    return out << cmp.print();
  }

}  // namespace BOOM

#endif  // BOOM_ASCII_DISTRIBUTION_COMPARE_HPP_
