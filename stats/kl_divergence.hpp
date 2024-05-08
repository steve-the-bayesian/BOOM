#ifndef BOOM_STATS_KL_DIVERGENCE_HPP_
#define BOOM_STATS_KL_DIVERGENCE_HPP_
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

#include "LinAlg/Vector.hpp"

namespace BOOM {

  // The Kullback Liebler divergence between discrete probability distributions
  // p1 and p2.   This is defined as the expected log likelihood ratio;
  //
  //   E log(p1 / p2)
  //
  // where the expectation is taken with respect to p1.
  double kl_divergence(const Vector &p1, const Vector &p2);
  
}  // namespace BOOM


#endif  // BOOM_STATS_KL_DIVERGENCE_HPP_
