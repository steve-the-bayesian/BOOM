#ifndef BOOM_CPPUTIL_DATA_CHECKING_HPP_
#define BOOM_CPPUTIL_DATA_CHECKING_HPP_

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

#include "LinAlg/VectorView.hpp"

namespace BOOM {

  // Args:
  //   probs:  The vector to check.
  //   require_positive: If true then signal an error if a zero probability is
  //     found.  If false then zero probabilities are allowed.
  //   size:  If size > 0 then requre probs.size() to match the 'size' argument.
  //   tolerance: A small positive value to use when checking for zero
  //     probabilities, or checking the sum of the elements vs 1.0
  //   throw_on_error: If true then an exception will be thrown if any elements
  //     of probs are less than zero, or if the sum does not equal 1.  If false,
  //     then the error message that would have been the body of the exception
  //     is returned.
  //
  // Returns:
  //   If the input is a valid probability distribution then the empty string is
  //   returned.  Otherwise an error message is returned (or thrown, depending
  //   on the value of 'throw_on_error').
  std::string check_probabilities(const ConstVectorView &probs,
                                  bool require_positive = true,
                                  int size = 0,
                                  double tolerance = 1e-6,
                                  bool throw_on_error=true);

}  // namespace BOOM

#endif  //  BOOM_CPPUTIL_DATA_CHECKING_HPP_
