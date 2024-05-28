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

#include "cpputil/data_checking.hpp"
#include "cpputil/report_error.hpp"
#include <sstream>

namespace BOOM {

  std::string check_probabilities(const ConstVectorView &probs,
                           bool require_positive,
                           int size,
                           double tolerance,
                           bool throw_on_error) {
    if (tolerance < 0) {
      report_error("check_probabilities:  tolerance must be positive.");
    }
    std::ostringstream err;
    if (size > 0 && probs.size() != size) {
      err << "The required size is " << size
          << ", but the supplied probabilities had "
          << probs.size() << " elements.";
      if (throw_on_error) {
        report_error(err.str());
      }
    }
    if (fabs(probs.sum() - 1.0) > tolerance) {
      err << "Prior class probabilities must sum to 1.  They sum to "
          << probs.sum()
          << ".";
      if (throw_on_error) {
        report_error(err.str());
      }
    }
    int min_pos = probs.imin();
    if (probs[min_pos] < (require_positive ? 0.0 : tolerance)) {
      std::ostringstream err;
      err << "probs[" << min_pos
          << "] = " << probs[min_pos]
          << ".  All probabilities must be non-negative.";
      if (throw_on_error) {
        report_error(err.str());
      }
    }
    return err.str();
  }

}  // namespace BOOM
