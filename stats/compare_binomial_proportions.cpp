// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#include <sstream>
#include "Bmath/Bmath.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  double compare_binomial_proportions(double successes1, double successes2,
                                      double trials1, double trials2,
                                      double prior_successes,
                                      double prior_failures) {
    successes1 += prior_successes;
    successes2 += prior_successes;
    double failures1 = trials1 - successes1 + prior_failures;
    double failures2 = trials2 - successes2 + prior_failures;
    trials1 = successes1 + failures1;
    trials2 = successes2 + failures2;

    bool complement = false;
    // We want to arrange the names of the arms, and the definition of
    // success and failure, so that successes2 is the smallest of the
    // four inputs.
    double min_value = successes2;
    min_value = std::min(min_value, successes1);
    min_value = std::min(min_value, failures1);
    min_value = std::min(min_value, failures2);

    if (successes2 <= min_value) {
      // do nothing
    } else if (successes1 <= min_value) {
      // Swap the definition of 'arm 1' and 'arm 2'.
      std::swap(successes1, successes2);
      std::swap(failures1, failures2);
      std::swap(trials1, trials2);
      complement = true;
    } else if (failures2 <= min_value) {
      // Swap the definitions of success and failure.
      std::swap(successes2, failures2);
      std::swap(successes1, failures1);
      // No need to swap trials, because no observations are moving
      // between arms.
      complement = true;
    } else if (failures1 <= min_value) {
      // Swap both.  This introduces the complement of the complement.
      std::swap(successes1, failures2);
      std::swap(successes2, failures1);
      std::swap(trials1, trials2);
      complement = false;
    } else {
      std::ostringstream err;
      err << "None of the four inputs was minimal in "
          << "compare_binomial_proportions.  "
          << "Something has gone horribly wrong." << std::endl
          << "min_value = " << min_value << std::endl
          << "successes1 = " << successes1 << std::endl
          << "successes2 = " << successes2 << std::endl
          << "failures1 = " << failures1 << std::endl
          << "failures2 = " << failures2 << std::endl;
      report_error(err.str());
    }

    double ans = 0;

    for (int s = 0; s <= lround(floor(successes2 - 1)); ++s) {
      double numerator1 = Rmath::lchoose(successes1 + successes2 - 1, s);
      double numerator2 =
          Rmath::lchoose(failures1 + failures2 - 1, trials2 - 1 - s);
      double denominator = Rmath::lchoose(trials1 + trials2 - 2, trials1 - 1);
      double pdf = exp(numerator1 + numerator2 - denominator);
      ans += pdf;
    }

    return complement ? 1 - ans : ans;
  }

}  // namespace BOOM
