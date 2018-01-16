/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include <r_interface/boom_r_tools.hpp>
#include <distributions/rng.hpp>

namespace BOOM {
  namespace RInterface {
    void seed_rng_from_R(SEXP rseed) {
      if (Rf_isNull(rseed)) {
        BOOM::GlobalRng::seed_with_timestamp();
      } else {
        int seed = Rf_asInteger(rseed);
        BOOM::GlobalRng::rng.seed(seed);
        srand(seed);
      }
    }
  }  // namespace RInterface
}  // namespace BOOM
