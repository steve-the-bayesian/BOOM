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

#ifndef BOOM_SEED_RNG_FROM_R_HPP_
#define BOOM_SEED_RNG_FROM_R_HPP_

#include <r_interface/boom_r_tools.hpp>
namespace BOOM{
  namespace RInterface{

    // If rseed is non-NULL, then the BOOM global random number
    // generator will be seeded with the time since the epoch.  If
    // rseed is convertable to an int then that seed will be used in
    // both BOOM's global RNG, and it will be passed to srand() so C++
    // algorithms relying on rand() will be seeded as well.
    void seed_rng_from_R(SEXP rseed);
  }
}

#endif //  BOOM_SEED_RNG_FROM_R_HPP_
