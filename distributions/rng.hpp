/*
  Copyright (C) 2009 Steven L. Scott

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

#ifndef BOOM_DISTRIBUTIONS_RNG_HPP
#define BOOM_DISTRIBUTIONS_RNG_HPP

#include <boost/random/ranlux.hpp>

namespace BOOM{
typedef boost::random::ranlux64_base_01 RNG;

struct GlobalRng{
 public:
  static RNG rng;
  static void seed_with_timestamp();
};

unsigned long seed_rng();  // generates a random seed from the global RNG
                           // used to seed other RNG's
unsigned long seed_rng(RNG &);
}

#endif// BOOM_DISTRIBUTIONS_RNG_HPP
