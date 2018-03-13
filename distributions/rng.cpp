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


#include <distributions/rng.hpp>
#include <distributions.hpp>
#include <cpputil/math_utils.hpp>
#include <ctime>

namespace BOOM{

  unsigned long seed_rng(RNG &rng){
    long ans = 0;
    while(ans<=2){
      double u = runif_mt(rng) * std::numeric_limits<long>::max();
      ans = lround(u);
    }
    return ans;
  }

  unsigned long seed_rng(){
    return seed_rng(GlobalRng::rng);
  }

  RNG GlobalRng::rng(8675309);

  void GlobalRng::seed_with_timestamp(){
    long seed = static_cast<long>(time(NULL));
    rng.seed(seed);
  }

}
