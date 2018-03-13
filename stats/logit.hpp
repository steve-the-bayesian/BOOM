/*
   Copyright (C) 2005 Steven L. Scott

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

#ifndef STATS_LOGIT_HPP
#define STATS_LOGIT_HPP

#include <distributions/Rmath_dist.hpp>
#include <cmath>
#include <LinAlg/Vector.hpp>

namespace BOOM{
  inline double logit(double x){ return qlogis(x);}
  inline double logit_inv(double x){ return plogis(x);}

  inline Vector logit(const Vector &x){
    Vector ans(x);
    for(int i = 0; i < ans.size(); ++i) ans[i] = logit(ans[i]);
    return ans;
  }

  inline Vector logit_inv(const Vector &x){
    Vector ans(x);
    for(int i = 0; i < ans.size(); ++i) ans[i] = logit_inv(ans[i]);
    return ans;
  }

  inline double lope(double x){
    // "lope" = log one plus exp..
    // stably computes log(1+exp(x))
    if(x>0) return x + ::log1p(exp(-x));
    else return ::log1p(exp(x));
  }
}

#endif // STATS_LOGIT_HPP
