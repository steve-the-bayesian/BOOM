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
#include <distributions.hpp>
#include <cmath>
#include <cassert>
namespace BOOM{
  double dexv(double x, double location, double scale, bool logscale){
    // density of the extreme value distribution with mean 'location' and
    // variance tau^2*pi^2/6, where tau = 'scale'

    assert(scale>0);
    const double mu = -0.577215664901533;
    double log_eps  = mu - (x-location)/scale;
    // eps has a standard exponential distribution
    double ans  =  log_eps - log(scale) -exp(log_eps);
    return logscale ? ans : exp(ans);
  }

  double rexv(double loc, double scale){
    return rexv_mt(GlobalRng::rng, loc, scale);}

  double rexv_mt(RNG & rng, double loc, double scale){
    if(scale==0.0) return loc;
    assert(scale>0);
    const double mu(-0.577215664901533);
    return (mu - log(rexp_mt(rng, 1.0)))*scale + loc;
  }
}
