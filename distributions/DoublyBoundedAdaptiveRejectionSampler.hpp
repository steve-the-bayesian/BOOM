// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#ifndef BOOM_DOUBLY_BOUNDED_ADAPTIVE_REJECTION_SAMPLER_HPP
#define BOOM_DOUBLY_BOUNDED_ADAPTIVE_REJECTION_SAMPLER_HPP

#include <functional>
#include <vector>
#include "distributions.hpp"

namespace BOOM {

  class DoublyBoundedAdaptiveRejectionSampler {
   public:
    typedef std::function<double(double)> Fun;
    DoublyBoundedAdaptiveRejectionSampler(double lo, double hi, const Fun &Logf,
                                          const Fun &Dlogf);
    double draw(RNG &);                // simluate a value
    void add_point(double x);          // adds the point to the hull
    double f(double x) const;          // log of the target distribution
    double df(double x) const;         // derivative of logf at x
    double h(double x, uint k) const;  // evaluates the outer hull at x
   private:
    Fun logf_;
    Fun dlogf_;

    // x contains the values of the points that have been tried.
    // initialized with lo and hi
    std::vector<double> x;

    // logf contains the values of the target density at all the
    // points in x.
    std::vector<double> logf;

    // derivatives corresponding to logf
    std::vector<double> dlogf;

    // first knot is at lo.  last is at hi.  interior knots contain
    // the point of intersection of the tangent lines to logf at the
    // points in x
    std::vector<double> knots;

    // cdf[0] = integral of first bit of hull.  cdf[i] = cdf[i-1] +
    // integral of hull part i;
    std::vector<double> cdf;

    void update_cdf();
    void refresh_knots();
    double compute_knot(uint k) const;
    typedef std::vector<double>::iterator IT;
  };

}  // namespace BOOM
#endif  // BOOM_DOUBLY_BOUNDED_ADAPTIVE_REJECTION_SAMPLER_HPP
