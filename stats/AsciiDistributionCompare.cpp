// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#include "stats/AsciiDistributionCompare.hpp"
#include "stats/EmpiricalDensity.hpp"

namespace BOOM {
  namespace {
    using ADC = AsciiDistributionCompare;
  }

  ADC::AsciiDistributionCompare(const Vector &x, const Vector &y, int xbuckets,
                                int ybuckets) {
    double xmin = std::min(min(x), min(y));
    double xmax = std::max(max(x), max(y));
    Vector x_density_values(xbuckets);
    Vector y_density_values(xbuckets);
    double dx = (xmax - xmin) / xbuckets;
    double xx = xmin;
    double max_density = 0;
    EmpiricalDensity xdensity(x);
    EmpiricalDensity ydensity(y);
    for (int i = 0; i < xbuckets; ++i) {
      x_density_values[i] = xdensity(xx);
      y_density_values[i] = ydensity(xx);
      max_density = std::max(
          max_density, std::max(x_density_values[i], y_density_values[i]));
      xx += dx;
    }
    graph_ = AsciiGraph(xmin, xmax, 0, max_density, xbuckets, ybuckets);
    for (int i = 0, xx = xmin; i < xbuckets; ++i, xx += dx) {
      graph_.plot(xx, x_density_values[i], 'X');
      graph_.plot(xx, y_density_values[i], '0');
    }
  }

}  // namespace BOOM
