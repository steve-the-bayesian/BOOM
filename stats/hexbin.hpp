#ifndef BOOM_STATS_HEXBIN_HPP_

/*
  Copyright (C) 2005-2020 Steven L. Scott

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

#define BOOM_STATS_HEXBIN_HPP_

#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include <map>

namespace BOOM {

  // A grid containing the counts needed to plot a hexbin.
  class Hexbin {
   public:

    // An empty hexbin grid to be filled in by add_data().
    //
    // Args:
    //   gridsize:  The number of hexagons in each (x/y) direction.
    Hexbin(int gridsize = 50)
        : gridsize_(gridsize)
    {}

    // A hexbin filled by x and y.
    // Args:
    //   x, y:  The x and y points to plot.
    //   gridsize:  The number of hexagons in each (x/y) direction.
    Hexbin(const Vector &x, const Vector &y, int gridsize = 50);

    // Add data to an empty hexbin plot.
    void add_data(const Vector &x, const Vector &y);

    // Return a 3-column matrix containing the (x, y) coordinate of the hexagon
    // centers (first two columns) and count (frequency, third column) for each
    // hexagon with a positive count.
    Matrix hexagons() const;

   private:

    // Create the axes for the plot based on the range of x and y.
    void initialize_bin_axes(const Vector &x, const Vector &y);

    // Add the point x, y to the hexagon grid.
    void increment_hexagon(double x, double y);

    // Given up to 4 candidate hexagons, return the center of the closest
    // hexagon to the given x, y point.  This is the hexagon containing (x, y).
    //
    // Args:
    //   x, y: The point around which to center the search.
    //   xcand0, xcand1:  The indices of points in x_axis_ bracketing x.
    //   ycand0, ycand1:  The indices of points in y_axis_ bracketing y.
    //
    // Returns:
    //   The pair x_axis_[X], y_axis_[Y], where X is either xcand0 or xcand1,
    //   and Y is either ycand0 or ycand1.
    std::pair<double, double> find_center(double x, double y,
                                          int xcand0, int xcand1,
                                          int ycand0, int ycand1) const;

    int gridsize_;

    Vector x_axis_;
    Vector y_axis_;

    // x_, y_ are the coordinates of the hexagon centers.
    // count_ is the number of observations in that hexagon.
    std::map<std::pair<double, double>, int> counts_;
  };

}

#endif  //  BOOM_STATS_HEXBIN_HPP_
