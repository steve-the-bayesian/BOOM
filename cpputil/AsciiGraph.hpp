// Copyright 2018 Google LLC. All Rights Reserved.
#ifndef BOOM_ASCII_GRAPH_HPP_
#define BOOM_ASCII_GRAPH_HPP_

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

#include <string>
#include <vector>

namespace BOOM {

  // An ASCII plotting device for primitive graphics that can be used, e.g. for
  // logging.
  class AsciiGraph {
   public:
    // Build the graph in a non-printable state.  It is an error to try to plot
    // to a graph built using this constructor.  This constructor allows
    // AsciiGraph objects to exist as data members of other objects that may not
    // have the necessary information to use the primary constructor.  This
    // constructor allows a temporry empty graph to be created with the
    // intention of being replaced before any plotting is attempted.
    AsciiGraph();

    AsciiGraph(const AsciiGraph &rhs) = default;
    AsciiGraph &operator=(const AsciiGraph &rhs) = default;
    AsciiGraph(AsciiGraph &&rhs) = default;
    AsciiGraph &operator=(AsciiGraph &&rhs) = default;

    // This is the primary constructor for AsciiGraph objects.
    // Args:
    //   xlo, xhi:  The conceptual horizontal limits of the plot region.
    //   ylo, yhi:  The conceptual vertical limits of the plot region.
    //   xbuckets: The number of physical characters (pixels) to use in the
    //     horizontal direction.
    //   ybuckets: The number of physical characters (pixels) to use in the
    //     vertical direction.
    //   throw_on_error: If true then the device will report an error when asked
    //     to plot a point at an (X,Y) value outside the range of the plot area.
    //     If false then points outside the plot area are ignored.
    AsciiGraph(double xlo, double xhi, double ylo, double yhi,
               int xbuckets = 80, int ybuckets = 30,
               bool throw_on_error = false);

    // Draw the specified plotting character at the specified coordinates.
    // Args:
    //   x, y:  The coordinates where the point is to be plotted.
    //   plotting_character: The character to draw at the specified coordinates.
    // Effects:
    //   The plotting_character will be drawn at the specified location.  If
    //   there was already a character there it will be overplotted.
    void plot(double x, double y, char plotting_character);

    // Plot a vertical or horizontal line at the specified value.
    void plot_horizontal_line(double value, char plotting_character = '-');
    void plot_vertical_line(double value, char plotting_character = '|');

    // Convert the graph into a single, printable string.
    std::string print() const;

    // Replaces all entries in the graph with spaces, which will look like an
    // empty graph when printed.
    void clear();

   private:
    // Returns -1 if value is outside the plotting area.
    int which_bucket(double value, double lo, double hi, int buckets) const;

    // The conceptual range of the plot area.
    double xlo_;
    double xhi_;
    double ylo_;
    double yhi_;

    // The number of horizontal and vertical pixels available.
    int xbuckets_;
    int ybuckets_;

    // Stores the actual graph, but the vertical order is reversed because plot
    // points are stored as a 'matrix', which increase as you move down from the
    // top, but plotted as cartesian coordinates, which increase as you move up
    // from the origin.
    std::vector<std::string> graph_;

    // Controls how the device reacts when asked to plot a symbol beyond the
    // plotting range.  If true then the device reports an error (using
    // report_error()).  If false then out-of-bounds points are ignored.
    bool throw_on_error_;
  };

}  // namespace BOOM

#endif  // BOOM_ASCII_GRAPH_HPP_
