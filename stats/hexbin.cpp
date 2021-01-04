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

#include "stats/hexbin.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/seq.hpp"

namespace BOOM {

  Hexbin::Hexbin(const Vector &x, const Vector &y, int gridsize)
      : gridsize_(gridsize)
  {
    if (x.size() != y.size()) {
      report_error("x and y must be the same length.");
    }

    add_data(x, y);
  }

  namespace {
    Vector seq_len(double from, double to, int len) {
      double dx = (to - from) / len;
      return seq<double>(from, to, dx);
    }
  }

  void Hexbin::initialize_bin_axes(const Vector &x, const Vector &y) {
    if (x.size() != y.size()) {
      report_error("Vectors must be the same size.");
    }

    auto xrange = range(x);
    auto yrange = range(y);

    x_axis_ = seq_len(xrange.first, xrange.second, gridsize_);
    y_axis_ = seq_len(yrange.first, yrange.second, gridsize_);
  }

  void Hexbin::add_data(const Vector &x, const Vector &y) {
    if (x.size() != y.size()) {
      report_error("Vectors must be the same size.");
    }
    if (x_axis_.empty()) {
      initialize_bin_axes(x, y);
    }
    for (size_t i = 0; i < x.size(); ++i) {
      increment_hexagon(x[i], y[i]);
    }
  }

  Matrix Hexbin::hexagons() const {
    Matrix ans(counts_.size(), 3);
    int index = -1;
    for (const auto &el: counts_) {
      ++index;
      ans(index, 0) = el.first.first;
      ans(index, 1) = el.first.second;
      ans(index, 2) = el.second;
    }
    return ans;
  }

  namespace {
    int find_lower_bound(double x, const Vector &axis) {
      if (x < axis[0]) {
        return -1;
      } else if (x > axis.back()) {
        return axis.size() - 1;
      } else {
        return std::lower_bound(axis.begin(), axis.end(), x) - axis.begin();
      }
    }

    int find_upper_bound(int index, const Vector &axis) {
      if (index + 1 == axis.size()) {
        return index;
      } else {
        return index + 1;
      }
    }

    int break_1d_tie(double x, const Vector &axis, int ind0, int ind1) {
      if (ind0 < 0 || ind0 == ind1) {
        return ind1;
      } else {
        return fabs(x - axis[ind0]) < fabs(x - axis[ind1]) ? ind0 : ind1;
      }
    }

  }   // namespace


  std::pair<double, double> Hexbin::find_center(
      double x, double y, int xcand0, int xcand1, int ycand0, int ycand1) const {

    int xmin = -1;
    int ymin = -1;

    // If either x or y is at an extreme point, then solve the 1-d problem.
    if (xcand0 < 0 || xcand0 == xcand1) {
      xmin = xcand1;
      ymin = break_1d_tie(y, y_axis_, ycand0, ycand1);

    } else if (ycand0 < 0 || ycand0 == ycand1) {
      ymin = ycand1;
      xmin = break_1d_tie(x, x_axis_, xcand0, xcand1);
    } else {
      // Otherwise there are 4 possible hexagon centers nearby.  Attribute the
      // point to the closest one.
      double min_value = std::hypot(x - x_axis_[xcand0], y - y_axis_[ycand0]);
      xmin = xcand0;
      ymin = xcand0;

      double h01 = std::hypot(x - x_axis_[xcand0], y - y_axis_[ycand1]);
      if (h01 < min_value) {
        min_value = h01;
        xmin = xcand0;
        ymin = ycand1;
      }

      double h10 = std::hypot(x - x_axis_[xcand1], y - y_axis_[ycand0]);
      if (h10 < min_value) {
        min_value = h10;
        xmin = xcand1;
        ymin = ycand0;
      }

      double h11 = std::hypot(x - x_axis_[xcand1], y - y_axis_[ycand1]);
      if (h11 < min_value) {
        xmin = xcand1;
        ymin = ycand1;
      }
    }
    return std::make_pair(x_axis_[xmin], y_axis_[ymin]);
  }

  void Hexbin::increment_hexagon(double x, double y) {
    int xcand0 = find_lower_bound(x, x_axis_);
    int xcand1 = find_upper_bound(xcand0, x_axis_);
    int ycand0 = find_lower_bound(y, y_axis_);
    int ycand1 = find_upper_bound(ycand0, y_axis_);

    auto center = find_center(x, y, xcand0, xcand1, ycand0, ycand1);
    ++counts_[center];
  }

}  // namespace BOOM
