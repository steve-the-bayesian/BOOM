// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

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

#include "stats/Mspline.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  Mspline::Mspline(const Vector &knots, int order)
      : SplineBase(knots), order_(order) {
    if (knots.size() < 2) {
      basis_dimension_ = 0;
    } else {
      const Vector &sorted_knots(SplineBase::knots());
      for (int i = 1; i < sorted_knots.size() - 1; ++i) {
        if (sorted_knots[i] <= sorted_knots[i - 1]) {
          std::ostringstream err;
          err << "This Mspline implementation does not allow "
                 "duplicate knots.  Knot vector: "
              << sorted_knots;
          report_error(err.str());
        }
      }
      basis_dimension_ = std::max<int>(0, number_of_knots() - order_);
    }
  }

  Vector Mspline::basis(double x) const {
    Vector ans(basis_dimension());
    for (int i = 0; i < ans.size(); ++i) {
      ans[i] = mspline_basis_function(x, order_, i);
    }
    return ans;
  }

  double Mspline::mspline_basis_function(double x, int order,
                                         int which_basis) const {
    if (order < 1) {
      return negative_infinity();
    }
    double left_knot = knot(which_basis);
    double right_knot = knot(which_basis + order);
    if (right_knot == left_knot) return 0;
    if (order == 1) {
      if (left_knot <= x && right_knot > x) {
        return 1.0 / (right_knot - left_knot);
      } else {
        return 0;
      }
    } else {
      return order *
             ((x - left_knot) *
                  mspline_basis_function(x, order - 1, which_basis) +
              (right_knot - x) *
                  mspline_basis_function(x, order - 1, which_basis + 1)) /
             ((order - 1) * (right_knot - left_knot));
    }
  }

  void Mspline::increment_basis_dimension() { ++basis_dimension_; }

  void Mspline::decrement_basis_dimension() { --basis_dimension_; }

  Ispline::Ispline(const Vector &knots, int order) : Mspline(knots, order) {}

  double Ispline::ispline_basis_function(double x, int order,
                                         int which_basis_element) const {
    if (order < 1) return negative_infinity();
    int knot_span_index = knot_span(x);
    if (x >= final_knot()) {
      return 1.0;
    } else if (which_basis_element > knot_span_index) {
      return 0.0;
    } else if (which_basis_element < knot_span_index - order + 1) {
      return 1.0;
    } else {
      double ans = 0;
      for (int m = which_basis_element; m <= knot_span_index; ++m) {
        ans += (knot(m + order + 1) - knot(m)) *
               mspline_basis_function(x, order + 1, m) / (order + 1);
      }
      return ans;
    }
  }

  Vector Ispline::basis(double x) const {
    Vector ans(basis_dimension());
    for (int i = 0; i < ans.size(); ++i) {
      ans[i] = ispline_basis_function(x, order(), i);
    }
    return ans;
  }

}  // namespace BOOM
