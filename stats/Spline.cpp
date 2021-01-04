// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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
#include "stats/Spline.hpp"
#include <algorithm>
#include <cstring>
#include <sstream>
#include "uint.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  SplineBase::SplineBase(const Vector &knots) : knots_(knots) { knots_.sort(); }

  Matrix SplineBase::basis_matrix(const Vector &x) const {
    Matrix ans(x.size(), this->basis_dimension());
    for (int i = 0; i < x.size(); ++i) {
      ans.row(i) = this->basis(x[i]);
    }
    return ans;
  }

  double SplineBase::final_knot() const {
    return knots_.empty() ? negative_infinity() : knots_.back();
  }

  void SplineBase::add_knot(double knot_location) {
    knots_.insert(std::lower_bound(knots_.begin(), knots_.end(), knot_location),
                  knot_location);
    increment_basis_dimension();
  }

  void SplineBase::remove_knot(int which_knot) {
    if (which_knot < 0 || which_knot >= number_of_knots()) {
      report_error("Requested knot is not in range.");
    }
    knots_.erase(knots_.begin() + which_knot);
    decrement_basis_dimension();
  }

  double SplineBase::knot(int i) const {
    if (knots_.empty()) {
      return negative_infinity();
    } else {
      if (i <= 0) {
        return knots_[0];
      } else if (i >= knots_.size()) {
        return knots_.back();
      } else {
        return knots_[i];
      }
    }
  }

  int SplineBase::knot_span(double x) const {
    Vector::const_iterator terminal_knot_position =
        std::upper_bound(knots_.begin(), knots_.end(), x);
    int terminal_knot = terminal_knot_position - knots_.begin();
    return terminal_knot - 1;
  }

}  // namespace BOOM
