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

#include "stats/EmpiricalDensity.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "stats/ECDF.hpp"

namespace BOOM {

  namespace {
    void reg_suf_update(double y, const Vector &x, Vector &xty,
                        SpdMatrix &xtx) {
      xtx.add_outer(x);
      xty += y * x;
    }
  }  // namespace

  EmpiricalDensity::EmpiricalDensity(const ConstVectorView &data,
                                     const Vector &knots)
      : spline_(knots), coefficients_(spline_.basis_dimension()) {
    ECDF cdf(data);
    double min_value = cdf.sorted_data()[0];
    double max_value = cdf.sorted_data().back();

    SpdMatrix xtx(coefficients_.size());
    Vector xty(coefficients_.size());

    double dx = (max_value - min_value) / 100.0;
    for (double x = min_value; x <= max_value; x += dx) {
      // The CDF value at x is the y value in a regression problem.  The basis
      // expansion of x is the 'x' value in the regression problem.
      reg_suf_update(cdf(x), spline_.basis(x), xty, xtx);
    }
    for (int i = 0; i < knots.size(); ++i) {
      reg_suf_update(cdf(knots[i]), spline_.basis(knots[i]), xty, xtx);
    }
    coefficients_ = xtx.solve(xty);
  }

  EmpiricalDensity::EmpiricalDensity(const ConstVectorView &data, int num_knots)
      : EmpiricalDensity(data, create_knots(data, num_knots)) {}

  double EmpiricalDensity::operator()(double x) const {
    double ans = coefficients_.dot(spline_.Mspline::basis(x));
    return ans < 0 ? 0 : ans;
  }

  Vector EmpiricalDensity::operator()(const Vector &values) const {
    Vector ans(values.size());
    for (int i = 0; i < values.size(); ++i) {
      ans[i] = operator()(values[i]);
    }
    return ans;
  }

  Vector EmpiricalDensity::create_knots(const ConstVectorView &data,
                                        int num_knots) const {
    if (num_knots <= 0) {
      return Vector(0);
    }
    auto limits = min_max(data);
    double dx = (limits.second - limits.first) / num_knots;
    Vector ans(num_knots);
    ans[0] = limits.first;
    for (int i = 1; i < num_knots; ++i) {
      ans[i] = ans[i - 1] + dx;
    }
    return ans;
  }

}  // namespace BOOM
