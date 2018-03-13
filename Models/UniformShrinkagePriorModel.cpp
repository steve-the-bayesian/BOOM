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

#include "Models/UniformShrinkagePriorModel.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  UniformShrinkagePriorModel::UniformShrinkagePriorModel(double median)
      : ParamPolicy(new UnivParams(median)) {}

  UniformShrinkagePriorModel *UniformShrinkagePriorModel::clone() const {
    return new UniformShrinkagePriorModel(*this);
  }

  void UniformShrinkagePriorModel::set_median(double z) {
    if (z <= 0) {
      report_error("Median of UniformShrinkagePriorModel must be positive.");
    }
    prm()->set(z);
  }

  double UniformShrinkagePriorModel::Logp(double x, double &g, double &h,
                                          uint nd) const {
    double z0 = median();
    double x_plus_z = x + z0;
    double ans = log(z0) - 2 * log(x_plus_z);
    if (nd > 0) {
      g = -2.0 / x_plus_z;
      if (nd > 1) {
        h = 2 / square(x_plus_z);
      }
    }
    return ans;
  }

  double UniformShrinkagePriorModel::Loglike(const Vector &z, Vector &gradient,
                                             Matrix &Hessian, uint nd) const {
    double z0 = z[0];
    const std::vector<Ptr<DoubleData> > &data(dat());
    int n = data.size();
    double ans = n * log(z0);
    if (nd > 0) {
      gradient.resize(1);
      gradient[0] = n / z0;
      if (nd > 1) {
        Hessian.resize(1, 1);
        Hessian(0, 0) = -n / (z0 * z0);
      }
    }

    for (int i = 0; i < n; ++i) {
      double z_plus_x = z0 + data[i]->value();
      ans -= 2.0 * log(z_plus_x);
      if (nd > 0) {
        gradient[0] -= 2.0 / z_plus_x;
        if (nd > 1) {
          Hessian(0, 0) += 2.0 / (z_plus_x * z_plus_x);
        }
      }
    }
    return ans;
  }

  double UniformShrinkagePriorModel::sim(RNG &rng) const {
    return rusp_mt(rng, median());
  }

}  // namespace BOOM
