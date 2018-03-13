// Copyright 2018 Google LLC. All Rights Reserved.
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

#include "stats/moments.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  inline double SQ(double x) { return x * x; }
  Vector mean(const Matrix &m) {
    int nr = nrow(m);
    Vector ave(nr, 1.0 / nr);
    Vector ans = ave * m;
    return ans;
  }

  SpdMatrix var(const Matrix &m) {
    SpdMatrix ans(m.ncol(), 0.0);
    Vector mu = mean(m);
    for (uint i = 0; i < m.nrow(); ++i) {
      Vector tmp = m.row(i) - mu;
      ans.add_outer(tmp);
    }
    ans /= (m.nrow() - 1);
    return ans;
  }

  SpdMatrix cor(const Matrix &m) {
    SpdMatrix V = var(m);
    Vector sd = sqrt(diag(V));
    SpdMatrix d(sd.size());
    d.set_diag(1.0 / sd);

    SpdMatrix ans = d * V * d;
    return ans;
  }

  namespace {
    template <class VECTOR>
    double mean_impl(const VECTOR &x) {
      return x.empty() ? 0 : x.sum() / x.size();
    }

    template <class VECTOR>
    double var_impl(const VECTOR &x) {
      uint n = x.size();
      if (n < 2) {
        return 0;
      }
      double mu = mean(x);
      double sumsq = 0;
      for (uint i = 0; i < n; ++i) sumsq += square(x[i] - mu);
      return sumsq / (n - 1);
    }

  }  // namespace

  double mean(const Vector &x) { return mean_impl(x); }
  double mean(const VectorView &x) { return mean_impl(x); }
  double mean(const ConstVectorView &x) { return mean_impl(x); }

  double var(const Vector &x) { return var_impl(x); }
  double var(const VectorView &x) { return var_impl(x); }
  double var(const ConstVectorView &x) { return var_impl(x); }

  double sd(const Vector &x) { return sqrt(var(x)); }
  double sd(const VectorView &x) { return sqrt(var(x)); }
  double sd(const ConstVectorView &x) { return sqrt(var(x)); }

  double mean(const std::vector<double> &x) { return mean(ConstVectorView(x)); }

  double var(const std::vector<double> &x) { return var(ConstVectorView(x)); }

  double sd(const std::vector<double> &x) { return sqrt(var(x)); }

  double cor(const std::vector<double> &x, const std::vector<double> &y) {
    int n = x.size();
    if (y.size() != n) {
      report_error("x and y must be the same size in cor(x, y).");
    }
    if (n <= 1) return 0;
    double cov = 0;
    double xbar = mean(x);
    double ybar = mean(y);
    double ssx = 0;
    double ssy = 0;
    for (int i = 0; i < n; ++i) {
      double xx = x[i] - xbar;
      double yy = y[i] - ybar;
      cov += xx * yy;
      ssx += xx * xx;
      ssy += yy * yy;
    }
    if ((ssx == 0) && (ssy == 0)) {
      return 1.0;  // Correlation between two constants is 1.
    } else if (cov == 0) {
      return 0;
    } else if ((ssx == 0) || (ssy == 0)) {
      return 0;  // Correlation of a non-constant with a constant
                 // should be zero.
    } else {
      cov /= (n - 1);
      double sdx = sqrt(ssx / (n - 1));
      double sdy = sqrt(ssy / (n - 1));
      return cov / (sdx * sdy);
    }
  }

  double mean(const std::vector<double> &x, double missing) {
    if (x.empty()) return 0.0;
    double total = 0;
    int count = 0;
    for (int i = 0; i < x.size(); ++i) {
      if (x[i] != missing) {
        total += x[i];
        ++count;
      }
    }
    if (count == 0) return 0.0;
    return total / count;
  }

  double var(const std::vector<double> &x, double missing_value_code) {
    if (x.size() <= 1) return 0.0;
    double sumsq = 0;
    double mu = mean(x, missing_value_code);
    int count = 0;
    for (int i = 0; i < x.size(); ++i) {
      if (x[i] != missing_value_code) {
        sumsq += SQ(x[i] - mu);
        ++count;
      }
    }
    if (count <= 1) return 0.0;
    return sumsq / (count - 1);
  }

  double sd(const std::vector<double> &x, double missing) {
    return sqrt(var(x, missing));
  }

  double mean(const std::vector<double> &x, const std::vector<bool> &observed) {
    if (observed.empty()) return mean(x);
    if (x.empty()) return 0.0;
    if (x.size() != observed.size()) {
      ostringstream err;
      err << "error in mean():  x.size() = " << x.size()
          << " observed.size() = " << observed.size() << endl;
      report_error(err.str());
    }
    double sum = 0;
    int count = 0;
    for (int i = 0; i < x.size(); ++i) {
      if (observed[i]) {
        sum += x[i];
        ++count;
      }
    }
    if (count == 0) return 0.0;
    return sum / count;
  }

  double var(const std::vector<double> &x, const std::vector<bool> &observed) {
    if (observed.empty()) return var(x);
    if (x.size() <= 1) return 0.0;
    if (x.size() != observed.size()) {
      ostringstream err;
      err << "error in var():  x.size() = " << x.size()
          << " observed.size() = " << observed.size() << endl;
      report_error(err.str());
    }
    double mu = mean(x, observed);
    int count = 0;
    double sumsq = 0;
    for (int i = 0; i < x.size(); ++i) {
      if (observed[i]) {
        sumsq += SQ(x[i] - mu);
        ++count;
      }
    }
    if (count <= 1) return 0.0;
    return sumsq / (count - 1);
  }

  double sd(const std::vector<double> &x, const std::vector<bool> &observed) {
    return sqrt(var(x, observed));
  }
}  // namespace BOOM
