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

#include "LinAlg/CorrelationMatrix.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"

#include "distributions.hpp"

namespace BOOM {

  typedef CorrelationMatrix CM;
  typedef SpdMatrix Spd;

  CM::CorrelationMatrix() : SpdMatrix() {}

  CM::CorrelationMatrix(int dim) : SpdMatrix(dim) { set_diag(1.0, true); }

  CM::CorrelationMatrix(int dim, double *m, bool ColMajor)
      : SpdMatrix(dim, m, ColMajor) {}

  CM::CorrelationMatrix(const Matrix &m) : SpdMatrix(var2cor(m)) {}

  CM::CorrelationMatrix(const CM &cm) : SpdMatrix(cm) {}

  CM &CM::operator=(const Matrix &x) {
    Spd::operator=(var2cor(x));
    return *this;
  }

  CM &CM::operator=(const CM &rhs) {
    if (&rhs != this) Spd::operator=(rhs);
    return *this;
  }

  Vector CM::vectorize(bool minimal) const {  // copies upper triangle
    uint n = ncol();
    uint ans_size = minimal ? nelem() : n * n;
    Vector ans(ans_size);
    Vector::iterator it = ans.begin();
    for (uint i = 0; i < n; ++i) {
      dVector::const_iterator b = col_begin(i);
      dVector::const_iterator e = minimal ? b + i : b + n;
      it = std::copy(b, e, it);
    }
    return ans;
  }

  Vector::const_iterator CM::unvectorize(Vector::const_iterator &b,
                                         bool minimal) {
    uint n = ncol();
    for (uint i = 0; i < n; ++i) {
      Vector::const_iterator e = minimal ? b + i : b + n;
      dVector::iterator dest = col_begin(i);
      std::copy(b, e, dest);
      b = e;
    }
    make_symmetric();
    return b;
  }

  void CM::unvectorize(const Vector &x, bool minimal) {
    Vector::const_iterator b(x.begin());
    unvectorize(b, minimal);
  }

  uint CM::nelem() const {
    uint n = nrow();
    return n * (n - 1) / 2;
  }

  //======================================================================
  CM var2cor(const SpdMatrix &v) {
    uint n = v.nrow();
    CM ans(n);
    Vector sd = ::BOOM::sqrt(v.diag());
    for (uint i = 0; i < n; ++i) {
      for (uint j = 0; j < i; ++j) {
        ans.unchecked(i, j) = ans.unchecked(j, i) =
            v.unchecked(i, j) / (sd[i] * sd[j]);
      }
    }
    return ans;
  }

  SpdMatrix cor2var(const CM &cor, const Vector &sd) {
    uint n = cor.nrow();
    assert(sd.size() == n);
    SpdMatrix ans(cor);
    for (uint i = 0; i < n; ++i) {
      for (uint j = 0; j < i; ++j) {
        ans.unchecked(i, j) *= sd[i] * sd[j];
        ans.unchecked(j, i) = ans.unchecked(i, j);
      }
      ans.unchecked(i, i) *= sd[i] * sd[i];
    }
    return ans;
  }

  bool CM::operator==(const CM &rhs) const {
    return SpdMatrix::operator==(rhs);
  }

  bool CM::operator!=(const CM &rhs) const {
    return !SpdMatrix::operator==(rhs);
  }

  CM random_cor_mt(RNG &rng, int dim) {
    SpdMatrix I(dim, 1.0);
    SpdMatrix Sigma = rWish_mt(rng, dim + 1, I, true);  // inverse wishart
    return var2cor(Sigma);
  }

}  // namespace BOOM
