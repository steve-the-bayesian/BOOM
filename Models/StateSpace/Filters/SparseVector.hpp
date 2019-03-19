// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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

#ifndef BOOM_SPARSE_VECTOR_HPP_
#define BOOM_SPARSE_VECTOR_HPP_

#include <map>

#include "LinAlg/SubMatrix.hpp"
#include "LinAlg/Vector.hpp"

#include "Models/ParamTypes.hpp"

#include "cpputil/Ptr.hpp"
#include "cpputil/RefCounted.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  class SparseVector;
  class SparseVectorReturnProxy {
   public:
    SparseVectorReturnProxy(int position, double value, SparseVector *v);
    SparseVectorReturnProxy &operator=(double new_value);

    // Implicit conversion to double is desired.
    operator double() const;  // NOLINT

   private:
    int position_;
    double value_;
    SparseVector *v_;
  };
  //======================================================================
  class SparseVector {
   public:
    explicit SparseVector(int n = 0);

    // A sparse vector where all elements are filled.
    explicit SparseVector(const Vector &dense);

    // Add rhs to the end of *this and return the new *this.
    SparseVector &concatenate(const SparseVector &rhs);
    int size() const;
    double operator[](int n) const;
    SparseVectorReturnProxy operator[](int n);
    SparseVector &operator*=(double x);
    SparseVector &operator/=(double x);
    double sum() const;
    double dot(const Vector &v) const;
    double dot(const VectorView &v) const;
    double dot(const ConstVectorView &v) const;

    bool operator==(const SparseVector &rhs) const {
      return size_ == rhs.size_ && elements_ == rhs.elements_;
    }
    // Replaces x with (x + this * coefficient).
    void add_this_to(Vector &x, double coefficient) const;
    void add_this_to(VectorView x, double coefficient) const;

    // Replaces m with (m + scale * this * this->transpose()).
    void add_outer_product(SpdMatrix &m, double scale = 1.0) const;

    // Returns this.transpose() * P * this, which is sum_{i,j} P(i,j)
    // * this[i] * this[j]
    double sandwich(const SpdMatrix &P) const;

    // Returns x * this.transpose.
    Matrix outer_product_transpose(const Vector &x, double scale = 1.0) const;

    // Return the dense vector equivalent to *this.
    Vector dense() const;

    std::map<int, double>::const_iterator begin() const {
      return elements_.begin();
    }

    std::map<int, double>::const_iterator end() const {
      return elements_.end();
    }

   private:
    std::map<int, double> elements_;
    int size_;
    void check_index(int n) const;
    friend class SparseVectorReturnProxy;
  };

  Vector operator*(const SpdMatrix &P, const SparseVector &v);
  Vector operator*(const SubMatrix P, const SparseVector &v);
  std::ostream &operator<<(std::ostream &, const SparseVector &v);

}  // namespace BOOM

#endif  // BOOM_SPARSE_VECTOR_HPP_
