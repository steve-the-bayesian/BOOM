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
  class SparseVectorView;

  // A value that can be returned from a SparseVector lookup operation to allow
  // assignment.  I.e. if v is a SparseVector then you can say v[3] = 8.0;
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

  // A value that be returned from a SparseVectorView lookup operation to allow
  // for assignment.  This class is very similar to the SparseVectorReturnProxy,
  // but it must manage the begin_ and end_ iterators held by the view in the
  // event that an assignment is made beyond the ends of the view.
  class SparseVectorViewReturnProxy {
   public:
    // Args:
    //   position:  The index of the proxy counting from the perspetive of the view.
    //   value:  The value held by the return proxy.
    //   view: The VectorView object containing the value that the proxy is
    //     standing in for.
    SparseVectorViewReturnProxy(int position, double value, SparseVectorView *v);

    // Assigning a new value will invalidate begin_ or end_ if the proxy points
    // to an element beyond the current begin_ or end_.
    SparseVectorViewReturnProxy &operator=(double new_value);

    // Implicit conversion to double is desired.
    operator double() const; // NOLINT

   private:
    int position_in_base_vector_;
    double value_;
    SparseVectorView *v_;
  };

  //===========================================================================
  // A numeric vector that is logically mostly zeros.  The nonzero values and
  // their positions are stored separately as pairs.
  class SparseVector {
   public:
    // Args:
    //   n: The extent (logical size) of the vector.
    explicit SparseVector(size_t n = 0);

    // A sparse vector where all elements are filled.
    explicit SparseVector(const Vector &dense);

    // Add rhs to the end of *this and return the new *this.
    SparseVector &concatenate(const SparseVector &rhs);
    size_t size() const {return size_;}
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
    // Replaces x with (x + this * weight).
    void add_this_to(Vector &x, double weight) const;
    void add_this_to(VectorView x, double weight) const;

    // Replaces m with (m + scale * this * this->transpose()).
    void add_outer_product(SpdMatrix &m, double scale = 1.0) const;

    // Returns this.transpose() * P * this, which is sum_{i,j} P(i,j)
    // * this[i] * this[j]
    double sandwich(const SpdMatrix &P) const;

    // Returns x * this.transpose.
    Matrix outer_product_transpose(const Vector &x, double scale = 1.0) const;

    // Return the dense vector equivalent to *this.
    Vector dense() const;

    // Iteration is over the nonzero elements in the vector.
    std::map<int, double>::const_iterator begin() const {
      return elements_.begin();
    }

    std::map<int, double>::const_iterator end() const {
      return elements_.end();
    }

   private:
    std::map<int, double> elements_;
    size_t size_;
    void check_index(int n) const;
    friend class SparseVectorReturnProxy;
    friend class SparseVectorViewReturnProxy;
    friend class SparseVectorView;
  };

  Vector operator*(const SpdMatrix &P, const SparseVector &v);
  Vector operator*(const SubMatrix P, const SparseVector &v);
  std::ostream &operator<<(std::ostream &, const SparseVector &v);

  //===========================================================================
  // A view into the elements of a SparseVector.  We'll probably also need a
  // SparseConstVectorView at some point.
  class SparseVectorView {
   public:
    SparseVectorView(SparseVector *base_vector,
                     size_t start,
                     size_t size);

    size_t size() const {return size_;}
    double operator[](int n) const;
    SparseVectorViewReturnProxy operator[](int n);
    SparseVectorView &operator*=(double x);
    SparseVectorView &operator/=(double x);
    double sum() const;
    double dot(const Vector &v) const;
    double dot(const VectorView &v) const;
    double dot(const ConstVectorView &v) const;

    bool operator==(const SparseVector &rhs) const;
    bool operator==(const SparseVectorView &rhs) const;

    // Replaces x with (x + this * weight).
    void add_this_to(Vector &x, double weight) const;
    void add_this_to(VectorView x, double weight) const;

    // Replaces m with (m + scale * this * this->transpose()).
    void add_outer_product(SpdMatrix &m, double scale = 1.0) const;

    // Returns this.transpose() * P * this, which is sum_{i,j} P(i,j)
    // * this[i] * this[j]
    double sandwich(const SpdMatrix &P) const;

    // Returns x * this.transpose.
    Matrix outer_product_transpose(const Vector &x, double scale = 1.0) const;

    // Return the dense vector equivalent to *this.
    Vector dense() const;

    std::map<int, double>::iterator begin();
    std::map<int, double>::iterator end();
    std::map<int, double>::const_iterator begin() const;
    std::map<int, double>::const_iterator end() const;

   private:
    // Return the index of the base vector corresponding to the notional
    // position in the view.
    size_t position_in_base_vector(int view_position) const {
      return start_ + view_position;
    }

    size_t position_in_view(int base_position) const {
      return base_position - start_;
    }
    void ensure_begin_valid() const;
    void ensure_end_valid() const;

    friend class SparseVectorViewReturnProxy;
    SparseVector *base_vector_;
    size_t start_;
    size_t size_;

    // Iterating over the VectorView is difficult because 'begin' and 'end' can
    // get modified if an element is inserted before 'begin' or after 'end'.
    // The C++ standard guarantees that inserting or deleting a value from a map
    // does not invalidate OTHER iterators, so we only need to worry about
    // insertions before begin_ or after end_.
    std::map<int, double>::iterator begin_;
    std::map<int, double>::iterator end_;
    std::map<int, double>::const_iterator cbegin_;
    std::map<int, double>::const_iterator cend_;
  };

}  // namespace BOOM

#endif  // BOOM_SPARSE_VECTOR_HPP_
