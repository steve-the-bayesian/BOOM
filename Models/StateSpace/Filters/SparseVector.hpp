// Copyright 2022-2023 Steven L. Scott

// Copyright 2018 Google LLC. All Rights Reserved.

/*
  Copyright (C) 2005-2018 Steven L. Scott

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
#include <iterator>

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

    SparseVectorViewReturnProxy &operator+= (double increment);
    SparseVectorViewReturnProxy &operator-= (double decrement);
    SparseVectorViewReturnProxy &operator*= (double scale);
    SparseVectorViewReturnProxy &operator/= (double scale);

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

  // An iterator over the nonzero elements in a SparseVectorView.  The iterator
  // class is
  template <class BASE_ITERATOR>
  class SparseVectorViewIteratorImpl {
   public:
    typedef BASE_ITERATOR BaseIterator;
    typedef typename BaseIterator::value_type value_type;

    SparseVectorViewIteratorImpl(const SparseVectorView *host)
        : host_(host) {}

    SparseVectorViewIteratorImpl & operator=(
        const SparseVectorViewIteratorImpl &rhs) {
      if (&rhs != this) {
        underlying_itertator_ = rhs.underlying_itertator_;
      }
      return *this;
    }

    template <class OTHER_ITERATOR>
    SparseVectorViewIteratorImpl & operator=(
        const SparseVectorViewIteratorImpl<OTHER_ITERATOR> &rhs) {
      underlying_itertator_ = rhs.base();
      return *this;
    }

    void set_underlying_iterator(const BaseIterator &it) {
      underlying_itertator_ = it;
    }

    SparseVectorViewIteratorImpl & operator=(const BaseIterator &it) {
      underlying_itertator_ = it;
      return *this;
    }

    bool operator==(const SparseVectorViewIteratorImpl &rhs) const {
      return underlying_itertator_ == rhs.underlying_itertator_
          && stride() == rhs.stride();
    }

    bool operator!=(const SparseVectorViewIteratorImpl &rhs) const {
      return underlying_itertator_ != rhs.underlying_itertator_
          || stride() != rhs.stride();
    }

    SparseVectorViewIteratorImpl &operator++() {
      advance_underlying_iterator(stride());
      return *this;
    }

    SparseVectorViewIteratorImpl operator++(int) {
      SparseVectorViewIteratorImpl ans(*this);
      advance_underlying_iterator(stride());
      return ans;
    }

    SparseVectorViewIteratorImpl operator--() {
      advance_underlying_iterator(-stride());
      return *this;
    }

    SparseVectorViewIteratorImpl operator--(int) {
      SparseVectorViewIteratorImpl ans(*this);
      advance_underlying_iterator(-stride());
      return ans;
    }

    SparseVectorViewIteratorImpl operator+=(int n) {
      advance_underlying_iterator(n * stride());
      return *this;
    }

    SparseVectorViewIteratorImpl operator-=(int n) {
      advance_underlying_iterator(-n * stride());
      return *this;
    }

    SparseVectorViewIteratorImpl operator+(int n) {
      SparseVectorViewIteratorImpl ans(*this);
      ans += n;
      return ans;
    }

    SparseVectorViewIteratorImpl operator-(int n) {
      SparseVectorViewIteratorImpl ans(*this);
      ans -= n;
      return ans;
    }

    std::ptrdiff_t operator-(const SparseVectorViewIteratorImpl &rhs) const {
      assert(stride() == rhs.stride());
      return pos() > rhs.pos() ? (pos() - rhs.pos()) / stride()
          : (rhs.pos() - pos()) / stride();
    }

    BaseIterator &operator->() {
      return underlying_itertator_;
    }

    const BaseIterator &operator->() const {
      return underlying_itertator_;
    }

    value_type &operator*() {
      return *underlying_itertator_;
    }

    const value_type &operator*() const {
      return *underlying_itertator_;
    }

    bool operator<(const SparseVectorViewIteratorImpl &rhs) const {
      return pos() < rhs.pos();
    }

    bool operator>(const SparseVectorViewIteratorImpl &rhs) const {
      return pos() > rhs.pos();
    }

    bool operator<=(const SparseVectorViewIteratorImpl &rhs) const {
      return pos() <= rhs.pos();
    }

    bool operator>=(const SparseVectorViewIteratorImpl &rhs) const {
      return pos() >= rhs.pos();
    }

    int stride() const;

    const BaseIterator &base() const {return underlying_itertator_;}
    BaseIterator &base() {return underlying_itertator_;}

   private:

    // Return true iff the 'pos' argument is an integer that would be hit by
    // moving an integer number of strides from begin_.
    bool included_position(int base_position) const;

    // Move the underlying iterator forward by 'nsteps'.  This can be a
    // backwards move if nsteps < 0.
    void advance_underlying_iterator(bool forward);

    // The position of the iterator in the base vector (the SparseVector object
    // to which the SparseVectorView refers.)
    int64_t pos() const {return underlying_itertator_->first;}

    //---------------------------------------------------------------------------
    // Data section
    //---------------------------------------------------------------------------

    // The underlying_itertator_ points to elements in the map contained by the
    // original SparseVector.
    BaseIterator underlying_itertator_;

    // The view that owns this iterator.
    const SparseVectorView *host_;
  };

  using SparseVectorViewIterator = SparseVectorViewIteratorImpl<
    std::map<int, double>::iterator>;
  using SparseVectorViewConstIterator = SparseVectorViewIteratorImpl<
    std::map<int, double>::const_iterator>;

  //===========================================================================
  // A view into the elements of a SparseVector.  We'll probably also need a
  // SparseConstVectorView at some point.
  class SparseVectorView {
   public:
    // Args:
    //   base_vector:  The SparseVector into which this is a view.
    //   start: The index of the initial element in the SparseVector.  This may
    //     be a structually zero element.
    //   size: The notional size of the view to create.  The is the number of
    //     elements (including the structural zeros) in the "vector".
    //   stride: The number of steps between included elements.  Typically
    //     'stride' will be 1, but if it is 2 then every other element is
    //     selected.  If 3 then every third element, etc.
    SparseVectorView(SparseVector *base_vector,
                     size_t start,
                     size_t size,
                     int stride);

    size_t size() const {return size_;}
    int stride() const {return stride_;}
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

    SparseVectorViewIterator begin();
    SparseVectorViewIterator end();
    SparseVectorViewConstIterator begin() const;
    SparseVectorViewConstIterator end() const;

    // Return the index of the base vector corresponding to the notional
    // position in the view.
    size_t position_in_base_vector(int64_t view_position) const {
      return start_ + view_position * stride_;
    }

    size_t position_in_view(int64_t base_position) const {
      int64_t delta = base_position - start_;
      int64_t pos = delta / stride_;
      int64_t remainder = delta % stride_;
      return remainder == 0 ? pos : -1;
    }

   private:
    // set begin_ and end_ to the first and last relevant values in elements_
    void initialize_iterators();

    void ensure_begin_valid() const;
    void ensure_end_valid() const;

    //---------------------------------------------------------------------------
    // Data section.
    //---------------------------------------------------------------------------
    friend class SparseVectorViewReturnProxy;
    SparseVector *base_vector_;

    // The index of the first element in the SparseVector.  This might be a
    // structural zero, and thus not be in the mapped elements.
    size_t start_;

    // The notional number of elements in the SparseVectorView.  This includes
    // structural zeros.
    size_t size_;

    // The distance between notional elements.  If start_ is zero and stride_ is
    // 2 then we take all the even numbered elements.
    int stride_;

    // Iterating over the VectorView is difficult because 'begin' and 'end' can
    // get modified if an element is inserted before 'begin' or after 'end'.
    // The C++ standard guarantees that inserting or deleting a value from a map
    // does not invalidate OTHER iterators, so we only need to worry about
    // insertions before begin_ or after end_.
    SparseVectorViewIterator begin_;
    SparseVectorViewIterator end_;
    SparseVectorViewConstIterator cbegin_;
    SparseVectorViewConstIterator cend_;
  };


  template <class BASE_ITERATOR>
  int SparseVectorViewIteratorImpl<BASE_ITERATOR>::stride() const {
    return host_->stride();
  }

  template <class BASE_ITERATOR>
  bool SparseVectorViewIteratorImpl<BASE_ITERATOR>::included_position(
      int base_position) const {
    // The logic for 'before_end' and 'after_begin' must handle the possibility
    // that the 'one past the end' map iterator might have an index value less
    // than zero.
    bool before_end = host_->end()->first > 0 ?
        base_position < host_->end()->first : base_position >= 0;
    bool after_begin = base_position < 0 || base_position > host_->begin()->first;
    return after_begin && before_end
        && (base_position - host_->begin()->first) % host_->stride() == 0;
  }


  template <class BASE_ITERATOR>
  void SparseVectorViewIteratorImpl<BASE_ITERATOR>::advance_underlying_iterator(
      bool forward) {
    if (forward) {
      while(true) {
        ++underlying_itertator_;
        if (included_position(underlying_itertator_->first)
            || underlying_itertator_ == host_->end().base()) {
          return;
        }
      }
    } else {
      while(true) {
        --underlying_itertator_;
        if (underlying_itertator_->first <= host_->position_in_base_vector(0)) {
          return;
        }
      }
    }
  }



}  // namespace BOOM

namespace std {
  template<> struct iterator_traits<BOOM::SparseVectorViewIterator> {
    using iterator_category = bidirectional_iterator_tag;
    using difference_type   = ptrdiff_t;
    using value_type        = std::map<int, double>::value_type;
    using pointer           = std::map<int, double>::pointer;
    using reference         = std::map<int, double>::reference;
  };

  template<> struct iterator_traits<BOOM::SparseVectorViewConstIterator> {
    using iterator_category = bidirectional_iterator_tag;
    using difference_type   = ptrdiff_t;
    using value_type        = std::map<int, const double>::value_type;
    using pointer           = std::map<int, const double>::pointer;
    using reference         = std::map<int, const double>::reference;
  };
}

#endif  // BOOM_SPARSE_VECTOR_HPP_
