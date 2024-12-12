// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007-2011 Steven L. Scott

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

#ifndef BOOM_ARRAY_HPP
#define BOOM_ARRAY_HPP

#include "LinAlg/ArrayIterator.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"

#include <vector>
#include <sstream>
#include <initializer_list>
#include "cpputil/report_error.hpp"

namespace BOOM {
  // ConstArrayBase implements the const methods common to the Array
  // class and its views.  It implements all size-related queries, as
  // well as indexing and equality comparisons.

  class ConstArrayBase {
   public:
    ConstArrayBase();
    ConstArrayBase(const ConstArrayBase &rhs) = default;
    ConstArrayBase(ConstArrayBase &&rhs) = default;
    ConstArrayBase(const std::vector<int> &dims);
    ConstArrayBase(const std::vector<int> &dims,
                   const std::vector<int> &strides);
    virtual ~ConstArrayBase() {}

    ConstArrayBase &operator=(const ConstArrayBase &rhs) = default;
    ConstArrayBase &operator=(ConstArrayBase &&rhs) = default;

    virtual const double *data() const = 0;

    double operator[](const std::vector<int> &index) const;

    bool empty() const;

    int ndim() const { return dims_.size(); }
    int dim(int i) const { return dims_[i]; }
    const std::vector<int> &dim() const { return dims_; }

    // stride(i) is the number of steps you must advance in data()
    // to increment the i'th index by one.
    int stride(int i) const { return strides_[i]; }
    const std::vector<int> &strides() const { return strides_; }

    // size() is the number of elements stored in the array.  It is
    // the product of dims_;
    size_t size() const;

    // If an Array is the same size and shape as another Array-like
    // thing then they can be compared with operator==
    bool operator==(const Vector &rhs) const;
    bool operator==(const VectorView &rhs) const;
    bool operator==(const ConstVectorView &rhs) const;
    // Matrix & allows for SpdMatrix comparisons as well
    bool operator==(const Matrix &rhs) const;
    bool operator==(const ConstArrayBase &rhs) const;

    // operator() is supported for up to six arguments.  An
    // exception will be thrown if the number of arguments supplied
    // does not match the dimension of dims_ and strides_.
    double operator()(int x1) const;
    double operator()(int x1, int x2) const;
    double operator()(int x1, int x2, int x3) const;
    double operator()(int x1, int x2, int x3, int x4) const;
    double operator()(int x1, int x2, int x3, int x4, int x5) const;
    double operator()(int x1, int x2, int x3, int x4, int x5, int x6) const;

    // Utillity functions for creating a std::vector<int> to be used as an
    // index.  Up to 6 dimensions are supported.  More can be added if needed,
    // but if arrays of greater than 6 dimensions are needed, then people will
    // probably create the dimensions programmatically.
    static std::vector<int> index1(int x1);
    static std::vector<int> index2(int x1, int x2);
    static std::vector<int> index3(int x1, int x2, int x3);
    static std::vector<int> index4(int x1, int x2, int x3, int x4);
    static std::vector<int> index5(int x1, int x2, int x3, int x4, int x5);
    static std::vector<int> index6(int x1, int x2, int x3, int x4, int x5,
                                   int x6);

    // If the number of dimensions is 2 or 1 then write the elements
    // to a matrix with the appropriate number of rows and columns.
    // If the dimension is 1 then a column matrix is returned.  If the
    // dimension is 3 or more then an exception is thrown.
    Matrix to_matrix() const;

    virtual ostream & print(ostream &out) const = 0;
    virtual std::string to_string() const = 0;

    // Return the position in the underlying data array of the object at
    // position [i, j, k, ...].
    //
    // Args:
    //   index:  The vector of indicies to be mapped to a memory position.
    //   dims:  The dimensions (extents) of an array.
    //   strides:  The strides between dimensions for an array.
    //
    // Returns:
    //   The array offset of the data at index [i, j, k, ...].
    static size_t array_index(const std::vector<int> &index,
                              const std::vector<int> &dims,
                              const std::vector<int> &strides);

    // Compute the vector of strides needed to store an array with the given set
    // of dimensions.  Views into an array may use different strides.  These are
    // for a dense, packed array.
    //
    // Args:
    //   dims: The dimensions of the array.
    //   strides: The vector that will receive the computed strides.  On output
    //     it will be the same size as 'dims'.
    //   fortran_order: If true then the strides will be computed according to
    //     'column major order' where the lowest indicies change the fastest.
    //     If false then strides will be computed for 'row major order' where
    //     the highest indices change the fastest.
    static void compute_strides(const std::vector<int> &dims,
                                std::vector<int> &strides,
                                bool fortran_order = true);

   private:
    std::vector<int> dims_;
    std::vector<int> strides_;

   protected:
    void reset_dims(const std::vector<int> &dims);
    void reset_strides(const std::vector<int> &strides);
    void compute_strides();
    static int product(const std::vector<int> &dims);
  };

  inline ostream & operator<<(ostream &out, const ConstArrayBase &array) {
    return array.print(out);
  }

  //======================================================================
  class ArrayBase : public ConstArrayBase {
   public:
    typedef ArrayIterator iterator;

    ArrayBase();
    ArrayBase(const ArrayBase &rhs) = default;
    ArrayBase(ArrayBase &&rhs) = default;
    ArrayBase(const std::vector<int> &dims);
    ArrayBase(const std::vector<int> &dims, const std::vector<int> &strides);

    ArrayBase &operator=(const ArrayBase &rhs) = default;
    ArrayBase &operator=(ArrayBase &&rhs) = default;

    using ConstArrayBase::data;
    virtual double *data() = 0;

    using ConstArrayBase::operator[];
    double &operator[](const std::vector<int> &index);

    using ConstArrayBase::operator();
    double &operator()(int x1);
    double &operator()(int x1, int x2);
    double &operator()(int x1, int x2, int x3);
    double &operator()(int x1, int x2, int x3, int x4);
    double &operator()(int x1, int x2, int x3, int x4, int x5);
    double &operator()(int x1, int x2, int x3, int x4, int x5, int x6);
  };

  //======================================================================
  inline bool operator==(const Vector &lhs, const ConstArrayBase &rhs) {
    return rhs == lhs;
  }
  inline bool operator==(const VectorView &lhs, const ConstArrayBase &rhs) {
    return rhs == lhs;
  }
  inline bool operator==(const ConstVectorView &lhs,
                         const ConstArrayBase &rhs) {
    return rhs == lhs;
  }
  inline bool operator==(const Matrix &lhs, const ConstArrayBase &rhs) {
    return rhs == lhs;
  }
  //======================================================================
  class Array;
  class ConstArrayView : public ConstArrayBase {
   public:
    typedef ConstArrayIterator const_iterator;
    typedef ConstArrayIterator iterator;

    ConstArrayView(const Array &);
    ConstArrayView(const double *data, const std::vector<int> &dims);
    ConstArrayView(const double *data, const std::vector<int> &dims,
                   const std::vector<int> &strides);
    ConstArrayView(const ConstArrayBase &rhs);

    const double *data() const override { return data_; }

    void reset(const double *data, const std::vector<int> &dims);
    void reset(const double *data, const std::vector<int> &dims,
               const std::vector<int> &strides);

    // 'slice' returns a lower dimensional view into an array.  If you have a
    // 3-way array indexed by (i, j, k), and you want to get the (i, k) slice
    // (that is, (i, 0, k), (i, 1, k), ...), then you call array.slice(i, -1,
    // k).  The negative index says 'give me all of these', analogous to a
    // missing index in R, or the : symbol in Python.  The return value is a
    // view into the array with dimension equal to the number of negative
    // arguments.
    ConstArrayView slice(const std::vector<int> &index) const;
    ConstArrayView slice(int x1) const;
    ConstArrayView slice(int x1, int x2) const;
    ConstArrayView slice(int x1, int x2, int x3) const;
    ConstArrayView slice(int x1, int x2, int x3, int x4) const;
    ConstArrayView slice(int x1, int x2, int x3, int x4, int x5) const;
    ConstArrayView slice(int x1, int x2, int x3, int x4, int x5, int x6) const;

    // vector_slice() works in exactly the same way as slice(), but it returns a
    // VectorView instead of an ArrayView.  Exactly one index must be negative.
    ConstVectorView vector_slice(const std::vector<int> &index) const;
    ConstVectorView vector_slice(int x1) const;
    ConstVectorView vector_slice(int x1, int x2) const;
    ConstVectorView vector_slice(int x1, int x2, int x3) const;
    ConstVectorView vector_slice(int x1, int x2, int x3, int x4) const;
    ConstVectorView vector_slice(int x1, int x2, int x3, int x4, int x5) const;
    ConstVectorView vector_slice(int x1, int x2, int x3, int x4, int x5,
                                 int x6) const;

    ConstArrayIterator begin() const;
    ConstArrayIterator end() const;

    // Args:
    //   dims: An array of integers over which to compute the sum.  The length
    //     of dims must be no greater than the number of dimensions of the
    //     array.  The elements of dims must be unique, non-negative, and less
    //     than this->ndim().
    //
    // Returns:
    //   The dimensions of the returned array correspond to this->dim() with any
    //   dimensions in 'dims' removed.
    Array sum(const std::vector<int> &dims) const;

    // The sum of all elements in the array;
    double scalar_sum() const;
    double sum() const {return scalar_sum();}

    ostream &print(ostream &out) const override;
    std::string to_string() const override;

    // Args:
    //   apply_over_dims: The dimensions of the array over which to apply the
    //     function.
    //   functor:  The function to apply.
    //
    // Returns:
    //   The array formed by applying the scalar valued function over the
    //   requested dimensions.
    //
    // Example: If the array has 3 dimensions (8, 6, 7) and apply_over_dims =
    //   {0, 2} then the return value will be an array of a single dimension of
    //   size 6.  Entry i in the returned array is the result of 'functor'
    //   applied to the sub-array(this->slice(-1, i, -1)).
    Array apply_scalar_function(
        const std::vector<int> &apply_over_dims,
        const std::function<double(const ConstArrayView &)> &functor) const;

   private:
    const double *data_;
  };

  //======================================================================
  class ArrayView : public ArrayBase {
   public:
    typedef ConstArrayIterator const_iterator;
    typedef ArrayIterator iterator;

    ArrayView(Array &);
    ArrayView(double *data, const std::vector<int> &dims);
    ArrayView(double *data, const std::vector<int> &dims,
              const std::vector<int> &strides);

    double *data() override { return data_; }
    const double *data() const override { return data_; }

    void reset(double *data, const std::vector<int> &dims);
    void reset(double *data, const std::vector<int> &dims,
               const std::vector<int> &strides);

    ArrayView &operator=(const Array &a);
    ArrayView &operator=(const ArrayView &a);
    ArrayView &operator=(const ConstArrayView &a);
    ArrayView &operator=(const Matrix &a);
    ArrayView &operator=(const Vector &a);
    ArrayView &operator=(const VectorView &a);
    ArrayView &operator=(const ConstVectorView &a);

    // 'slice' returns a lower dimensional view into an array.  If you have a
    // 3-way array indexed by (i, j, k), and you want to get the (i, k) slice
    // (that is, (i, 0, k), (i, 1, k), ...), then you call array.slice(i, -1,
    // k).  The negative index says 'give me all of these', analogous to a
    // missing index in R.  The return value is a view into the array with
    // dimension equal to the number of negative arguments.
    ConstArrayView slice(const std::vector<int> &index) const;
    ConstArrayView slice(int x1) const;
    ConstArrayView slice(int x1, int x2) const;
    ConstArrayView slice(int x1, int x2, int x3) const;
    ConstArrayView slice(int x1, int x2, int x3, int x4) const;
    ConstArrayView slice(int x1, int x2, int x3, int x4, int x5) const;
    ConstArrayView slice(int x1, int x2, int x3, int x4, int x5, int x6) const;

    ArrayView slice(const std::vector<int> &index);
    ArrayView slice(int x1);
    ArrayView slice(int x1, int x2);
    ArrayView slice(int x1, int x2, int x3);
    ArrayView slice(int x1, int x2, int x3, int x4);
    ArrayView slice(int x1, int x2, int x3, int x4, int x5);
    ArrayView slice(int x1, int x2, int x3, int x4, int x5, int x6);

    // vector_slice() works in exactly the same way as slice(), but it
    // returns a VectorView instead of an ArrayView.  Exactly one
    // index must be negative.
    VectorView vector_slice(const std::vector<int> &index);
    VectorView vector_slice(int x1);
    VectorView vector_slice(int x1, int x2);
    VectorView vector_slice(int x1, int x2, int x3);
    VectorView vector_slice(int x1, int x2, int x3, int x4);
    VectorView vector_slice(int x1, int x2, int x3, int x4, int x5);
    VectorView vector_slice(int x1, int x2, int x3, int x4, int x5, int x6);

    ConstVectorView vector_slice(const std::vector<int> &index) const;
    ConstVectorView vector_slice(int x1) const;
    ConstVectorView vector_slice(int x1, int x2) const;
    ConstVectorView vector_slice(int x1, int x2, int x3) const;
    ConstVectorView vector_slice(int x1, int x2, int x3, int x4) const;
    ConstVectorView vector_slice(int x1, int x2, int x3, int x4, int x5) const;
    ConstVectorView vector_slice(int x1, int x2, int x3, int x4, int x5,
                                 int x6) const;

    ConstArrayIterator begin() const;
    ConstArrayIterator end() const;
    ArrayIterator begin();
    ArrayIterator end();

    // Args:
    //   dims: An array of integers over which to compute the sum.  The length
    //     of dims must be no greater than the number of dimensions of the
    //     array.  The elements of dims must be unique, non-negative, and less
    //     than this->ndim().
    //
    // Returns:
    //   The dimensions of the returned array correspond to this->dim() with any
    //   dimensions in 'dims' removed.
    Array sum(const std::vector<int> &dims) const;
    double scalar_sum() const;
    double sum() const {return scalar_sum();}

    ostream &print(ostream &out) const override;
    std::string to_string() const override;

   private:
    double *data_;
  };

  //======================================================================
  class Array : public ArrayBase {
   public:
    typedef std::vector<double>::iterator iterator;
    typedef std::vector<double>::const_iterator const_iterator;

    // Sets data to zero
    Array() {}
    explicit Array(const std::vector<int> &dims, double initial_value = 0);
    Array(const std::vector<int> &dims, const std::vector<double> &data);
    Array(const std::vector<int> &dims, const double *data);

    explicit Array(const std::initializer_list<int> &dims,
                   double initial_value = 0);

    // Convenience constructor for a 3-way array.  The first array dimension is
    // the index of the vector.  The second and third dimensions are the rows
    // and columns of the elements of 'matrices.'  If 'matrices' is empty then
    // all three dimensions are zero.  Otherwise, an error will be reported if
    // the matrices are not all the same size.
    explicit Array(const std::vector<Matrix> &matrices);

    Array(const Array &rhs) = default;
    Array(Array &&rhs) = default;
    Array &operator=(const Array &rhs) = default;
    Array &operator=(Array &&rhs) = default;

    Array &resize(const std::vector<int> &dims);

    // The following assignment opertors expect the array to have the same size
    // as the RHS, and will produce errors if the dimensions differ.
    Array &operator=(const ArrayView &a);
    Array &operator=(const ConstArrayView &a);
    Array &operator=(const Matrix &a);
    Array &operator=(const Vector &a);
    Array &operator=(const VectorView &a);
    Array &operator=(const ConstVectorView &a);
    Array &operator=(double x);

    Array &operator+=(const Array &rhs);
    Array &operator+=(const ConstArrayView &rhs);
    Array &operator+=(double x);
    Array &operator-=(double x);
    Array &operator*=(double x);
    Array &operator/=(double x);

    template <class FwdIt>
    Array &assign(FwdIt begin, FwdIt end) {
      data_.assign(begin, end);
      if (data_.size() != this->size()) {
        report_error("Wrong sized data passed to Array::assign");
      }
      return *this;
    }

    double *data() override { return data_.data(); }
    const double *data() const override { return data_.data(); }

    using ConstArrayBase::operator==;
    bool operator==(const Array &rhs) const;

    // Fill the array with U(0,1) random numbers
    void randomize();

    // 'slice' returns a lower dimensional view into an array.  If you
    // have a 3-way array indexed by (i, j, k), and you want to get
    // the (i, k) slice (that is, (i, 0, k), (i, 1, k), ...), then you
    // call array.slice(i, -1, k).  The negative index says 'give me
    // all of these', analogous to a missing index in R.  The return
    // value is a view into the array with dimension equal to the
    // number of negative arguments.
    ConstArrayView slice(const std::vector<int> &index) const;
    ConstArrayView slice(int x1) const;
    ConstArrayView slice(int x1, int x2) const;
    ConstArrayView slice(int x1, int x2, int x3) const;
    ConstArrayView slice(int x1, int x2, int x3, int x4) const;
    ConstArrayView slice(int x1, int x2, int x3, int x4, int x5) const;
    ConstArrayView slice(int x1, int x2, int x3, int x4, int x5, int x6) const;

    ArrayView slice(const std::vector<int> &index);
    ArrayView slice(int x1);
    ArrayView slice(int x1, int x2);
    ArrayView slice(int x1, int x2, int x3);
    ArrayView slice(int x1, int x2, int x3, int x4);
    ArrayView slice(int x1, int x2, int x3, int x4, int x5);
    ArrayView slice(int x1, int x2, int x3, int x4, int x5, int x6);

    // vector_slice() works in exactly the same way as slice(), but it
    // returns a ConstVectorView instead of a ConstArrayView.  Exactly
    // one index must be negative.
    ConstVectorView vector_slice(const std::vector<int> &index) const;
    ConstVectorView vector_slice(int x1) const;
    ConstVectorView vector_slice(int x1, int x2) const;
    ConstVectorView vector_slice(int x1, int x2, int x3) const;
    ConstVectorView vector_slice(int x1, int x2, int x3, int x4) const;
    ConstVectorView vector_slice(int x1, int x2, int x3, int x4, int x5) const;
    ConstVectorView vector_slice(int x1, int x2, int x3, int x4, int x5,
                                 int x6) const;

    VectorView vector_slice(const std::vector<int> &index);
    VectorView vector_slice(int x1);
    VectorView vector_slice(int x1, int x2);
    VectorView vector_slice(int x1, int x2, int x3);
    VectorView vector_slice(int x1, int x2, int x3, int x4);
    VectorView vector_slice(int x1, int x2, int x3, int x4, int x5);
    VectorView vector_slice(int x1, int x2, int x3, int x4, int x5, int x6);

    Array apply_scalar_function(
        const std::vector<int> &apply_over_dims,
        const std::function<double(const ConstArrayView &view)> &functor) const {
      return ConstArrayView(*this).apply_scalar_function(
          apply_over_dims, functor);
    }

    // The easiest way to iterate over all the array elements is to iterate over
    // data_.
    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

    // The array iterator interface is helpful when you need to iterate over the
    // elements, but also have access to the vector of position indices.
    ArrayIterator abegin();
    ArrayIterator aend();
    ConstArrayIterator abegin() const;
    ConstArrayIterator aend() const;

    // Args:
    //   dims: An array of integers over which to compute the sum.  The length
    //     of dims must be no greater than the number of dimensions of the
    //     array.  The elements of dims must be unique, non-negative, and less
    //     than this->ndim().
    //
    // Returns:
    //   The dimensions of the returned array correspond to this->dim() with any
    //   dimensions in 'dims' removed.
    Array sum(const std::vector<int> &dims) const;
    double scalar_sum() const;
    double sum() const {return scalar_sum();}

    ostream &print(ostream &out) const override;
    std::string to_string() const override;

    using ArrayBase::operator[];

    // Return the i'th element of the Array.  This is storage order dependent,
    // but for single dimension arrays it allows access to the underlying data
    // vector.
    double operator[](int i) const { return data_[i]; }
    double &operator[](int i) { return data_[i]; }

   private:
    Vector data_;
  };

  Array operator+(const Array &arr, double x);
  Array operator-(const Array &arr, double x);
  Array operator*(const Array &arr, double x);
  Array operator/(const Array &arr, double x);

  Array operator+(double x, const Array &arr);
  Array operator-(double x, const Array &arr);
  Array operator*(double x, const Array &arr);
  Array operator/(double x, const Array &arr);


  // A mapping from a multivariate index to a column-major array storage format.
  //
  // Template Args:
  //   INT: The desired storage type for integer values.
  //
  // Args:
  //   index:  The collection of indices pointing to a single array element.
  //   dim:  The dimensions (aka "extent" or "shape") of the array.
  //   strides: The number of steps between successive elements in the same
  //     dimension.  strides[i] is the number of steps you must advance in the
  //     raw data structure to increment the i'th index by one.
  //
  // Returns:
  //   The position of the element at 'index' in the underlying memory storage
  //   block.
  template <class INT = int>
  inline INT array_index(const std::vector<INT> &index,
                         const std::vector<INT> &dim,
                         const std::vector<INT> &strides) {
    if (index.size() != dim.size()) {
      std::ostringstream err;
      err << "Wrong number of dimensions passed to "
          << "ConstArrayBase::operator[]."
          << "  Expected " << dim.size() << " got " << index.size() << "."
          << endl;
      report_error(err.str());
    }
    INT pos = 0;
    for (INT i = 0; i < dim.size(); ++i) {
      INT ind = index[i];
      if (ind < 0 || ind >= dim[i]) {
        std::ostringstream err;
        err << "Index " << i
            << " out of bounds in ConstArrayBase::operator[]."
            << " Value passed = " << ind << " legal range: [0, " << dim[i] - 1
            << "]." << endl;
        report_error(err.str());
      }
      pos += index[i] * strides[i];
    }
    return pos;
  }

  // Compute the vector of strides for an array stored in dense vector storage.
  // Args:
  //   dims: The dimensions (aka "extents" or "shape") of the array to be
  //     stored.
  //
  // Returns:
  //   A vector of strides, where strides[i] is the number of positions in raw
  //   storage between array elements in dimension i.  For example, in a 2-way
  //   array (a matrix), moving to the next row (dimension 0) involves moving 1
  //   step, while moving to the next column (dimension 1) means moving "number
  //   of rows" spaces.
  template <class INT>
  std::vector<INT> compute_array_strides(const std::vector<INT> &dims) {
    std::vector<INT> strides(dims);
    INT last_stride = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
      strides[i] = last_stride;
      last_stride *= dims[i];
    }
    return strides;
  }

  //===========================================================================
  // The GenericArray class needs an iterator.  To make things work we need to
  // forward declare the GenericArray template class, declare the iterator
  // class, and instantiate the iterator's member functions that depend on
  // GenericArray after the GenericArray definition.
  template<class VALUE, class INT> class GenericArray;

  template <class VALUE, class INT>
  class GenericArrayIterator {
   public:
    GenericArrayIterator(GenericArray<VALUE, INT> *host);

    VALUE& operator*();
    bool operator==(const GenericArrayIterator &rhs) const;
    bool operator!=(const GenericArrayIterator &rhs) const;

    GenericArrayIterator &operator++() {
      ++position_;
      return *this;
    }

    const std::vector<int> &position() const {
      return position_.position();
    }

    void set_position(const std::vector<int> &position) {
      position_.set_position(position);
    }

    GenericArrayIterator &set_to_end() {
      position_.set_to_end();
      return *this;
    }

   private:
    GenericArray<VALUE, INT> *host_;
    ArrayPositionManager position_;
  };

  template <class VALUE, class INT>
  class GenericArrayConstIterator {
   public:
    GenericArrayConstIterator(const GenericArray<VALUE, INT> *host);

    const VALUE& operator*() const;
    bool operator==(const GenericArrayConstIterator &rhs) const;
    bool operator!=(const GenericArrayConstIterator &rhs) const;

    GenericArrayConstIterator &operator++() {
      ++position_;
      return *this;
    }

    const std::vector<int> &position() const {
      return position_.position();
    }

    void set_position(const std::vector<int> &position) {
      position_.set_position(position);
    }

    GenericArrayConstIterator &set_to_end() {
      position_.set_to_end();
      return *this;
    }

   private:
    GenericArray<VALUE, INT> *host_;
    ArrayPositionManager position_;
  };


  //===========================================================================
  // A generic array of objects.
  //
  // Template Args:
  //   VALUE:  The type of value stored in the array.
  //   INT:  The type of integer used to index the array.
  template <class VALUE, class INT = int>
  class GenericArray {
   public:

    GenericArray() {}

    GenericArray(const std::vector<INT> &dims, const std::vector<VALUE> &data)
        : dims_(dims),
          data_(data)
    {
      check_dimensions(dims_, data_.size());
      strides_ = compute_array_strides(dims);
    }

    bool empty() const {return dims_.empty();}

    int ndim() const { return dims_.size(); }
    int dim(int i) const { return dims_[i]; }
    const std::vector<int> &dim() const { return dims_; }

    // stride(i) is the number of steps you must advance in data()
    // to increment the i'th index by one.
    int stride(int i) const { return strides_[i]; }
    const std::vector<int> &strides() const { return strides_; }

    // size() is the number of elements stored in the array.  It is
    // the product of dims_;
    int size() const;

    // Iterators
    GenericArrayIterator<VALUE, INT> begin() {
      return GenericArrayIterator<VALUE, INT>(this);
    }

    GenericArrayIterator<VALUE, INT> end() {
      GenericArrayIterator<VALUE, INT> it(this);
      it.set_to_end();
      return it;
    }

    GenericArrayConstIterator<VALUE, INT> begin() const {
      return GenericArrayConstIterator<VALUE, INT>(this);
    }

    GenericArrayConstIterator<VALUE, INT> end() const {
      GenericArrayConstIterator<VALUE, INT> it(this);
      it.set_to_end();
      return it;
    }

    void check_dimensions(const std::vector<INT> &dims, size_t size) const {
      INT product = 1;
      for (const auto &el : dims) {
        product *= el;
      }
      if (product != size) {
        std::ostringstream err;
        err << "Generic array failed dimension check.  The product of [";
        for (size_t i = 0; i < dims.size(); ++i) {
          err << dims[i];
          if (i + 1 < dims.size()) {
            err << ", ";
          }
        }
        err << "] is " << product << " but the required size is "
            << size << ".";
        report_error(err.str());
      }
    }

    VALUE &operator[](const std::vector<INT> &index) {
      return data_[array_index<INT>(index, dims_, strides_)];
    }

    const VALUE &operator[](const std::vector<INT> &index) const {
      return data_[array_index<INT>(index, dims_, strides_)];
    }

   private:
    std::vector<INT> dims_;
    std::vector<VALUE> data_;
    std::vector<INT> strides_;
  };

  //===========================================================================
  // Member function definitions for GenericArrayIterator and
  // GenericArrayConstIterator.

  template<class VALUE, class INT>
  GenericArrayIterator<VALUE, INT>::GenericArrayIterator(
      GenericArray<VALUE, INT> *host):
      host_(host),
      position_(host->dim())
  {}

  template<class VALUE, class INT>
  VALUE &GenericArrayIterator<VALUE, INT>::operator*() {
    if (position_.at_end()) {
      report_error("GenericArrayIterator dereference past end of data.");
    }
    return (*host_)[position_.position()];
  }

  template<class VALUE, class INT>
  bool GenericArrayIterator<VALUE, INT>::operator==(
      const GenericArrayIterator &rhs) const {
    return (host_ == rhs.host_) && (position_ == rhs.position_);
  }

  template<class VALUE, class INT>
  bool GenericArrayIterator<VALUE, INT>::operator!=(
      const GenericArrayIterator &rhs) const {
    return (host_ != rhs.host_) || (position_ != rhs.position_);
  }

  // Now for the Const case.
  template<class VALUE, class INT>
  GenericArrayConstIterator<VALUE, INT>::GenericArrayConstIterator(
      const GenericArray<VALUE, INT> *host):
      host_(host),
      position_(host->dim())
  {}

  template<class VALUE, class INT>
  const VALUE &GenericArrayConstIterator<VALUE, INT>::operator*() const {
    if (position_.at_end()) {
      report_error("GenericArrayConstIterator dereference past end of data.");
    }
    return (*host_)[position_.position()];
  }

  template<class VALUE, class INT>
  bool GenericArrayConstIterator<VALUE, INT>::operator==(
      const GenericArrayConstIterator &rhs) const {
    return (host_ == rhs.host_) && (position_ == rhs.position_);
  }

  template<class VALUE, class INT>
  bool GenericArrayConstIterator<VALUE, INT>::operator!=(
      const GenericArrayConstIterator &rhs) const {
    return (host_ != rhs.host_) || (position_ != rhs.position_);
  }

  //===========================================================================
  // Free functions
  //===========================================================================

  double max(const ConstArrayView &view);
  inline double max(const Array &array) {
    return max(ConstArrayView(array));
  }

  double min(const ConstArrayView &view);
  inline double min(const Array &array) {
    return min(ConstArrayView(array));
  }

  // A functor for finding the index of the maximal value in a 1-D array.  If
  // there are multiple copies of the same max value, then ties are broken
  // uniformly at random.
  class ArrayArgMax {
   public:
    ArrayArgMax(RNG &rng = ::BOOM::GlobalRng::rng);

    size_t operator()(const ConstArrayView &view) const;

   private:
    mutable RNG rng_;
    mutable std::vector<int> candidates_;
  };

}  // namespace BOOM

#endif  // BOOM_ARRAY_HPP
