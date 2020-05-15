// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2011 Steven L. Scott

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

#include "LinAlg/Array.hpp"
#include <algorithm>
#include <cstdarg>
#include <sstream>
#include "cpputil/report_error.hpp"
#include "cpputil/ToString.hpp"
#include "distributions.hpp"


namespace BOOM {

  ConstArrayBase::ConstArrayBase() {}

  ConstArrayBase::ConstArrayBase(const std::vector<int> &dims) : dims_(dims) {
    compute_strides();
  }

  ConstArrayBase::ConstArrayBase(const std::vector<int> &dims,
                                 const std::vector<int> &strides)
      : dims_(dims), strides_(strides) {}

  void ConstArrayBase::compute_strides() {
    strides_.resize(dims_.size());
    int last_stride = 1;
    for (int i = 0; i < dims_.size(); ++i) {
      strides_[i] = last_stride;
      last_stride *= dims_[i];
    }
  }

  namespace {
    // Utility functions for implementing indexing and slicing.

    // takes N int arguments and returns an N-vector of ints.  This
    // function is not exposed to the world because it is
    // potentially insecure if the wrong number of arguments are
    // supplied.
    template <int N>
    std::vector<int> create_index(int first, ...) {
      va_list ap;
      std::vector<int> ans(N);
      ans[0] = first;

      va_start(ap, first);
      for (int i = 1; i < N; ++i) {
        int value(va_arg(ap, int));
        ans[i] = value;
      }
      va_end(ap);
      return (ans);
    }

    // Returns the position in the column-major array
    inline int array_index(const std::vector<int> &index,
                           const std::vector<int> &dim,
                           const std::vector<int> &strides) {
      if (index.size() != dim.size()) {
        std::ostringstream err;
        err << "Wrong number of dimensions passed to "
            << "ConstArrayBase::operator[]."
            << "  Expected " << dim.size() << " got " << index.size() << "."
            << endl;
        report_error(err.str());
      }
      int pos = 0;
      for (int i = 0; i < dim.size(); ++i) {
        int ind = index[i];
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

    template <class V>
    bool vector_compare(const V &v, const ConstArrayBase &array) {
      int n = array.size();
      if ((array.ndim() != 1) || (n != v.size())) return false;
      const double *x = array.data();
      for (int i = 0; i < n; ++i) {
        if (x[i] != v[i]) return false;
      }
      return true;
    }

    inline void check_slice_size(const std::vector<int> &index,
                                 const std::vector<int> &dims) {
      if (index.size() == dims.size()) return;

      std::ostringstream msg;
      msg << "Array::slice expects an argument of length " << dims.size()
          << " but was passed an argument of length " << index.size() << " : [";
      for (int i = 0; i < index.size(); ++i) {
        msg << index[i];
        if (i + 1 < index.size()) msg << ",";
      }
      msg << "]" << endl;
      report_error(msg.str());
    }

    template <class RETURN_TYPE, class INPUT_TYPE>
    RETURN_TYPE template_slice_array(INPUT_TYPE host_data,
                                     const std::vector<int> &index,
                                     const std::vector<int> &host_dims,
                                     const std::vector<int> &host_strides) {
      check_slice_size(index, host_dims);
      std::vector<int> view_dims;
      std::vector<int> view_strides;
      std::vector<int> view_initial_position(index.size());
      for (int i = 0; i < index.size(); ++i) {
        if (index[i] < 0) {
          view_dims.push_back(host_dims[i]);
          view_strides.push_back(host_strides[i]);
          view_initial_position[i] = 0;
        } else {
          view_initial_position[i] = index[i];
        }
      }
      INPUT_TYPE view_data =
          host_data +
          array_index(view_initial_position, host_dims, host_strides);
      return RETURN_TYPE(view_data, view_dims, view_strides);
    }

    inline ArrayView slice_array(double *host_data,
                                 const std::vector<int> &index,
                                 const std::vector<int> &host_dims,
                                 const std::vector<int> &host_strides) {
      return template_slice_array<ArrayView, double *>(
          host_data, index, host_dims, host_strides);
    }

    inline ConstArrayView slice_const_array(
        const double *host_data, const std::vector<int> &index,
        const std::vector<int> &host_dims,
        const std::vector<int> &host_strides) {
      return template_slice_array<ConstArrayView, const double *>(
          host_data, index, host_dims, host_strides);
    }

    // Return a vector-like slice of one dimension of the input array.
    // Template Args:
    //   RETURN_TYPE: either ConstVectorView or VectorView.
    //   INPUT_TYPE: either 'const double *' or 'double *'.
    // Args:
    //   host_data: A pointer to the data owned by the input array or
    //     view.
    //   index: The location at which to slice the array.  Exactly one
    //     element must be negative.  That element is the dimension of
    //     the slice.  For example: in a 3-way array, if index == {2,
    //     -1, 7} then the return is a view into elements {2, 0, 7},
    //     {2, 1, 7}, {2, 2, 7}, ...
    //   host_dims:  The dimensions of the input array.
    //   host_strides:  The strides of the input array.
    template <class RETURN_TYPE, class INPUT_TYPE>
    RETURN_TYPE template_vector_slice_array(
        INPUT_TYPE host_data, const std::vector<int> &index,
        const std::vector<int> &host_dims,
        const std::vector<int> &host_strides) {
      int ndim = host_dims.size();
      check_slice_size(index, host_dims);
      std::vector<int> initial_position(ndim);
      int which_slice = -1;
      for (int i = 0; i < ndim; ++i) {
        if (index[i] >= 0) {
          initial_position[i] = index[i];
        } else {
          if (which_slice >= 0) {
            report_error(
                "multiple slicing indices were "
                "provided in Array::vector_slice.");
          }
          which_slice = i;
          initial_position[i] = 0;
        }
      }
      int pos = array_index(initial_position, host_dims, host_strides);
      RETURN_TYPE ans(
          host_data + pos, host_dims[which_slice], host_strides[which_slice]);
      return ans;
    }

    // Syntactic sugar for getting ConstVectorView slices of const
    // arrays or views.
    inline ConstVectorView vector_slice_const_array(
        const double *host_data, const std::vector<int> &index,
        const std::vector<int> &host_dims,
        const std::vector<int> &host_strides) {
      return template_vector_slice_array<ConstVectorView, const double *>(
          host_data, index, host_dims, host_strides);
    }

    // Syntactic sugar for getting VectorView slices of non-const
    // arrays or views.
    inline VectorView vector_slice_array(double *host_data,
                                         const std::vector<int> &index,
                                         const std::vector<int> &host_dims,
                                         const std::vector<int> &host_strides) {
      return template_vector_slice_array<VectorView, double *>(
          host_data, index, host_dims, host_strides);
    }

  }  // namespace

  double ConstArrayBase::operator[](const std::vector<int> &index) const {
    int pos = array_index(index, dims_, strides_);
    return data()[pos];
  }

  int ConstArrayBase::size() const {
    int ans = 1;
    for (int i = 0; i < dims_.size(); ++i) ans *= dims_[i];
    return ans;
  }

  bool ConstArrayBase::operator==(const Vector &rhs) const {
    return vector_compare(rhs, *this);
  }
  bool ConstArrayBase::operator==(const VectorView &rhs) const {
    return vector_compare(rhs, *this);
  }
  bool ConstArrayBase::operator==(const ConstVectorView &rhs) const {
    return vector_compare(rhs, *this);
  }
  bool ConstArrayBase::operator==(const Matrix &rhs) const {
    if (ndim() != 2 || dim(0) != rhs.nrow() || dim(1) != rhs.ncol())
      return false;
    const double *x(this->data());
    const double *y(rhs.data());
    int n = rhs.size();
    for (int i = 0; i < n; ++i) {
      if (x[i] != y[i]) return false;
    }
    return true;
  }

  bool ConstArrayBase::operator==(const ConstArrayBase &rhs) const {
    if (&rhs == this) return true;
    if (dim() != rhs.dim()) return false;
    ConstArrayIterator left(this);
    ConstArrayIterator right(&rhs);
    int n = size();
    for (int i = 0; i < n; ++i) {
      if (*left != *right) return false;
      ++left;
      ++right;
    }
    return true;
  }

  void ConstArrayBase::reset_dims(const std::vector<int> &dims) {
    dims_ = dims;
  }
  void ConstArrayBase::reset_strides(const std::vector<int> &strides) {
    strides_ = strides;
  }

  std::vector<int> ConstArrayBase::index1(int x1) {
    return std::vector<int>(1, x1);
  }
  std::vector<int> ConstArrayBase::index2(int x1, int x2) {
    return create_index<2>(x1, x2);
  }
  std::vector<int> ConstArrayBase::index3(int x1, int x2, int x3) {
    return create_index<3>(x1, x2, x3);
  }
  std::vector<int> ConstArrayBase::index4(int x1, int x2, int x3, int x4) {
    return create_index<4>(x1, x2, x3, x4);
  }
  std::vector<int> ConstArrayBase::index5(int x1, int x2, int x3, int x4,
                                          int x5) {
    return create_index<5>(x1, x2, x3, x4, x5);
  }
  std::vector<int> ConstArrayBase::index6(int x1, int x2, int x3, int x4,
                                          int x5, int x6) {
    return create_index<6>(x1, x2, x3, x4, x5, x6);
  }

  double ConstArrayBase::operator()(int x1) const {
    return (*this)[index1(x1)];
  }
  double ConstArrayBase::operator()(int x1, int x2) const {
    return (*this)[index2(x1, x2)];
  }
  double ConstArrayBase::operator()(int x1, int x2, int x3) const {
    return (*this)[index3(x1, x2, x3)];
  }
  double ConstArrayBase::operator()(int x1, int x2, int x3, int x4) const {
    return (*this)[index4(x1, x2, x3, x4)];
  }
  double ConstArrayBase::operator()(int x1, int x2, int x3, int x4,
                                    int x5) const {
    return (*this)[index5(x1, x2, x3, x4, x5)];
  }
  double ConstArrayBase::operator()(int x1, int x2, int x3, int x4, int x5,
                                    int x6) const {
    return (*this)[index6(x1, x2, x3, x4, x5, x6)];
  }

  Matrix ConstArrayBase::to_matrix() const {
    if (dims_.size() > 2) {
      report_error("to_matrix() called from array with 3 or more dimensions.");
    } else if (dims_.size() == 2) {
      Matrix ans(dims_[0], dims_[1]);
      for (int i = 0; i < dims_[0]; ++i) {
        for (int j = 0; j < dims_[1]; ++j) {
          ans(i, j) = (*this)(i, j);
        }
      }
      return ans;
    } else if (dims_.size() == 1) {
      Matrix ans(dims_[0], 1);
      for (int i = 0; i < dims_[0]; ++i) {
        ans(i, 0) = (*this)(i);
      }
      return ans;
    } else {
      report_error("to_matrix() called on an array with empty dims_.");
    }
    return Matrix(0, 0);  // Never get here.
  }

  namespace {
    template <class ARRAY>
    ostream &print_array(ostream &out, const ARRAY &array) {
      if (array.ndim() == 1) {
        out << array.vector_slice(-1) << std::endl;
      } else {
        int outer_dim = array.dim()[0];
        for (int i = 0; i < outer_dim; ++i) {
          std::vector<int> index(array.ndim(), -1);
          index[0] = i;
          array.slice(index).print(out);
        }
      }
      return out;
    }

    template <class ARRAY>
    std::string array_to_string(const ARRAY &array) {
      std::ostringstream out;
      print_array(out, array);
      return out.str();
    }

    bool all_ones(const std::vector<int> &v) {
      for (const auto &el : v) {
        if (el != 1) {
          return false;
        }
      }
      return true;
    }

  }  // namespace


  //======================================================================
  ArrayBase::ArrayBase() {}

  ArrayBase::ArrayBase(const std::vector<int> &dims) : ConstArrayBase(dims) {}

  ArrayBase::ArrayBase(const std::vector<int> &dims,
                       const std::vector<int> &strides)
      : ConstArrayBase(dims, strides) {}

  double &ArrayBase::operator[](const std::vector<int> &index) {
    int pos = array_index(index, dim(), strides());
    return data()[pos];
  }

  double &ArrayBase::operator()(int x1) { return (*this)[index1(x1)]; }
  double &ArrayBase::operator()(int x1, int x2) {
    return (*this)[index2(x1, x2)];
  }
  double &ArrayBase::operator()(int x1, int x2, int x3) {
    return (*this)[index3(x1, x2, x3)];
  }
  double &ArrayBase::operator()(int x1, int x2, int x3, int x4) {
    return (*this)[index4(x1, x2, x3, x4)];
  }
  double &ArrayBase::operator()(int x1, int x2, int x3, int x4, int x5) {
    return (*this)[index5(x1, x2, x3, x4, x5)];
  }
  double &ArrayBase::operator()(int x1, int x2, int x3, int x4, int x5,
                                int x6) {
    return (*this)[index6(x1, x2, x3, x4, x5, x6)];
  }

  //======================================================================
  ArrayView::ArrayView(Array &a) : ArrayBase(a.dim()), data_(a.data()) {}

  ArrayView::ArrayView(double *data, const std::vector<int> &dims)
      : ArrayBase(dims), data_(data) {}

  ArrayView::ArrayView(double *data, const std::vector<int> &dims,
                       const std::vector<int> &strides)
      : ArrayBase(dims, strides), data_(data) {}

  void ArrayView::reset(double *data, const std::vector<int> &dims) {
    data_ = data;
    reset_dims(dims);
    compute_strides();
  }

  void ArrayView::reset(double *data, const std::vector<int> &dims,
                        const std::vector<int> &strides) {
    data_ = data;
    reset_dims(dims);
    reset_strides(strides);
  }

  ArrayView ArrayView::slice(const std::vector<int> &index) {
    return slice_array(data(), index, dim(), strides());
  }
  ConstArrayView ArrayView::slice(const std::vector<int> &index) const {
    return slice_const_array(data(), index, dim(), strides());
  }
  ArrayView ArrayView::slice(int x1) { return this->slice(index1(x1)); }
  ArrayView ArrayView::slice(int x1, int x2) {
    return this->slice(index2(x1, x2));
  }
  ArrayView ArrayView::slice(int x1, int x2, int x3) {
    return this->slice(index3(x1, x2, x3));
  }
  ArrayView ArrayView::slice(int x1, int x2, int x3, int x4) {
    return this->slice(index4(x1, x2, x3, x4));
  }
  ArrayView ArrayView::slice(int x1, int x2, int x3, int x4, int x5) {
    return this->slice(index5(x1, x2, x3, x4, x5));
  }
  ArrayView ArrayView::slice(int x1, int x2, int x3, int x4, int x5, int x6) {
    return this->slice(index6(x1, x2, x3, x4, x5, x6));
  }

  ConstArrayView ArrayView::slice(int x1) const {
    return this->slice(index1(x1));
  }
  ConstArrayView ArrayView::slice(int x1, int x2) const {
    return this->slice(index2(x1, x2));
  }
  ConstArrayView ArrayView::slice(int x1, int x2, int x3) const {
    return this->slice(index3(x1, x2, x3));
  }
  ConstArrayView ArrayView::slice(int x1, int x2, int x3, int x4) const {
    return this->slice(index4(x1, x2, x3, x4));
  }
  ConstArrayView ArrayView::slice(int x1, int x2, int x3, int x4,
                                  int x5) const {
    return this->slice(index5(x1, x2, x3, x4, x5));
  }
  ConstArrayView ArrayView::slice(int x1, int x2, int x3, int x4, int x5,
                                  int x6) const {
    return this->slice(index6(x1, x2, x3, x4, x5, x6));
  }

  ConstVectorView ArrayView::vector_slice(const std::vector<int> &index) const {
    return vector_slice_const_array(data(), index, dim(), strides());
  }
  ConstVectorView ArrayView::vector_slice(int x1) const {
    return this->vector_slice(index1(x1));
  }
  ConstVectorView ArrayView::vector_slice(int x1, int x2) const {
    return this->vector_slice(index2(x1, x2));
  }
  ConstVectorView ArrayView::vector_slice(int x1, int x2, int x3) const {
    return this->vector_slice(index3(x1, x2, x3));
  }
  ConstVectorView ArrayView::vector_slice(int x1, int x2, int x3,
                                          int x4) const {
    return this->vector_slice(index4(x1, x2, x3, x4));
  }
  ConstVectorView ArrayView::vector_slice(int x1, int x2, int x3, int x4,
                                          int x5) const {
    return this->vector_slice(index5(x1, x2, x3, x4, x5));
  }
  ConstVectorView ArrayView::vector_slice(int x1, int x2, int x3, int x4,
                                          int x5, int x6) const {
    return this->vector_slice(index6(x1, x2, x3, x4, x5, x6));
  }

  VectorView ArrayView::vector_slice(const std::vector<int> &index) {
    return vector_slice_array(data(), index, dim(), strides());
  }
  VectorView ArrayView::vector_slice(int x1) {
    return this->vector_slice(index1(x1));
  }
  VectorView ArrayView::vector_slice(int x1, int x2) {
    return this->vector_slice(index2(x1, x2));
  }
  VectorView ArrayView::vector_slice(int x1, int x2, int x3) {
    return this->vector_slice(index3(x1, x2, x3));
  }
  VectorView ArrayView::vector_slice(int x1, int x2, int x3, int x4) {
    return this->vector_slice(index4(x1, x2, x3, x4));
  }
  VectorView ArrayView::vector_slice(int x1, int x2, int x3, int x4, int x5) {
    return this->vector_slice(index5(x1, x2, x3, x4, x5));
  }
  VectorView ArrayView::vector_slice(int x1, int x2, int x3, int x4, int x5,
                                     int x6) {
    return this->vector_slice(index6(x1, x2, x3, x4, x5, x6));
  }

  ArrayIterator ArrayView::begin() { return ArrayIterator(this); }
  ConstArrayIterator ArrayView::begin() const {
    return ConstArrayIterator(this);
  }

  ArrayIterator ArrayView::end() {
    ArrayIterator ans(this);
    ans.set_to_end();
    return ans;
  }

  ConstArrayIterator ArrayView::end() const {
    ConstArrayIterator ans(this);
    ans.set_to_end();
    return ans;
  }

  ArrayView &ArrayView::operator=(const Array &a) {
    if (dim() != a.dim()) {
      report_error("wrong size of Array supplied to ArrayView::operator= ");
    }
    std::copy(a.begin(), a.end(), begin());
    return *this;
  }
  ArrayView &ArrayView::operator=(const ArrayView &a) {
    if (&a == this) return *this;
    if (dim() != a.dim()) {
      report_error("wrong size of Array supplied to ArrayView::operator= ");
    }
    std::copy(a.begin(), a.end(), begin());
    return *this;
  }
  ArrayView &ArrayView::operator=(const ConstArrayView &a) {
    if (dim() != a.dim()) {
      report_error("wrong size of Array supplied to ArrayView::operator= ");
    }
    std::copy(a.begin(), a.end(), begin());
    return *this;
  }
  ArrayView &ArrayView::operator=(const Matrix &a) {
    if (ndim() != 2 || nrow(a) != dim(0) || ncol(a) != dim(1)) {
      report_error("wrong size of Array supplied to ArrayView::operator= ");
    }
    std::copy(a.begin(), a.end(), begin());
    return *this;
  }
  ArrayView &ArrayView::operator=(const Vector &a) {
    if (ndim() != 1 || a.size() != dim(0)) {
      report_error("wrong size of Array supplied to ArrayView::operator= ");
    }
    std::copy(a.begin(), a.end(), begin());
    return *this;
  }
  ArrayView &ArrayView::operator=(const VectorView &a) {
    if (ndim() != 1 || a.size() != dim(0)) {
      report_error("wrong size of Array supplied to ArrayView::operator= ");
    }
    std::copy(a.begin(), a.end(), begin());
    return *this;
  }
  ArrayView &ArrayView::operator=(const ConstVectorView &a) {
    if (ndim() != 1 || a.size() != dim(0)) {
      report_error("wrong size of Array supplied to ArrayView::operator= ");
    }
    std::copy(a.begin(), a.end(), begin());
    return *this;
  }

  ostream &ArrayView::print(ostream &out) const {
    return print_array(out, *this);
  }
  std::string ArrayView::to_string() const {
    return array_to_string(*this);
  }

  //======================================================================
  ConstArrayView::ConstArrayView(const Array &rhs)
      : ConstArrayBase(rhs), data_(rhs.data()) {}

  ConstArrayView::ConstArrayView(const double *data,
                                 const std::vector<int> &dims)
      : ConstArrayBase(dims), data_(data) {}

  ConstArrayView::ConstArrayView(const double *data,
                                 const std::vector<int> &dims,
                                 const std::vector<int> &strides)
      : ConstArrayBase(dims, strides), data_(data) {}

  ConstArrayView::ConstArrayView(const ConstArrayBase &rhs)
      : ConstArrayBase(rhs), data_(rhs.data()) {}

  void ConstArrayView::reset(const double *data, const std::vector<int> &dims) {
    data_ = data;
    reset_dims(dims);
    compute_strides();
  }

  void ConstArrayView::reset(const double *data, const std::vector<int> &dims,
                             const std::vector<int> &strides) {
    data_ = data;
    reset_dims(dims);
    reset_strides(strides);
  }

  ConstArrayView ConstArrayView::slice(const std::vector<int> &index) const {
    return slice_const_array(data(), index, dim(), strides());
  }

  ConstArrayView ConstArrayView::slice(int x1) const {
    return this->slice(index1(x1));
  }
  ConstArrayView ConstArrayView::slice(int x1, int x2) const {
    return this->slice(index2(x1, x2));
  }
  ConstArrayView ConstArrayView::slice(int x1, int x2, int x3) const {
    return this->slice(index3(x1, x2, x3));
  }
  ConstArrayView ConstArrayView::slice(int x1, int x2, int x3, int x4) const {
    return this->slice(index4(x1, x2, x3, x4));
  }
  ConstArrayView ConstArrayView::slice(int x1, int x2, int x3, int x4,
                                       int x5) const {
    return this->slice(index5(x1, x2, x3, x4, x5));
  }
  ConstArrayView ConstArrayView::slice(int x1, int x2, int x3, int x4, int x5,
                                       int x6) const {
    return this->slice(index6(x1, x2, x3, x4, x5, x6));
  }

  ConstVectorView ConstArrayView::vector_slice(
      const std::vector<int> &index) const {
    return vector_slice_const_array(data(), index, dim(), strides());
  }
  ConstVectorView ConstArrayView::vector_slice(int x1) const {
    return this->vector_slice(index1(x1));
  }
  ConstVectorView ConstArrayView::vector_slice(int x1, int x2) const {
    return this->vector_slice(index2(x1, x2));
  }
  ConstVectorView ConstArrayView::vector_slice(int x1, int x2, int x3) const {
    return this->vector_slice(index3(x1, x2, x3));
  }
  ConstVectorView ConstArrayView::vector_slice(int x1, int x2, int x3,
                                               int x4) const {
    return this->vector_slice(index4(x1, x2, x3, x4));
  }
  ConstVectorView ConstArrayView::vector_slice(int x1, int x2, int x3, int x4,
                                               int x5) const {
    return this->vector_slice(index5(x1, x2, x3, x4, x5));
  }
  ConstVectorView ConstArrayView::vector_slice(int x1, int x2, int x3, int x4,
                                               int x5, int x6) const {
    return this->vector_slice(index6(x1, x2, x3, x4, x5, x6));
  }

  ConstArrayIterator ConstArrayView::begin() const {
    return ConstArrayIterator(this);
  }

  ConstArrayIterator ConstArrayView::end() const {
    ConstArrayIterator ans(this);
    ans.set_to_end();
    return ans;
  }

  ostream & ConstArrayView::print(ostream &out) const {
    return print_array(out, *this);
  }
  std::string ConstArrayView::to_string() const {
    return array_to_string(*this);
  }

  //======================================================================
  Array::Array(const std::vector<int> &dims, double initial_value)
      : ArrayBase(dims), data_(ConstArrayBase::size(), initial_value) {}

  Array::Array(const std::vector<int> &dims, const std::vector<double> &data)
      : ArrayBase(dims), data_(data) {
    if (data_.size() != size()) {
      std::ostringstream err;
      err << "Wrong size data argument given to Array() constructor.  Expected "
          << size() << " elements, based on supplied dimensions: [ ";
      for (int i = 0; i < dims.size(); ++i) {
        err << dims[i] << " ";
      }
      err << "].  Got " << data.size() << ".";
      report_error(err.str());
    }
  }

  Array::Array(const std::vector<int> &dims, const double *data)
      : ArrayBase(dims) {
    size_t length = 1;
    for (int i = 0; i < dims.size(); ++i) {
      length *= dims[i];
    }
    data_.assign(data, data + length);
  }

  Array &Array::operator=(const ArrayView &a) {
    ArrayView(*this) = a;
    return *this;
  }
  Array &Array::operator=(const ConstArrayView &a) {
    ArrayView(*this) = a;
    return *this;
  }
  Array &Array::operator=(const Matrix &a) {
    ArrayView(*this) = a;
    return *this;
  }
  Array &Array::operator=(const Vector &a) {
    ArrayView(*this) = a;
    return *this;
  }
  Array &Array::operator=(const VectorView &a) {
    ArrayView(*this) = a;
    return *this;
  }
  Array &Array::operator=(const ConstVectorView &a) {
    ArrayView(*this) = a;
    return *this;
  }
  Array &Array::operator=(double x) {
    data_ = x;
    return *this;
  }


  Array &Array::operator+=(const Array &rhs) {
    if (dim() == rhs.dim()) {
      data_ += rhs.data_;
    } else {
      report_error("Can't add arrays with different dims.");
    }
    return *this;
  }

  Array &Array::operator+=(const ConstArrayView &rhs) {
    if (dim() !=  rhs.dim()) {
      std::ostringstream err;
      err << "Attempt to add an array of dimension "
          << ToString(rhs.dim()) << " to an array of dimension "
          << ToString(dim()) << ".";
      report_error(err.str());
    }

    if (all_ones(rhs.strides())) {
      data_ += ConstVectorView(rhs.data(), rhs.size(), 1);
    } else {
      auto it = begin();
      for (auto cit = rhs.begin(); cit != rhs.end(); ++cit) {
        *it += *cit;
        ++it;
      }
    }
    return *this;
  }

  int ConstArrayBase::product(const std::vector<int> &dims) {
    int ans = 1;
    for (int i = 0; i < dims.size(); ++i) {
      ans *= dims[i];
    }
    return ans;
  }

  void Array::randomize() {
    for (iterator it = begin(); it != end(); ++it) {
      *it = runif();
    }
  }

  ArrayView Array::slice(const std::vector<int> &index) {
    return slice_array(data(), index, dim(), strides());
  }
  ConstArrayView Array::slice(const std::vector<int> &index) const {
    return slice_const_array(data(), index, dim(), strides());
  }

  ArrayView Array::slice(int x1) { return this->slice(index1(x1)); }
  ArrayView Array::slice(int x1, int x2) { return this->slice(index2(x1, x2)); }
  ArrayView Array::slice(int x1, int x2, int x3) {
    return this->slice(index3(x1, x2, x3));
  }
  ArrayView Array::slice(int x1, int x2, int x3, int x4) {
    return this->slice(index4(x1, x2, x3, x4));
  }
  ArrayView Array::slice(int x1, int x2, int x3, int x4, int x5) {
    return this->slice(index5(x1, x2, x3, x4, x5));
  }
  ArrayView Array::slice(int x1, int x2, int x3, int x4, int x5, int x6) {
    return this->slice(index6(x1, x2, x3, x4, x5, x6));
  }

  ConstArrayView Array::slice(int x1) const { return this->slice(index1(x1)); }
  ConstArrayView Array::slice(int x1, int x2) const {
    return this->slice(index2(x1, x2));
  }
  ConstArrayView Array::slice(int x1, int x2, int x3) const {
    return this->slice(index3(x1, x2, x3));
  }
  ConstArrayView Array::slice(int x1, int x2, int x3, int x4) const {
    return this->slice(index4(x1, x2, x3, x4));
  }
  ConstArrayView Array::slice(int x1, int x2, int x3, int x4, int x5) const {
    return this->slice(index5(x1, x2, x3, x4, x5));
  }
  ConstArrayView Array::slice(int x1, int x2, int x3, int x4, int x5,
                              int x6) const {
    return this->slice(index6(x1, x2, x3, x4, x5, x6));
  }

  ConstVectorView Array::vector_slice(const std::vector<int> &index) const {
    ConstArrayView view(*this);
    return view.vector_slice(index);
  }
  ConstVectorView Array::vector_slice(int x1) const {
    return this->vector_slice(index1(x1));
  }
  ConstVectorView Array::vector_slice(int x1, int x2) const {
    return this->vector_slice(index2(x1, x2));
  }
  ConstVectorView Array::vector_slice(int x1, int x2, int x3) const {
    return this->vector_slice(index3(x1, x2, x3));
  }
  ConstVectorView Array::vector_slice(int x1, int x2, int x3, int x4) const {
    return this->vector_slice(index4(x1, x2, x3, x4));
  }
  ConstVectorView Array::vector_slice(int x1, int x2, int x3, int x4,
                                      int x5) const {
    return this->vector_slice(index5(x1, x2, x3, x4, x5));
  }
  ConstVectorView Array::vector_slice(int x1, int x2, int x3, int x4, int x5,
                                      int x6) const {
    return this->vector_slice(index6(x1, x2, x3, x4, x5, x6));
  }
  VectorView Array::vector_slice(const std::vector<int> &index) {
    ArrayView view(*this);
    return view.vector_slice(index);
  }
  VectorView Array::vector_slice(int x1) {
    return this->vector_slice(index1(x1));
  }
  VectorView Array::vector_slice(int x1, int x2) {
    return this->vector_slice(index2(x1, x2));
  }
  VectorView Array::vector_slice(int x1, int x2, int x3) {
    return this->vector_slice(index3(x1, x2, x3));
  }
  VectorView Array::vector_slice(int x1, int x2, int x3, int x4) {
    return this->vector_slice(index4(x1, x2, x3, x4));
  }
  VectorView Array::vector_slice(int x1, int x2, int x3, int x4, int x5) {
    return this->vector_slice(index5(x1, x2, x3, x4, x5));
  }
  VectorView Array::vector_slice(int x1, int x2, int x3, int x4, int x5,
                                 int x6) {
    return this->vector_slice(index6(x1, x2, x3, x4, x5, x6));
  }

  bool Array::operator==(const Array &rhs) const {
    return (dim() == rhs.dim()) && (data_ == rhs.data_);
  }

  ArrayIterator Array::abegin() {
    return ArrayIterator(this);
  }
  ArrayIterator Array::aend() {
    ArrayIterator it(this);
    it.set_to_end();
    return it;
  }

  ConstArrayIterator Array::abegin() const {
    return ConstArrayIterator(this);
  }
  ConstArrayIterator Array::aend() const {
    ConstArrayIterator it(this);
    it.set_to_end();
    return it;
  }

  ostream &Array::print(ostream &out) const {
    return print_array(out, *this);
  }
  std::string Array::to_string() const {
    return array_to_string(*this);
  }
  //======================================================================

}  // namespace BOOM
