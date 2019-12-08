// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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

#ifndef BOOM_SELECTOR_HPP
#define BOOM_SELECTOR_HPP

#include "uint.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "LinAlg/DiagonalMatrix.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "LinAlg/Vector.hpp"

#include "distributions/rng.hpp"

namespace BOOM {

  // TODO:  remove the inheritance from vector<bool>
  //
  // A Selector models inclusion or exclusion from a set of positions.  The job
  // of a Selector is often to extract a subset of elements from a Vector or
  // Matrix, resulting in a smaller Vector or Matrix.
  class Selector : public std::vector<bool> {
   public:
    Selector();
    explicit Selector(uint p, bool all = true);  // all true or all false

    // Using this constructor, Selector s("10") would have s[0] = true
    // and s[1] = false.
    explicit Selector(const char *zeros_and_ones);
    explicit Selector(const std::string &zeros_and_ones);
    explicit Selector(const std::vector<bool> &values);
    Selector(const std::vector<uint> &pos, uint n);

    bool operator==(const Selector &rhs) const;
    bool operator!=(const Selector &rhs) const;
    void swap(Selector &rhs);

    // Append one or more elements elements to the end.  Return *this.
    Selector &append(const Selector &new_trailing_elements);

    void push_back(bool element);
    void erase(uint which_element);

    // Convert to a vector of 0's and 1's.
    Vector to_Vector() const;

    //---------- Element counts ----------

    // The number of included items.
    uint nvars() const;

    // The number of excluded items.
    uint nvars_excluded() const;

    // The number of items (included + excluded).
    uint nvars_possible() const;

    //---------- Adding and dropping variables ----------

    // Add element i to the included set.  If it is already present, then do
    // nothing.
    Selector &add(uint i);
    Selector &operator+=(uint i) { return add(i); }

    // Remove element i from the included set.  If it is already absent then do
    // nothing.
    Selector &drop(uint i);
    Selector &operator-=(uint i) { return drop(i); }

    // Add element i if it is absent, otherwise drop it.
    Selector &flip(uint i);

    // Add or drop all elements.
    void drop_all();
    void add_all();

    //-------------- Set theory:  And, Or, Not, ... ------------------
    // Returns a selector of the same size as this, but with all elements
    // flipped.
    Selector complement() const;

    // Returns an indicator of whether element i is included.
    bool inc(uint i) const;

    // Returns true iff every included element in rhs is included in *this.
    bool covers(const Selector &rhs) const;

    // Returns the set union: locations which are in either Selector.  Note that
    // lower-case union is a reserved c++ keyword.
    Selector Union(const Selector &rhs) const;

    // Set union.  Add any elements that are present in rhs, while keeping any
    // that are already present in *this.  Return *this.
    Selector &cover(const Selector &rhs);
    Selector &operator+=(const Selector &rhs) {return this->cover(rhs);}

    // Returns the set intersection, locations which are in both Selectors.
    Selector intersection(const Selector &rhs) const;

    // Intersection.  Drop any elements that are absent in rhs.
    Selector &operator*=(const Selector &rhs);

    // Returns a Selector that is 1 in places where this disagrees with rhs.
    Selector exclusive_or(const Selector &rhs) const;

    // --------- Selecting subsets ----------

    // Returns the position of the ith nonzero element in the expanded sparse
    // vector.
    uint indx(uint i) const;  // i=0..n-1, ans in 0..N-1
    uint sparse_index(uint dense_index) const { return indx(dense_index); }

    // Returns the position in the condensed (dense) vector corresponding to
    // position I in the expanded (sparse) vector.
    uint INDX(uint I) const;  // I=0..N-1, ans in 0..n-1
    uint dense_index(uint sparse_index) const { return INDX(sparse_index); }

    // Returns the index of a randomly selected included (or excluded)
    // element.  If no (all) elements are included then -1 is returned
    // as an error code.
    int random_included_position(RNG &rng) const;
    int random_excluded_position(RNG &rng) const;

    // Return the index of the first included value at or before 'position'.  If
    // no elements in this position or lower are included, then return -1.
    int first_included_at_or_before(uint position) const;

    Vector select(const Vector &x) const;          // x includes intercept
    Vector select(const VectorView &x) const;
    Vector select(const ConstVectorView &x) const;

    SpdMatrix select(const SpdMatrix &) const;
    Matrix select_cols(const Matrix &M) const;
    Matrix select_square(const Matrix &M) const;  // selects rows and columns
    Matrix select_rows(const Matrix &M) const;
    Matrix select_rows(const SubMatrix &M) const;
    Matrix select_rows(const ConstSubMatrix &M) const;

    DiagonalMatrix select_square(const DiagonalMatrix &diag) const;
    DiagonalMatrix select(const DiagonalMatrix &diag) const {
      return select_square(diag);
    }

    template <class T>
    std::vector<T> select(const std::vector<T> &stuff) const;

    SpdMatrix expand(const SpdMatrix &dense_part_of_sparse_matrix);
    Vector expand(const Vector &x) const;
    Vector expand(const VectorView &x) const;
    Vector expand(const ConstVectorView &x) const;

    // Fill the missing elements of a vector with specfic values.
    // Args:
    //   v: The vector to be partially filled.
    //   value/values:  The values to be filled in for v[i] where *this[i] is false.
    //     In the vector version, values must have size nvars_excluded().
    // Returns:
    //   The excluded elements of v are filled with the supplied values, and the
    //   modified v is returned.
    Vector &zero_missing_elements(Vector &v) const {
      return fill_missing_elements(v, 0.0); }
    Vector &fill_missing_elements(Vector &v, double value) const;
    Vector &fill_missing_elements(Vector &v,
                                  const ConstVectorView &values) const;

    template <class T>
    T sub_select(const T &x, const Selector &rhs) const;
    // x is an object obtained by select(original_object).
    // this->covers(rhs).  sub_select(x,rhs) returns the object that
    // would have been obtained by rhs.select(original_object)

    //---------- Sparse linear algebra ----------

    // Fill ans with select_cols(M) * select(v).
    void sparse_multiply(const Matrix &M, const Vector &v,
                         VectorView ans) const;
    Vector sparse_multiply(const Matrix &M, const Vector &v) const;
    Vector sparse_multiply(const Matrix &M, const VectorView &v) const;
    Vector sparse_multiply(const Matrix &M, const ConstVectorView &v) const;

    double sparse_dot_product(const Vector &full_size_vector,
                              const Vector &sparse_vector) const;
    double sparse_dot_product(const Vector &full_size_vector,
                              const VectorView &sparse_vector) const;
    double sparse_dot_product(const Vector &full_size_vector,
                              const ConstVectorView &sparse_vector) const;
    double sparse_dot_product(const VectorView &full_size_vector,
                              const Vector &sparse_vector) const;
    double sparse_dot_product(const VectorView &full_size_vector,
                              const VectorView &sparse_vector) const;
    double sparse_dot_product(const VectorView &full_size_vector,
                              const ConstVectorView &sparse_vector) const;
    double sparse_dot_product(const ConstVectorView &full_size_vector,
                              const Vector &sparse_vector) const;
    double sparse_dot_product(const ConstVectorView &full_size_vector,
                              const VectorView &sparse_vector) const;
    double sparse_dot_product(const ConstVectorView &full_size_vector,
                              const ConstVectorView &sparse_vector) const;

    double sparse_sum(const ConstVectorView &view) const;
    double sparse_sum(const VectorView &view) const;
    double sparse_sum(const Vector &vector) const;

    const std::vector<uint> &included_positions() const {
      return included_positions_;
    }

   private:
    // sorted vector of included indices
    std::vector<uint> included_positions_;

    // Set to true if all elements are included.
    bool include_all_;

    // Recompute included_positions_ from scratch.
    void reset_included_positions();

    // Checks that the size of this is equal to 'p'.
    void check_size_eq(uint p, const std::string &fun_name) const;

    // Checks that the size of *this is greater than 'p'.
    void check_size_gt(uint p, const std::string &fun_name) const;
  };
  //______________________________________________________________________

  std::ostream &operator<<(std::ostream &, const Selector &);
  std::istream &operator>>(std::istream &, Selector &);

  template <class T>
  std::vector<T> select(const std::vector<T> &v, const std::vector<bool> &vb) {
    uint n = v.size();
    assert(vb.size() == n);
    std::vector<T> ans;
    for (uint i = 0; i < n; ++i)
      if (vb[i]) ans.push_back(v[i]);
    return ans;
  }

  template <class T>
  std::vector<T> Selector::select(const std::vector<T> &v) const {
    assert(v.size() == nvars_possible());
    if (include_all_ || nvars() == nvars_possible()) return v;
    std::vector<T> ans;
    ans.reserve(nvars());
    for (uint i = 0; i < nvars(); ++i) {
      ans.push_back(v[indx(i)]);
    }
    return ans;
  }

  template <class T>
  T Selector::sub_select(const T &x, const Selector &rhs) const {
    assert(rhs.nvars() <= this->nvars());
    assert(this->covers(rhs));

    Selector tmp(nvars(), false);
    for (uint i = 0; i < rhs.nvars(); ++i) {
      tmp.add(INDX(rhs.indx(i)));
    }
    return tmp.select(x);
  }

  //===========================================================================
  // A selector object indicating which elements in a matrix of coefficients are
  // nonzero.
  class SelectorMatrix {
   public:
    SelectorMatrix(int nrow, int ncol, bool include_all = true) {
      for (int i = 0; i < ncol; ++i) {
        columns_.push_back(Selector(nrow, include_all));
      }
    }

    // Args:
    //   nrow: Number of rows.
    //   ncol: Number of columns.
    //   selector: Elements.  Size must be nrow * ncol.  The matrix is filled in
    //     column major order (i.e. column-by-column).
    SelectorMatrix(int nrow, int ncol, const Selector &selector) {
      int counter = 0;
      for (int j = 0; j < ncol; ++j) {
        columns_.emplace_back(nrow, false);
        for (int i = 0; i < nrow; ++i) {
          columns_[j][i] = selector[counter++];
        }
      }
    }

    // The total number of included positions in the matrix.
    int nvars() const {
      int ans = 0;
      for (const auto &col : columns_) ans += col.nvars();
      return ans;
    }

    int nrow() const {
      if (columns_.empty()) return 0;
      return columns_[0].size();
    }

    int ncol() const {return columns_.size();}
    bool operator()(int i, int j) const {return columns_[j][i];}

    // Returns true iff all coefficients are included.
    bool all_in() const;

    // Returns true iff all coefficients are excluded.
    bool all_out() const;

    void add_all() {for (auto &el : columns_) el.add_all();}
    void drop_all() {for (auto &el : columns_) el.drop_all();}
    void flip(int i, int j) {columns_[j].flip(i);}
    void add(int i, int j) {columns_[j].add(i);}
    void drop(int i, int j) {columns_[j].drop(i);}

    const Selector &col(int i) const {return columns_[i];}
    Selector row(int i) const;

    // Indicate whether each row is included by at least one column.
    Selector row_any() const;

    // Indicate whether each row is included in all columns.
    Selector row_all() const;

    // Returns the selector obtained by stacking the columns of the selector
    // matrix.
    Selector vectorize() const;

    // Return the vector obtained by selecting the included elements of mat, and
    // then stacking them column-wise.
    Vector vector_select(const Matrix &mat) const;

    // Expand the subset of selected values back to the matrix from which it was
    // selected.
    Matrix expand(const Vector &subset) const;

    // Flip all bits with probability .5.
    void randomize();

   private:
    // The selector elements map to the selector matrix elements in column-major
    // order.
    std::vector<Selector> columns_;
  };



}  // namespace BOOM
#endif  // BOOM_SELECTOR_HPP
