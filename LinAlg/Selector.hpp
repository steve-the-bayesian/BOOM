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

#include <BOOM.hpp>

#include <vector>
#include <iostream>
#include <cassert>
#include <string>

#include <LinAlg/Vector.hpp>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/SpdMatrix.hpp>

#include <distributions/rng.hpp>

namespace BOOM{

  //TODO(stevescott):  remove the inheritance from vector<bool>
  //
  // A Selector models inclusion or exclusion from a set of positions.
  // The job of a Selector is often to extract a subset of elements
  // from a Vector or Matrix, resulting in a smaller Vector or Matrix.
  class Selector : public std::vector<bool> {
   public:
    Selector();
    explicit Selector(uint p, bool all=true);  // all true or all false

    // Using this constructor, Selector s("10") would have s[0] = true
    // and [1] = false;
    explicit Selector(const std::string &zeros_and_ones);
    Selector(const std::vector<bool> &);
    Selector(const std::vector<uint> &pos, uint n);

    bool operator==(const Selector &rhs) const;
    bool operator!=(const Selector &rhs) const;
    void swap(Selector &rhs);

    // Append one or more elements elements to the end.  Return *this.
    Selector & append(bool new_last_element);
    Selector & append(const Selector &new_trailing_elements);

    // Add element i to the included set.  If it is already present,
    // then do nothing.
    Selector & add(uint i);
    Selector & operator+=(uint i) {return add(i);}

    // Remove element i from the included set.  If it is already
    // absent then do nothing.
    Selector & drop(uint i);
    Selector & operator-=(uint i) {return drop(i);}

    // Add element i if it is absent, otherwise drop it.
    Selector & flip(uint i);

    // Set union.  Add any elements that are present in rhs, while
    // keeping any that are already present in *this.  Return *this.
    Selector & operator+=(const Selector &rhs);

    // Intersection.  Drop any elements that are absent in rhs.
    Selector & operator*=(const Selector &rhs);

    // Returns a selector of the same size as this, but with all
    // elements flipped.
    Selector complement() const;

    // Add or drop all elements.
    void drop_all();
    void add_all();

    uint nvars() const;          // =="n"
    uint nvars_possible() const; // =="N"
    uint nvars_excluded() const; // == N-n;

    // Returns an indicator of whether element i is included.
    bool inc(uint i) const;

    // Returns true iff every included element in rhs is included in
    // *this.
    bool covers(const Selector &rhs) const;

    // Returns the set union: locations which are in either Selector.
    // Note that lower-case union is a reserved c++ keyword.
    Selector Union(const Selector &rhs) const;

    // Returns the set intersection, locations which are in both Selectors.
    Selector intersection(const Selector &rhs) const;

    // Returns a Selector that is 1 in places where this disagrees with rhs.
    Selector exclusive_or(const Selector &rhs) const;
    Selector & cover(const Selector &rhs);  // makes *this cover rhs

    // Returns the position of the ith nonzero element in the expanded
    // sparse vector.
    uint indx(uint i) const;  // i=0..n-1, ans in 0..N-1

    // Returns the position in the condensed (dense) vector
    // corresponding to position I in the expanded (sparse) vector.
    uint INDX(uint I) const;  // I=0..N-1, ans in 0..n-1

    // Convert the Selector to an explicit vector of 0's and 1's.
    Vector to_Vector() const;

    Vector select(const Vector &x) const;          // x includes intercept
    Vector select_add_int(const Vector &x) const;  // intercept is implicit
    SpdMatrix select(const SpdMatrix &) const;
    Matrix select_cols(const Matrix &M) const;
    Matrix select_cols_add_int(const Matrix &M) const;
    Matrix select_square(const Matrix &M) const;  // selects rows and columns
    Matrix select_rows(const Matrix &M) const;

    Vector select(const VectorView & x) const;
    Vector select(const ConstVectorView & x) const;
    SpdMatrix expand(const SpdMatrix &dense_part_of_sparse_matrix);

    Vector expand(const Vector &x) const;
    Vector expand(const VectorView &x) const;
    Vector expand(const ConstVectorView &x) const;

    Vector & zero_missing_elements(Vector &v) const;

    // Fill ans with select_cols(M) * select(v).
    void sparse_multiply(const Matrix &M,
                         const Vector &v,
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

    template <class T>
    std::vector<T> select(const std::vector<T> &v) const;

    template<class T>
    T sub_select(const T &x, const Selector &rhs) const;
    // x is an object obtained by select(original_object).
    // this->covers(rhs).  sub_select(x,rhs) returns the object that
    // would have been obtained by rhs.select(original_object)

    // Returns the index of a randomly selected included (or excluded)
    // element.  If no (all) elements are included then -1 is returned
    // as an error code.
    int random_included_position(RNG &rng) const;
    int random_excluded_position(RNG &rng) const;

    void push_back(bool element);
    void erase(uint which_element);

   private:
    // sorted vector of included indices
    std::vector<uint> included_positions_;

    // Set to true if all elements are included.
    bool include_all_;

    // Recompute included_positions_ from scratch.
    void reset_included_positions();

    // Checks that the size of this is equal to 'p'.
    void check_size_eq(uint p, const string &fun_name) const;
    // Checks that the size of *this is greater than 'p'.
    void check_size_gt(uint p, const string &fun_name) const;

  };
  //______________________________________________________________________

  ostream & operator<<(ostream &, const Selector &);
  istream & operator>>(istream &, Selector &);

  Selector find_contiguous_subset(const Vector &big, const Vector &small);
  // find_contiguous_subset returns the indices of 'big' that contain
  // the vector 'small.'  If 'small' is not found then an empty
  // include is returned.

  template <class T>
  std::vector<T> select(const std::vector<T> &v, const std::vector<bool> &vb) {
    uint n = v.size();
    assert(vb.size()==n);
    std::vector<T> ans;
    for(uint i=0; i<n; ++i) if(vb[i]) ans.push_back(v[i]);
    return ans;
  }

  template <class T>
  std::vector<T> Selector::select(const std::vector<T> &v) const{
    assert(v.size()==nvars_possible());
    std::vector<T> ans;
    ans.reserve(nvars());
    for(uint i=0; i<nvars_possible(); ++i)
      if(inc(i))
        ans.push_back(v[i]);
    return ans;}

  template <class T>
  T Selector::sub_select(const T &x, const Selector &rhs) const{
    assert(rhs.nvars() <= this->nvars());
    assert(this->covers(rhs));

    Selector tmp(nvars(), false);
    for(uint i=0; i<rhs.nvars(); ++i) {
      tmp.add(INDX(rhs.indx(i)));
    }
    return tmp.select(x);
  }

}  // namespace BOOM
#endif // BOOM_SELECTOR_HPP
