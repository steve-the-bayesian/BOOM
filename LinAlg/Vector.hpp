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

#ifndef BOOM_LINALG_VECTOR_HPP
#define BOOM_LINALG_VECTOR_HPP

#include <cmath>
#include <functional>
#include <initializer_list>
#include <iosfwd>
#include <string>
#include <vector>

#include "cpputil/math_utils.hpp"
#include "distributions/rng.hpp"
#include "uint.hpp"

namespace BOOM {
  class SpdMatrix;
  class Matrix;
  class VectorView;
  class ConstVectorView;

  class Vector
      : public std::vector<double>
  {
   public:
    typedef std::vector<double> dVector;
    typedef dVector::iterator iterator;
    typedef dVector::const_iterator const_iterator;
    typedef dVector::reverse_iterator reverse_iterator;
    typedef dVector::const_reverse_iterator const_reverse_iterator;

    //--------- constructors, destructor, assigment, operator== ----------
    explicit Vector(uint n = 0, double x = 0);

    // Create a vector from a string that is comma or white space separated
    // cppcheck-suppress noExplicitConstructor
    explicit Vector(const std::string &s);

    // Create a vector from a string that is delimited with delimiter 'sep'
    Vector(const std::string &s, const std::string &sep);

    // cppcheck-suppress noExplicitConstructor
    Vector(const std::initializer_list<double> &init);  // NOLINT

    // A vector can be build using a stream of numbers, e.g. from a file.
    explicit Vector(std::istream &in);

    // Conversion from std::vector<double> is covered under the
    // template container structure, but the specialization for
    // dVector is likely to be faster.
    //
    // cppcheck-suppress noExplicitConstructor
    Vector(const dVector &d);  // NOLINT

    template <class FwdIt>
    Vector(FwdIt begin, FwdIt end);

    Vector(const Vector &rhs) = default;
    Vector(Vector &&rhs) = default;

    // cppcheck-suppress noExplicitConstructor
    Vector(const VectorView &);  // NOLINT
    // cppcheck-suppress noExplicitConstructor
    Vector(const ConstVectorView &);  // NOLINT

    // This constructors works with arbitrary STL containers.
    template <typename NUMERIC,
              template <typename ELEM, typename ALLOC = std::allocator<ELEM> >
              class CONTAINER>
    // cppcheck-suppress noExplicitConstructor
    Vector(const CONTAINER<NUMERIC> &rhs)  // NOLINT
        : dVector(rhs.begin(), rhs.end()) {}

    Vector &operator=(const Vector &rhs) = default;
    Vector &operator=(Vector &&rhs) = default;
    Vector &operator=(double);
    Vector &operator=(const VectorView &);
    Vector &operator=(const ConstVectorView &);
    Vector &operator=(const std::vector<double> &);
    Vector &operator=(const std::initializer_list<double> &);

    Vector &swap(Vector &);
    bool operator==(const Vector &rhs) const;

    // Returns true if empty, or if std::isfinite returns true on all
    // elements.  Returns false otherwise.
    bool all_finite() const;

    // Set all elements to zero.  This is slightly faster than using operator=
    // 0.0;
    void set_to_zero();

    Vector zero() const;  // returns a same sized Vector filled with 0's.
    Vector one() const;   // returns a same sized Vector filled with 1's.

    // Fill the Vector with U(0,1) random numbers.
    Vector &randomize(RNG &rng = GlobalRng::rng);

    // Fill the Vector with N(mu, sd^2) random numbers.
    Vector &randomize_gaussian(
        double mean = 0.0, double sd = 1.0, RNG &rng = GlobalRng::rng);

    //-------------- STL vector stuff ---------------------
    double *data();
    const double *data() const;
    uint stride() const { return 1; }
    uint length() const;  // same as size()

    //------------ resizing operations
    template <class VEC>
    Vector &concat(const VEC &v);
    Vector &push_back(double x);

    //------------------ checked subscripting -----------------------
    const double &operator()(uint n) const;
    double &operator()(uint n);

    //---------------- input/output -------------------------
    std::ostream &write(std::ostream &, bool endl = true) const;
    std::istream &read(std::istream &);

    //--------- math ----------------
    Vector &operator+=(double x);
    Vector &operator-=(double x);
    Vector &operator*=(double x);
    Vector &operator/=(double x);

    Vector &operator+=(const Vector &y);
    Vector &operator+=(const ConstVectorView &y);
    Vector &operator+=(const VectorView &y);

    Vector &operator-=(const Vector &y);
    Vector &operator-=(const ConstVectorView &y);
    Vector &operator-=(const VectorView &y);

    Vector &operator*=(const Vector &y);
    Vector &operator*=(const ConstVectorView &y);
    Vector &operator*=(const VectorView &y);

    Vector &operator/=(const Vector &y);
    Vector &operator/=(const ConstVectorView &y);
    Vector &operator/=(const VectorView &y);

    //--------- linear algebra
    Vector &axpy(const Vector &x, double w);           // *this += w*x
    Vector &axpy(const VectorView &x, double w);       // *this += w*x
    Vector &axpy(const ConstVectorView &x, double w);  // *this += w*x
    Vector &add_Xty(const Matrix &X, const Vector &y, double w = 1.0);
    // *this += w * X^T *y

    Vector &mult(const Matrix &A, Vector &ans) const;     // v^T A
    Vector mult(const Matrix &A) const;                   // v^T A
    Vector &mult(const SpdMatrix &A, Vector &ans) const;  // v^T A
    Vector mult(const SpdMatrix &A) const;                // v^T A

    double dot(const Vector &y) const;  // dot product ignores lower bounds
    double dot(const VectorView &y) const;
    double dot(const ConstVectorView &y) const;

    double affdot(const Vector &y) const;
    double affdot(const VectorView &y) const;
    double affdot(const ConstVectorView &y) const;
    // affine dot product:  dim(y) == dim(x)-1. ignores lower bounds

    SpdMatrix outer() const;                              // x x^t
    void outer(SpdMatrix &ans) const;                     // ans+=x x^t
    Matrix outer(const Vector &y, double a = 1.0) const;  // a*x y^t
    void outer(const Vector &y, Matrix &ans,
               double a = 1.0) const;  // ans += axy^t

    Vector &normalize_prob();     // *this/= sum(*this)
    Vector &normalize_logprob();  // *this = exp(*this)/sum(exp(*this))
    Vector &normalize_L2();       // *this /= *this.dot(*this);

    double normsq() const;    // *this.dot(*this);
    double abs_norm() const;  // sum of absolute values.. faster than sum
    double max_abs() const;   // returns -1 if empty.

    double min() const;
    double max() const;
    uint imax() const;  // index of maximal/minmal element
    uint imin() const;
    double sum() const;
    double prod() const;

    // Sort the elements of this vector (smallest to largest).  Return *this.
    Vector &sort();

    // apply fun to each element of *this, and return *this.
    Vector &transform(const std::function<double(double)> &fun);

    // Move the number in position 'from' to position 'to'.  Intermediate
    // elements are shifted to make room.
    void shift_element(int from, int to);

   private:
    bool inrange(uint n) const { return n < static_cast<uint>(size()); }
  };

  //----------------------------------------------------------------------

  template <class FwdIt>  // definition of template constructor
  Vector::Vector(FwdIt Beg, FwdIt End) : dVector(Beg, End) {}

  template <class VEC>
  Vector &Vector::concat(const VEC &v) {
    reserve(size() + v.size());
    dVector::insert(end(), v.begin(), v.end());
    return *this;
  }

  //======================= Vector functions ======================
  void permute_Vector(Vector &v, const std::vector<uint> &perm);
  Vector str2vec(const std::vector<std::string> &);
  Vector str2vec(const std::string &line);
  Vector scan_vector(const std::string &fname);

  // Field operators for Vector x Vector.
  Vector operator+(const Vector &a, const Vector &b);
  Vector operator-(const Vector &a, const Vector &b);
  Vector operator*(const Vector &a, const Vector &b);
  Vector operator/(const Vector &a, const Vector &b);

  // Field operators for Vector x double.
  Vector operator+(const Vector &x, double a);
  Vector operator+(double a, const Vector &x);
  Vector operator-(const Vector &x, double a);
  Vector operator-(double a, const Vector &x);
  Vector operator*(const Vector &x, double a);
  Vector operator*(double a, const Vector &x);
  Vector operator/(const Vector &x, double a);
  Vector operator/(double a, const Vector &x);

  // Field operators for Vector x VectorView
  Vector operator+(const Vector &x, const VectorView &y);
  Vector operator+(const VectorView &x, const Vector &y);
  Vector operator-(const Vector &x, const VectorView &y);
  Vector operator-(const VectorView &x, const Vector &y);
  Vector operator*(const Vector &x, const VectorView &y);
  Vector operator*(const VectorView &x, const Vector &y);
  Vector operator/(const Vector &x, const VectorView &y);
  Vector operator/(const VectorView &x, const Vector &y);

  // Field operators for Vector x ConstVectorView
  Vector operator+(const Vector &x, const ConstVectorView &y);
  Vector operator+(const ConstVectorView &x, const Vector &y);
  Vector operator-(const Vector &x, const ConstVectorView &y);
  Vector operator-(const ConstVectorView &x, const Vector &y);
  Vector operator*(const Vector &x, const ConstVectorView &y);
  Vector operator*(const ConstVectorView &x, const Vector &y);
  Vector operator/(const Vector &x, const ConstVectorView &y);
  Vector operator/(const ConstVectorView &x, const Vector &y);

  // Operators between VectorView and ConstVectorView are defined in
  // VectorView.hpp.

  // unary transformations
  Vector operator-(const Vector &x);  // unary minus

  using std::exp;
  using std::log;
  using std::pow;
  using std::sqrt;

  Vector log(const Vector &x);
  Vector log(const VectorView &x);
  Vector log(const ConstVectorView &x);

  Vector exp(const Vector &x);
  Vector exp(const VectorView &x);
  Vector exp(const ConstVectorView &x);

  Vector sqrt(const Vector &x);
  Vector sqrt(const VectorView &x);
  Vector sqrt(const ConstVectorView &x);

  Vector pow(const Vector &x, double p);
  Vector pow(const VectorView &x, double p);
  Vector pow(const ConstVectorView &x, double p);

  Vector pow(const Vector &x, int p);
  Vector pow(const VectorView &x, int p);
  Vector pow(const ConstVectorView &x, int p);

  Vector abs(const Vector &x);
  Vector abs(const VectorView &x);
  Vector abs(const ConstVectorView &x);

  inline int length(const Vector &x) { return x.length(); }
  inline double sum(const Vector &x) { return x.sum(); }
  inline double prod(const Vector &x) { return x.prod(); }
  inline double max(const Vector &x) { return x.max(); }
  inline double min(const Vector &x) { return x.min(); }

  std::pair<double, double> range(const Vector &x);
  std::pair<double, double> range(const VectorView &x);
  std::pair<double, double> range(const ConstVectorView &x);
  inline void swap(Vector &x, Vector &y) { x.swap(y); }
  Vector cumsum(const Vector &x);
  // IO
  std::ostream &operator<<(std::ostream &out, const Vector &x);
  // prints to stdout.  This function is here so it can be called from gdb.
  void print(const Vector &v);
  void print_vector(const Vector &v);

  // Print R code that can read in the vector values.
  std::string to_Rstring(const Vector &v);
  std::istream &operator>>(std::istream &, Vector &);
  Vector read_Vector(std::istream &in);

  // size changing operations

  Vector concat(const Vector &x, const Vector &y);
  Vector concat(const Vector &x, const VectorView &y);
  Vector concat(const Vector &x, const ConstVectorView &y);
  Vector concat(const Vector &x, double y);

  Vector concat(const VectorView &x, const Vector &y);
  Vector concat(const VectorView &x, const VectorView &y);
  Vector concat(const VectorView &x, const ConstVectorView &y);
  Vector concat(const VectorView &x, double y);

  Vector concat(const ConstVectorView &x, const Vector &y);
  Vector concat(const ConstVectorView &x, const VectorView &y);
  Vector concat(const ConstVectorView &x, const ConstVectorView &y);
  Vector concat(const ConstVectorView &x, double y);

  Vector concat(double x, const Vector &y);
  Vector concat(double x, const VectorView &y);
  Vector concat(double x, const ConstVectorView &y);

  template <class VEC>
  Vector concat(const std::vector<VEC> &v) {
    if (v.size() == 0) return Vector();
    Vector ans(v.front());
    for (uint i = 1; i < v.size(); ++i) ans.concat(v[i]);
    return ans;
  }

  Vector select(const Vector &v, const std::vector<bool> &inc, uint nvars);
  Vector select(const Vector &v, const std::vector<bool> &inc);
  Vector sort(const Vector &v);
  Vector sort(const VectorView &v);
  Vector sort(const ConstVectorView &v);
  Vector rev(const Vector &v);
  Vector rev(const VectorView &v);
  Vector rev(const ConstVectorView &v);

  // Args:
  //   v: A vector with elements sorted in ascending order (as returned by
  //     'sort').
  //   quantile: A number between 0 and 1 indicating which element of v is
  //     desired.
  //
  // Returns:
  //   The requested quantile of v. If v is empty negative_infinity is
  //   returned. If 'quantile' is between two entries in v then a linear
  //   interpolation of the entries is returned.
  double sorted_vector_quantile(const ConstVectorView &v, double quantile);

  // Round the elements to the nearest integer using lround.
  std::vector<int> round(const Vector &v);
  std::vector<int> round(const VectorView &v);
  std::vector<int> round(const ConstVectorView &v);

  template <class V1, class V2>
  Vector linear_combination(double a, const V1 &x, double b, const V2 &y) {
    Vector ans(x);
    ans *= a;
    ans.axpy(y, b);
    return ans;
  }

  // Returns the min (first) and max (second) elements in the vector-like object
  // v.  As implemented, v can be a Vector, VectorView, ConstVectorView, or any
  // STL container holding double's.
  template <class VECTOR>
  std::pair<double, double> min_max(const VECTOR &v) {
    double min_value = infinity();
    double max_value = negative_infinity();
    for (auto value : v) {
      if (value < min_value) min_value = value;
      if (value > max_value) max_value = value;
    }
    return std::pair<double, double>(min_value, max_value);
  }

  template <class VECTOR>
  bool all_finite(const VECTOR &v) {
    uint n = v.size();
    const double *d = v.data();
    uint stride = v.stride();
    for (uint i = 0; i < n; ++i) {
      if (!std::isfinite(*d)) {
        return false;
      }
      d += stride;
    }
    return true;
  }

}  // namespace BOOM

#endif  // BOOM_LINALG_VECTOR_HPP
