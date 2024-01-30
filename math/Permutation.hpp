#ifndef BOOM_MATH_PERMUTATION_HPP_
#define BOOM_MATH_PERMUTATION_HPP_
/*
  Copyright (C) 2005-2024 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include <vector>
#include <ostream>
#include "uint.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "cpputil/index_table.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  // A Permutation is an ordering of the integer 0, ... , N-1.  A permutation p
  // changes the order of a vector v through (p * v)[i] = v[p[i]].
  //
  // You can use a permutation by applying it to a vector of objects, by
  // referring to individual elements, or by accessing the underlying vector of
  // positions.  A permutation can be inverted, and finding the inverse of a
  // permutation is fairly inexpensive.
  template <class INT=Int>
  class Permutation {
   public:
    explicit Permutation(const std::vector<INT> &elements):
        elements_(elements)
    {}

    // Return a permutation 'perm' of 'things' such that perm * things is sorted
    // in ascending order.  
    template <class T>
    static Permutation order(const std::vector<T> &things) {
      return Permutation<INT>(index_table<T, INT>(things));
    }

    size_t size() const {
      return elements_.size();
    }

    const std::vector<INT> &elements() const {
      return elements_;
    }

    // Return an individual element of the permutation.
    INT operator[](size_t i) const {
      return elements_[i];
    }

    // Return the composition of two permutations: *this * rhs.
    Permutation<INT> apply(const Permutation<INT> &rhs) const {
      if (this->size() != rhs.size()) {
        report_error("Permutations of different sizes cannot be composed.");
      }
      return Permutation<INT>(*this * rhs.elements_);
    }
    
    // Permute the input.
    template <class T>
    std::vector<T> apply(const std::vector<T> &input) const {
      std::vector<T> ans(input.size());
      INT size = elements_.size();
      for (INT i = 0; i < size; ++i) {
        ans[i] = input[elements_[i]];
      }
      return ans;
    }

    // Permute the input.
    Vector apply(const Vector &v) const {
      return Vector(apply(std::vector<double>(v)));
    }

    // Return a permutation ans such ans * (*this) * x == x.
    Permutation inverse() const {
      return Permutation(index_table<INT, INT>(elements_));
    }

    // Permute the input in place.  Return a reference to the input, after
    // permuting.
    Vector &apply_inplace(Vector &input) const {
      permute_inplace(elements_, input);
      return input;
    }
    
    // Permute the input in place.  Return a reference to the input, after
    // permuting.
    VectorView &apply_inplace(VectorView &input) const {
      permute_inplace(elements_, input);
      return input;
    }

    // Print the permutation's elements.
    std::ostream &print(std::ostream &out) const {
      out << "[";
      for (size_t i = 0; i < elements_.size(); ++i) {
        out << elements_[i];
        if (i + 1 < elements_.size()) {
          out << ", ";
        }
      }
      out << "]";
      return out;
    }
    
   private:
    std::vector<INT> elements_;
  };


  template <class T, class INT=Int>
  std::vector<T> operator*(const Permutation<INT> &permutation,
                           const std::vector<T> &v) {
    return permutation.apply(v);
  }

  template <class INT>
  Permutation<Int> operator*(const Permutation<INT> &p1,
                             const Permutation<INT> &p2) {
    return p1.apply(p2);
  }
  
  template <class INT=Int>
  std::ostream & operator<<(std::ostream &out,
                            const Permutation<INT> &perm) {
    return perm.print(out);
  }
  
}  // namespace BOOM


#endif  //  BOOM_MATH_PERMUTATION_HPP_
