#ifndef BOOM_CPPUTIL_SORTED_VECTOR_HPP_
#define BOOM_CPPUTIL_SORTED_VECTOR_HPP_

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

// Matt Austern (of Google, and the C++ standards committe) argues against using
// std::set for most purposes.  This code is a restyled version of his
// implementation of a sorted vector that (other than requiring linear time for
// insert operations) is a drop-in replacement for std::set.  His arguments are
// given in https://lafstern.org/matt/col1.pdf

// TODO: As of right now this is not a fully fledged STL container class, and it
//       is missing some elements of both the set and vector classes.  The plan
//       is to add those as they become relevant in actual working code that
//       uses this class.

#include <vector>
#include <algorithm>
#include <initializer_list>

namespace BOOM {
  template <class T, class COMPARE = std::less<T>>
  class SortedVector {
   public:
    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;

    // Default constructor.
    SortedVector(const COMPARE& c = COMPARE())
        : elements_(), cmp(c) {}

    // Range constructor.  Duplicate elements in the range will be discarded
    // after sorting.
    template <class InputIterator>
    SortedVector(InputIterator first, InputIterator last,
                  const COMPARE& c = COMPARE())
        : elements_(first, last),
          cmp(c)
    {
      std::sort(begin(), end(), cmp);
      auto el = std::unique(elements_.begin(), elements_.end());
      elements_.erase(el, elements_.end());
    }

    explicit SortedVector(const std::vector<T> &things)
        : SortedVector(things.begin(), things.end())
    {}

    explicit SortedVector(const std::initializer_list<T> &things)
        : SortedVector(std::vector<T>(things))
    {}

    iterator       begin()       { return elements_.begin(); }
    iterator       end()         { return elements_.end(); }
    const_iterator begin() const { return elements_.begin(); }
    const_iterator end()   const { return elements_.end(); }

    iterator insert(const T& t) {
      iterator i = std::lower_bound(begin(), end(), t, cmp);
      if (i == end() || cmp(t, *i))
        elements_.insert(i, t);
      return i;
    }

    const_iterator find(const T& t) const {
      const_iterator i = std::lower_bound(begin(), end(), t, cmp);
      return i == end() || cmp(t, *i) ? end() : i;
    }

    bool contains(const T &t) const {
      return find(t) != end();
    }

    size_t size() const {
      return elements_.size();
    }

    bool operator==(const SortedVector &rhs) const {
      return elements_ == rhs.elements_;
    }

    T & operator[](size_t i) {
      return elements_[i];
    }

    const T & operator[](size_t i) const {
      return elements_[i];
    }

    // Set operations
    SortedVector<T, COMPARE> intersection(const SortedVector &rhs) const {
      SortedVector<T, COMPARE> ans;
      for (const auto &el : elements_) {
        if (rhs.contains(el)) {
          ans.insert(el);
        }
      }
      return ans;
    }

    // This is called set_union instead of union because union is a C++ reserved
    // word (a union of two types).
    SortedVector<T, COMPARE> set_union(const SortedVector &rhs) const {
      SortedVector<T, COMPARE> ans;
      std::merge(
          elements_.begin(),
          elements_.end(),
          rhs.elements_.begin(),
          rhs.elements_.end(),
          std::back_inserter(ans.elements_));
      auto it = std::unique(ans.elements_.begin(),
                            ans.elements_.end());
      ans.elements_.erase(it, ans.elements_.end());
      return ans;
    }

   private:
    std::vector<T> elements_;
    COMPARE cmp;
  };

}  // namespace BOOM

#endif  //  BOOM_CPPUTIL_SORTED_VECTOR_HPP_
