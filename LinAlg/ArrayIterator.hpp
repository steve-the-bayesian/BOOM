// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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
#ifndef BOOM_ARRAY_ITERATOR_HPP_
#define BOOM_ARRAY_ITERATOR_HPP_

#include <iterator>
#include <vector>

namespace BOOM {

  class ArrayBase;
  class ConstArrayBase;

  class ArrayPositionManager {
   public:
    explicit ArrayPositionManager(const std::vector<int> &dims);
    void operator++();

    // Move the position back to the beginning.
    void reset();

    // Move the position to one past the end.
    void set_to_end();

    // Check whether position is one past the end.
    bool at_end() const { return at_end_; }

    bool operator==(const ArrayPositionManager &rhs) const;
    bool operator!=(const ArrayPositionManager &rhs) const;

    const std::vector<int> &position() const { return position_; }
    void set_position(const std::vector<int> &position);

   private:
    const std::vector<int> &dims_;
    std::vector<int> position_;
    bool at_end_;
  };

  //======================================================================
  class ArrayIterator {
   public:
    ArrayIterator(ArrayBase *host, const std::vector<int> &starting_position);
    explicit ArrayIterator(ArrayBase *host);

    // Standard iterator operations.
    double &operator*();

    bool operator==(const ArrayIterator &rhs) const {
      return (host_ == rhs.host_) && (position_ == rhs.position_);
    }

    bool operator!=(const ArrayIterator &rhs) const { return !(*this == rhs); }

    ArrayIterator &operator++() {
      ++position_;
      return *this;
    }

    // Array-specific stuff.

    // The vector-valued set of indices being pointed to.
    const std::vector<int> &position() const { return position_.position(); }

    // The (scalar valued) array offset in the block of memory being pointed to.
    size_t scalar_position() const;

    // Point the iterator at a given position given by array indices [i, j, k].
    void set_position(const std::vector<int> &position) {
      position_.set_position(position);
    }

    // Set the iterator to "one-past-the-end" of the memory block.
    ArrayIterator &set_to_end() {
      position_.set_to_end();
      return *this;
    }

   private:
    ArrayBase *host_;
    ArrayPositionManager position_;
  };

  //======================================================================
  class ConstArrayIterator {
   public:
    // Iterator begins at the beginning of the array.
    ConstArrayIterator(const ConstArrayBase *host);

    // Begins at a specific position.
    ConstArrayIterator(const ConstArrayBase *host,
                       const std::vector<int> &starting_position);

    // Standard iterator operations.
    double operator*() const;

    bool operator==(const ConstArrayIterator &rhs) const {
      return (host_ == rhs.host_) && (position_ == rhs.position_);
    }

    bool operator!=(const ConstArrayIterator &rhs) const {
      return !(*this == rhs);
    }

    ConstArrayIterator &operator++() {
      ++position_;
      return *this;
    }

    // Array-specific stuff.

    // The vector-valued set of indices being pointed to.
    const std::vector<int> &position() const { return position_.position(); }

    // The (scalar valued) array offset in the block of memory being pointed to.
    size_t scalar_position() const;

    // Point the iterator at a given position given by array indices [i, j, k].
    void set_position(const std::vector<int> &position) {
      position_.set_position(position);
    }

    // Set the iterator to "one-past-the-end" of the memory block.
    ConstArrayIterator &set_to_end() {
      position_.set_to_end();
      return *this;
    }

   private:
    const ConstArrayBase *host_;
    ArrayPositionManager position_;
  };

}  // namespace BOOM


// To make the iterator a 'first class' iterator we need to specialize
// std::iterator_traits.  This should be done inside namespace std.
namespace std {
  template <> struct iterator_traits<BOOM::ArrayIterator> {
    using iterator_category = forward_iterator_tag;
    //  using difference_type   = ptrdiff_t;
    using difference_type   = void;
    using value_type        = double;
    using pointer           = double*;
    using reference         = double&;
  };

  template<> struct iterator_traits<BOOM::ConstArrayIterator> {
    using iterator_category = forward_iterator_tag;
    //  using difference_type   = ptrdiff_t;
    using difference_type   = void;
    using value_type        = double;
    using pointer           = double*;
    using reference         = double&;
  };
}  // namespace std


#endif  //  BOOM_ARRAY_ITERATOR_HPP_
