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

#include "LinAlg/ArrayIterator.hpp"
#include "LinAlg/Array.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  ArrayPositionManager::ArrayPositionManager(const std::vector<int> &dims)
      : dims_(dims), position_(dims.size(), 0), at_end_(false) {
    if (dims_.empty()) {
      at_end_ = true;
    }
  }

  void ArrayPositionManager::operator++() {
    if (at_end_) {
      // If you're at the end then don't do anything.
      return;
    }
    // Iterate through the indices of the array until you find one
    // you can advance.
    for (int which_index = 0; which_index < dims_.size(); ++which_index) {
      ++position_[which_index];
      if (position_[which_index] < dims_[which_index]) {
        // This is the normal case.  Increment the iterator in its
        // current position.
        return;
      } else {
        // Here position_[which_index] has reached the end of its
        // range.  Set it to zero and continue in the for-loop.  We
        // will increment the next position in the next part of the
        // loop.
        position_[which_index] = 0;
      }
    }
    // At this point we've made it all the way through the for loop,
    // which means we were unable to increment any of the position
    // indices.  That means we've reached the end of the data.  Signal
    // that we've reached the end of the data and then return.
    set_to_end();
  }

  void ArrayPositionManager::reset() {
    position_.assign(dims_.size(), 0);
    at_end_ = false;
    if (dims_.empty()) {
      at_end_ = true;
    }
  }

  void ArrayPositionManager::set_to_end() {
    at_end_ = true;
    position_.assign(position_.size(), -1);
  }

  // A conscious decision was made to not check dims_.  Presumably
  // operator== will be called many times as one loops through an
  // array, and checking dims each time just seems wasteful.
  //
  // This class is intended to be an implementation detail for use in
  // ArrayIterator and ConstArrayIterator.  Those will have their own
  // operator==, which can check that the host array is the same,
  // which implies dims being the same here.
  bool ArrayPositionManager::operator==(const ArrayPositionManager &rhs) const {
    return (at_end_ == rhs.at_end_) && (position_ == rhs.position_);
  }

  bool ArrayPositionManager::operator!=(const ArrayPositionManager &rhs) const {
    return !(*this == rhs);
  }

  void ArrayPositionManager::set_position(const std::vector<int> &position) {
    if (position.size() != dims_.size()) {
      ostringstream err;
      err << "The 'position' argument passed to set_position had the wrong "
          << "number of dimensions.  Host array has " << dims_.size()
          << " dimensions, but argument has " << position.size() << ".";
      report_error(err.str());
    }
    for (int i = 0; i < dims_.size(); ++i) {
      if (position[i] < 0 || position[i] >= dims_[i]) {
        std::ostringstream err;
        err << "Dimension " << i << " of 'position' argument is out of bounds."
            << std::endl
            << "  Argument value: " << position[i] << std::endl
            << "  Legal value are between 0 and " << dims_[i] - 1 << ".";
        report_error(err.str());
      }
    }
    position_ = position;
    at_end_ = false;
  }

  //======================================================================

  ArrayIterator::ArrayIterator(ArrayBase *host,
                               const std::vector<int> &starting_position)
      : host_(host), position_(host->dim()) {
    position_.set_position(starting_position);
  }

  ArrayIterator::ArrayIterator(ArrayBase *host)
      : host_(host), position_(host->dim()) {}

  double &ArrayIterator::operator*() {
    if (position_.at_end()) {
      report_error("ArrayIterator dereference past end of data.");
    }
    return (*host_)[position_.position()];
  }

  size_t ArrayIterator::scalar_position() const {
    return host_->array_index(position_.position(), host_->dim(), host_->strides());
  }

  //======================================================================

  ConstArrayIterator::ConstArrayIterator(
      const ConstArrayBase *host, const std::vector<int> &starting_position)
      : host_(host), position_(host->dim()) {
    position_.set_position(starting_position);
  }

  ConstArrayIterator::ConstArrayIterator(const ConstArrayBase *host)
      : host_(host), position_(host->dim()) {}

  double ConstArrayIterator::operator*() const {
    if (position_.at_end()) {
      report_error("ConstArrayIterator dereference past end of data.");
    }
    return (*host_)[position()];
  }

  size_t ConstArrayIterator::scalar_position() const {
    return host_->array_index(position_.position(), host_->dim(), host_->strides());
  }

}  // namespace BOOM
