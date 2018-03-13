// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
  02110-1301, USA
*/

#ifndef BOOM_REF_COUNTED_HPP
#define BOOM_REF_COUNTED_HPP

#include <atomic>

namespace BOOM {

  class RefCounted {
   public:
    RefCounted() : reference_count_(0) {}
    RefCounted(const RefCounted &) : reference_count_(0) {}
    virtual ~RefCounted() {}

    // If this object is assigned a new value, nothing is done to the
    // reference count, so assignment is a no-op.
    RefCounted &operator=(const RefCounted &rhs) { return *this; }

    void up_count() { ++reference_count_; }
    void down_count() { --reference_count_; }
    unsigned int ref_count() const { return reference_count_; }

   private:
    std::atomic<unsigned int> reference_count_;
  };

}  // namespace BOOM
#endif  // BOOM_REF_COUNTED_HPP
