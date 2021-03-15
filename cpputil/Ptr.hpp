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

#ifndef BOOM_SMART_PTR_H
#define BOOM_SMART_PTR_H

#include <memory>
#include <cassert>

#define NEW(T, y) Ptr<T> y = new T
// NEW(very_long_type_name, variable_name)(constructor, arguments)
// is shorthand for
//
// Ptr<very_long_type_name> variable_name = new
// very_long_type_name(constructor, arguments)

namespace BOOM {

  // Ptr is modeled off of boost::intrusive_ptr.  Until late 2019 it was
  // implemented in terms of intrusive_ptr, but the dependence was eliminated so
  // that BOOM could be independent of boost.
  template <class T>
  class Ptr {  // intrusive pointers
   public:
    typedef T element_type;

    Ptr() : managed_pointer_(nullptr) {}

    // NOLINTNEXTLINE  Implicit conversions are intentional.
    Ptr(T *p, bool add_ref = true) : managed_pointer_(p) {
      if (p  && add_ref) {
        intrusive_ptr_add_ref(managed_pointer_);
      }
    }

    Ptr(const Ptr &rhs)
        : managed_pointer_(rhs.managed_pointer_)
    {
      bump_up();
    }

    // Move the managed pointer to a new home.  Do not adjust the reference
    // count.  Leave rhs in an empty state.
    Ptr(Ptr &&rhs)
        : managed_pointer_(rhs.managed_pointer_) {
      rhs.managed_pointer_ = nullptr;
    }

    // NOLINTNEXTLINE  Implicit conversions are intentional.
    template <class Y> Ptr(const Ptr<Y> &rhs)
        : managed_pointer_(rhs.get()) {
      bump_up();
    }

    ~Ptr() {
      bump_down();
    }

    Ptr &operator=(const Ptr &rhs) {
      if (&rhs != this) {
        set(rhs.managed_pointer_);
      }
      return *this;
    }

    template <class Y>
    Ptr &operator=(const Ptr<Y> &rhs) {
      set(rhs.get());
      return *this;
    }

    template <class Y>
    Ptr &operator=(T *r) {
      set(r);
      return *this;
    }

    template <class U>
    bool operator<(const Ptr<U> &rhs) const {
      return managed_pointer_ < rhs.get();
    }

    T &operator*() const {
      assert(managed_pointer_ != nullptr);
      return *managed_pointer_;
    }
    T *operator->() const {
      assert(managed_pointer_ != nullptr);
      return managed_pointer_;
    }

    T *get() const { return managed_pointer_; }

    template <class U>
    Ptr<U> dcast() const {
      return Ptr<U>(dynamic_cast<U *>(managed_pointer_));
    }

    template <class U>
    Ptr<U> scast() const {
      return Ptr<U>(static_cast<U *>(managed_pointer_));
    }

    template <class U>
    Ptr<U> rcast() const {
      return Ptr<U>(reinterpret_cast<U *>(managed_pointer_));
    }

    template <class U>
    Ptr<U> ccast() const {
      return Ptr<U>(const_cast<U *>(managed_pointer_));
    }

    explicit operator bool () const noexcept {
      return managed_pointer_;
    }  // never throws

    bool operator!() const { return !managed_pointer_; }

    template <class OTHER>
    bool operator==(const Ptr<OTHER> &rhs) const {
      return managed_pointer_ == rhs.get();
    }

    template <class OTHER>
    bool operator!=(const Ptr<OTHER> &rhs) const {
      return managed_pointer_ != rhs.get();
    }

    void swap(Ptr &b) {
      // Reference counts follow managed_pointer_ and b.managed_pointer_, so no
      // adjustments are needed.
      std::swap(managed_pointer_, b.managed_pointer_);
    }

    void reset(T *new_value = nullptr) {
      set(new_value);
    }

   private:
    T * managed_pointer_;

    // Increase the reference count of the managed pointer by 1.
    void bump_up() {
      if (managed_pointer_) {
        intrusive_ptr_add_ref(managed_pointer_);
      }
    }

    // Decrease the reference count of the managed pointer by 1.
    void bump_down() {
      if (managed_pointer_) {
        intrusive_ptr_release(managed_pointer_);
      }
    }

    void set(const T *rhs) const {
      bump_down();
      managed_pointer_ = rhs;
      bump_up();
    }

    void set(T* rhs) {
      bump_down();
      managed_pointer_ = rhs;
      bump_up();
    }
  };

}  // namespace BOOM

#endif  // BOOM_SMART_PTR_H
