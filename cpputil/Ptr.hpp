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

#include <boost/intrusive_ptr.hpp>
#include <memory>

#define NEW(T, y) Ptr<T> y = new T
// NEW(very_long_type_name, variable_name)(constructor, arguments)
// is shorthand for
//
// Ptr<very_long_type_name> variable_name = new
// very_long_type_name(constructor, arguments)

namespace BOOM {

  template <class T, bool INTRUSIVE = true>
  class Ptr;

  //======================================================================
  template <class T>
  class Ptr<T, true> {  // intrusive pointers
   public:
    typedef T element_type;
    typedef Ptr<T, false> this_type;
    typedef T *this_type::*unspecified_bool_type;

    const boost::intrusive_ptr<T> &get_boost() const {
      return managed_pointer_;
    }
    boost::intrusive_ptr<T> &get_boost() { return managed_pointer_; }

    Ptr() = default;

    // NOLINTNEXTLINE  Implicit conversions are intentional.
    Ptr(T *p, bool add_ref = true) : managed_pointer_(p, add_ref) {}
    Ptr(const Ptr &rhs) = default;

    // CRAN complains about an ASAN test failure, which may be a false alarm.
    // If it is not, then the failure has something to do with this move
    // constructor.  See https://github.com/steve-the-bayesian/BOOM/issues/35
    // 
    Ptr(Ptr &&rhs) = default;
    // If the default move constructor is not used, then it seems the following
    // constructor should be used instead.  Not sure how this would be different
    // than the default.
    //    
    // Ptr(Ptr &&rhs)
    //     : managed_pointer_(std::move(rhs.managed_pointer_))
    // {}

    // NOLINTNEXTLINE  Implicit conversions are intentional.
    template <class Y> Ptr(const Ptr<Y, true> &rhs)
        : managed_pointer_(rhs.get_boost()) {}

    ~Ptr() {}  // deletes pt

    Ptr &operator=(const Ptr &rhs) {
      if (&rhs != this) {
        managed_pointer_ = rhs.managed_pointer_;
      }
      return *this;
    }

    template <class Y>
    Ptr &operator=(const Ptr<Y, true> &rhs) {
      managed_pointer_ = rhs.get_boost();
      return *this;
    }

    template <class Y>
    Ptr &operator=(T *r) {
      managed_pointer_ = r;
      return *this;
    }

    template <class U>
    bool operator<(const Ptr<U, true> &rhs) const {
      return get() < rhs.get();
    }

    T &operator*() const { return *managed_pointer_; }
    T *operator->() const { return managed_pointer_.operator->(); }
    T *get() const { return managed_pointer_.get(); }

    template <class U>
    Ptr<U, true> dcast() const {
      return Ptr<U, true>(dynamic_cast<U *>(managed_pointer_.get()));
    }

    template <class U>
    Ptr<U, true> scast() const {
      return Ptr<U, true>(static_cast<U *>(managed_pointer_.get()));
    }

    template <class U>
    Ptr<U, true> rcast() const {
      return Ptr<U, true>(reinterpret_cast<U *>(managed_pointer_.get()));
    }

    template <class U>
    Ptr<U, true> ccast() const {
      return Ptr<U, true>(const_cast<U *>(managed_pointer_.get()));
    }

    operator unspecified_bool_type() const {
      return !managed_pointer_ ? 0 : &this_type::get_boost();
    }  // never throws

    bool operator!() const { return !managed_pointer_; }

    template <class OTHER>
    bool operator==(const Ptr<OTHER> &rhs) const {
      return managed_pointer_.get() == rhs.get();
    }

    template <class OTHER>
    bool operator!=(const Ptr<OTHER> &rhs) const {
      return managed_pointer_.get() != rhs.get();
    }

    void swap(Ptr &b) { managed_pointer_.swap(b.get_boost()); }
    void reset() { managed_pointer_.reset(); }
    void reset(T *new_value) { managed_pointer_.reset(new_value); }

   private:
    boost::intrusive_ptr<T> managed_pointer_;
  };

  //======================================================================
  template <class T>
  class Ptr<T, false> {
   public:
    const std::shared_ptr<T> &get_boost() const { return managed_pointer_; }
    std::shared_ptr<T> &get_boost() { return managed_pointer_; }

    typedef T element_type;
    typedef Ptr<T, false> this_type;
    typedef T *this_type::*unspecified_bool_type;

    Ptr() : managed_pointer_() {}

    // NOLINTNEXTLINE  Implicit conversions are intentional.
    template <class Y> Ptr(Y *p)
        : managed_pointer_(p) {}

    // NOLINTNEXTLINE  Implicit conversions are intentional.
    template <class Y, class D> Ptr(Y *p, D d)
        : managed_pointer_(p, d) {}
    
    ~Ptr() {}

    Ptr(const Ptr &rhs) : managed_pointer_(rhs.managed_pointer_) {}

    // NOLINTNEXTLINE  Implicit conversions are intentional.
    template <class Y> Ptr(const Ptr<Y> &rhs)
        : managed_pointer_(rhs.get_boost()) {}

    // NOLINTNEXTLINE  Implicit conversions are intentional.
    template <class Y> Ptr(const std::shared_ptr<T> &rhs)
        : managed_pointer_(rhs) {}

    Ptr &operator=(const Ptr &rhs) {
      if (&rhs == this) return *this;
      managed_pointer_ = rhs.managed_pointer_;
      return *this;
    }

    template <class Y>
    Ptr &operator=(const Ptr<Y> &rhs) {
      managed_pointer_ = rhs.get_boost();
      return *this;
    }

    template <class Y>
    Ptr &operator=(Y *rhs) {  // normal boost pointers prohibit this
      if (managed_pointer_.get() == rhs) return *this;
      if (!rhs)
        managed_pointer_.reset();
      else
        Ptr(rhs).swap(*this);
      return *this;
    }

    Ptr &operator=(const std::shared_ptr<T> &rhs) {
      managed_pointer_ = rhs;
      return *this;
    }

    inline void reset() { managed_pointer_.reset(); }
    template <class Y>
    inline void reset(Y *p) {
      managed_pointer_.reset(p);
    }
    template <class Y, class D>
    inline void reset(Y *p, D d) {
      managed_pointer_.reset(p, d);
    }

    bool operator!() const { return !managed_pointer_; }

    T &operator*() const { return *managed_pointer_; }
    T *operator->() const { return managed_pointer_.operator->(); }
    T *get() const { return managed_pointer_.get(); }

    bool operator<(const Ptr<T, false> &rhs) const { return get() < rhs.get(); }

    bool unique() const { return managed_pointer_.unique(); }
    long use_count() const { return managed_pointer_.use_count(); }

    void swap(Ptr &b) { managed_pointer_.swap(b.get_boost()); }

    template <class U>
    Ptr<U, false> dcast() const {
      return Ptr<U, false>(boost::dynamic_pointer_cast<U>(managed_pointer_));
    }

    template <class U>
    Ptr<U, false> scast() const {
      return Ptr<U, false>(boost::static_pointer_cast<U>(managed_pointer_));
    }

    template <class U>
    Ptr<U, false> ccast() const {
      return Ptr<U, false>(boost::const_pointer_cast<U>(managed_pointer_));
    }

    template <class U>
    friend bool operator<(const Ptr &, const Ptr<U, false> &);

    template <class A, class B>
    friend std::basic_ostream<A, B> operator<<(std::basic_ostream<A, B> &os,
                                               const Ptr &);

    template <class D>
    friend D *get_deleter(const Ptr<T, false> &);

   private:
    std::shared_ptr<T> managed_pointer_;
  };

  //-------------------------------------------------------

  template <class T, class U>
  inline bool operator==(const Ptr<T, false> &a, const Ptr<U, false> &b) {
    return a.get() == b.get();
  }

  template <class T, class U>
  inline bool operator!=(const Ptr<T, false> &a, const Ptr<U, false> &b) {
    return a.get() != b.get();
  }

  template <class T, class U>
  inline bool operator<(const Ptr<T, false> &a, const Ptr<T, false> &b) {
    return a.managed_pointer_ < b.managed_pointer_;
  }

  //-------------------------------------------------------
  template <class T>
  inline void swap(Ptr<T, false> &a, Ptr<T, false> &b) {
    a.swap(b);
  }

  template <class T>
  inline T *get_pointer(Ptr<T, false> &a) {
    return a.get();
  }

  //-------------------------------------------------------
  template <class A, class B, class T>
  inline std::basic_ostream<A, B> &operator<<(std::basic_ostream<A, B> &os,
                                              const Ptr<T, false> &a) {
    os << a.managed_pointer_;
    return os;
  }

  template <class T, class D>
  inline D *get_deleter(const Ptr<T, false> &a) {
    return a.managed_pointer_.get_deleter();
  }

  template <class To, class From>
  Ptr<To, false> dcast(const Ptr<From, false> &a) {
    std::shared_ptr<To> tmp = boost::dynamic_pointer_cast<To>(a.get_boost());
    return Ptr<To, false>(tmp);
  }

  template <class To, class From>
  Ptr<To, true> dcast(const Ptr<From, true> &a) {
    return Ptr<To>(dynamic_cast<To *>(a.get()));
  }

}  // namespace BOOM

#endif  // BOOM_SMART_PTR_H
