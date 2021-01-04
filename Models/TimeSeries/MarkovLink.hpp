// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_TS_MARKOV_LINK_HPP
#define BOOM_TS_MARKOV_LINK_HPP

#include "Models/DataTypes.hpp"

namespace BOOM {

  template <class D>
  class MarkovLink {
    Ptr<D> prev_;
    Ptr<D> next_;

   public:
    MarkovLink();
    explicit MarkovLink(const Ptr<D> &Prev);
    MarkovLink(const MarkovLink &rhs);
    virtual ~MarkovLink() { clear_links(); }  // problems?????
    D *prev() const { return prev_.get(); }
    D *next() const { return next_.get(); }
    MarkovLink<D> &operator=(const MarkovLink &rhs);
    void unset_prev() { prev_ = Ptr<D>(); }
    void unset_next() { next_ = Ptr<D>(); }
    void set_prev(const Ptr<D> &p) { prev_ = p; }
    void set_next(const Ptr<D> &n) { next_ = n; }
    void clear_links() {
      unset_prev();
      unset_next();
    }
  };

  template <class D>
  MarkovLink<D>::MarkovLink()
      : prev_(nullptr),
        next_(nullptr)
  {}

  template <class D>
  MarkovLink<D>::MarkovLink(const Ptr<D> &last) :
      prev_(last),
      next_(nullptr)
  {}

  template <class D>
  MarkovLink<D>::MarkovLink(const MarkovLink &rhs)
      : prev_(rhs.prev_), next_(rhs.next_) {}

  template <class D>
  MarkovLink<D> &MarkovLink<D>::operator=(const MarkovLink &rhs) {
    if (&rhs != this) {
      prev_ = rhs.prev_;
      next_ = rhs.next_;
    }
    return *this;
  }

}  // namespace BOOM
#endif  // BOOM_TS_MARKOV_LINK_HPP
