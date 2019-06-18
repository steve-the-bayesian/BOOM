// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2014 Steven L. Scott

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

#ifndef BOOM_TIME_SERIES_BASE_CLASS_HPP
#define BOOM_TIME_SERIES_BASE_CLASS_HPP

#include <type_traits>
#include <sstream>
#include <vector>
#include "Models/DataTypes.hpp"
#include "Models/TimeSeries/MarkovLink.hpp"

namespace BOOM {
  //======================================================================
  // Return true if d inherits from MarkovLink and its links are set.
  // Otherwise return false.
  template <class D>
  bool linked(const Ptr<D> &d) {
    // true if either prev or next is set
    bool linkable = std::is_base_of<MarkovLink<D>, D>::value;
    if (!linkable) return false;
    return (!!d->next() || !!d->prev());
  }

  //======================================================================

  // A TimeSeries is a type of Data that holds vector of pointers to
  // a specific Data type.
  template <class D>
  class TimeSeries : virtual public Data, public std::vector<Ptr<D> > {
   public:
    typedef D data_point_type;
    typedef TimeSeries<D> ts_type;

    TimeSeries();
    explicit TimeSeries(const D &);
    explicit TimeSeries(const std::vector<Ptr<D> > &v, bool reset_links = true);

    TimeSeries(const TimeSeries &);                   // value semantics
    TimeSeries<D> *clone() const override;            // value semantics
    TimeSeries<D> &operator=(const TimeSeries &rhs);  // copies pointers

    TimeSeries<D> &unique_copy(const TimeSeries &rhs);  // clones pointers

    // Copy the pointers in the given sequence without setting links
    template <class FwdIt>
    TimeSeries<D> &ref(FwdIt Beg, FwdIt End);

    // If the data held in the TimeSeries inherit from MarkovLink then
    // reset their links so they point in sequence.
    void set_links();

    std::ostream &display(std::ostream &) const override;
    uint element_size(bool minimal = true) const;  // size of one data point
    uint length() const;                           // length of the series
    virtual uint size(bool minimal = true) const;

    // Adding data to the time series... add_1 and add_series
    // assimilate their entries so that the time series is remains
    // contiguous, with each element pointing to the next and previous
    // elements.
    virtual void add_1(const Ptr<D> &);
    virtual void add_series(const Ptr<TimeSeries<D> > &);

    // Simply add a pointer to the end of the time series, without
    // bothering to check its links.  The point may or may not refer
    // to other elements already in the series.
    void just_add(const Ptr<D> &);

    void clear();

   private:
    // Makes *this a copy of rhs.  Copies underlying data and sets links
    void clone_series(const TimeSeries &rhs);
  };

  typedef TimeSeries<DoubleData> ScalarTimeSeries;
  inline Ptr<ScalarTimeSeries> make_ts(const Vector &y) {
    uint n = y.size();
    std::vector<Ptr<DoubleData> > ts;
    ts.reserve(n);
    for (uint i = 0; i < n; ++i) {
      NEW(DoubleData, yi)(y[i]);
      ts.push_back(yi);
    }
    NEW(ScalarTimeSeries, ans)(ts);
    return ans;
  }

  //======================================================================
  // The structs defined below are used to implment the TimeSeries
  // templates using template meta-programming.  The main problem
  // being solved is whether or not the data elements in the time
  // series have links back and forth.  If they do then the TimeSeries
  // will set those links as data elements are added.

  template <class D, class T>
  struct time_series_data_adder {
    void operator()(const Ptr<D> &, Ptr<D>) {}
  };

  template <class D>
  struct time_series_data_adder<D, std::true_type> {
    void operator()(const Ptr<D> &last, Ptr<D> d) {
      if (linked(d)) {
        // If the links for d are already set, then do nothing.
        return;
      }
      if (!last->next()) last->set_next(d);
      if (!d->prev()) d->set_prev(last);
    }
  };

  template <class D>
  struct is_linkable : public std::is_base_of<MarkovLink<D>, D> {};

  template <class D, class T = is_linkable<D> >
  struct time_series_link_clearer {
    void operator()(const Ptr<D> &) {}
  };

  template <class D>
  struct time_series_link_clearer<D, std::true_type> {
    void operator()(const Ptr<D> &d) { d->clear_links(); }
  };

  template <class D, class T>
  struct set_links_impl {
    void operator()(std::vector<Ptr<D> > &) {}
  };

  template <class D>
  struct set_links_impl<D, std::true_type> {
    void operator()(std::vector<Ptr<D> > &v) {
      uint n = v.size();
      if (n == 0) return;
      for (uint i = 0; i < n; ++i) {
        if (i > 0) v[i]->set_prev(v[i - 1]);
        if (i < n - 1) v[i]->set_next(v[i + 1]);
      }
      v.front()->unset_prev();
      v.back()->unset_next();
    }
  };

  //======================================================================

  template <class D>
  TimeSeries<D>::TimeSeries() : Data(), std::vector<Ptr<D> >() {}

  template <class D>
  TimeSeries<D>::TimeSeries(const D &d) {}

  template <class D>
  void TimeSeries<D>::clone_series(const TimeSeries<D> &rhs) {
    uint n = rhs.length();
    std::vector<Ptr<D> >::resize(n);
    for (uint i = 0; i < n; ++i) (*this)[i] = rhs[i]->clone();
    set_links();
  }

  template <class D>
  void TimeSeries<D>::set_links() {
    typedef typename is_linkable<D>::type isLinked;
    set_links_impl<D, isLinked> impl;
    impl(*this);
  }

  template <class D>
  TimeSeries<D>::TimeSeries(const std::vector<Ptr<D> > &v, bool reset_links)
      : std::vector<Ptr<D> >(v)  // copies pointers
  {
    if (reset_links) set_links();
  }

  template <class D>
  TimeSeries<D>::TimeSeries(const TimeSeries &rhs)
      : Data(), std::vector<Ptr<D> >() {
    clone_series(rhs);
  }

  template <class D>
  TimeSeries<D> &TimeSeries<D>::operator=(const TimeSeries<D> &rhs) {
    if (&rhs == this) return *this;
    // changed 10/21/2005.  No longer clones underlying data
    //    clone_series(rhs);
    std::vector<Ptr<D> >::operator=(rhs);  // now just the pointers are copied
    return *this;
  }

  template <class D>
  TimeSeries<D> &TimeSeries<D>::unique_copy(const TimeSeries<D> &rhs) {
    if (&rhs == this) return *this;
    clone_series(rhs);
    return *this;
  }

  template <class D>
  template <class FwdIt>
  TimeSeries<D> &TimeSeries<D>::ref(FwdIt Beg, FwdIt End) {
    std::vector<Ptr<D> >::assign(Beg, End);
    return *this;
  }

  template <class D>
  TimeSeries<D> *TimeSeries<D>::clone() const {
    return new TimeSeries<D>(*this);
  }

  template <class D>
  std::ostream &TimeSeries<D>::display(std::ostream &out) const {
    for (uint i = 0; i < length(); ++i) {
      (*this)[i]->display(out);
      out << std::endl;
    }
    return out;
  }

  template <class D>
  uint TimeSeries<D>::element_size(bool minimal) const {
    if (this->empty()) return 0;
    return std::vector<Ptr<D> >::back()->size(minimal);
  }

  template <class D>
  uint TimeSeries<D>::size(bool) const {
    return std::vector<Ptr<D> >::size();
  }

  template <class D>
  uint TimeSeries<D>::length() const {
    return std::vector<Ptr<D> >::size();
  }

  template <class D>
  void TimeSeries<D>::add_1(const Ptr<D> &d) {
    if (length() > 0) {
      Ptr<D> last = std::vector<Ptr<D> >::back();
      time_series_data_adder<D, is_linkable<D> > adder;
      adder(last, d);
    }
    just_add(d);
  }

  template <class D>
  void TimeSeries<D>::add_series(const Ptr<TimeSeries<D> > &d) {
    for (uint i = 0; i < d->length(); ++i) add_1((*d)[i]);
  }

  template <class D>
  void TimeSeries<D>::just_add(const Ptr<D> &d) {
    std::vector<Ptr<D> >::push_back(d);
  }

  template <class D>
  void TimeSeries<D>::clear() {
    std::vector<Ptr<D> >::clear();
  }
}  // namespace BOOM

#endif  // BOOM_TIME_SERIES_BASE_CLASS_HPP
