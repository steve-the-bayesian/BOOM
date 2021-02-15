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
  class TimeSeries : virtual public Data, public std::vector<Ptr<D>> {
   public:
    typedef D DataPointType;
    typedef TimeSeries<D> ts_type;

    TimeSeries();
    explicit TimeSeries(const D &);
    explicit TimeSeries(const std::vector<Ptr<D>> &v);

    TimeSeries(const TimeSeries &);                   // value semantics
    TimeSeries<D> *clone() const override;            // value semantics
    TimeSeries<D> &operator=(const TimeSeries &rhs);  // copies pointers

    // // Copy the pointers in the given sequence without setting links
    // template <class FwdIt>
    // TimeSeries<D> &ref(FwdIt Beg, FwdIt End);

    // If the data held in the TimeSeries inherit from MarkovLink then
    // reset their links so they point in sequence.
    //    void set_links();

    std::ostream &display(std::ostream &) const override;
    uint element_size(bool minimal = true) const;  // size of one data point
    uint length() const;                           // length of the series
    virtual uint size(bool minimal = true) const;

    // Adding data to the time series.
    virtual void add_data_point(const Ptr<D> &);

    // Concatenate series to the end of *this.
    virtual void add_series(const Ptr<TimeSeries<D>> &series);

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

  template <class D>
  TimeSeries<D>::TimeSeries() : Data(), std::vector<Ptr<D> >() {}

  template <class D>
  TimeSeries<D>::TimeSeries(const D &d) {}

  template <class D>
  TimeSeries<D>::TimeSeries(const std::vector<Ptr<D> > &v)
      : std::vector<Ptr<D> >(v)  // copies pointers
  {}

  template <class D>
  TimeSeries<D>::TimeSeries(const TimeSeries &rhs)
      : Data(rhs), std::vector<Ptr<D> >() {
    this->reserve(rhs.size());
    for (const auto &el : rhs) {
      this->push_back(el->clone());
    }
  }

  template <class D>
  TimeSeries<D> &TimeSeries<D>::operator=(const TimeSeries<D> &rhs) {
    if (&rhs != this) {
      this->clear();
      this->reserve(rhs.size());
      for (const auto &el : rhs) {
        this->push_back(el->clone());
      }
    }
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
  void TimeSeries<D>::add_data_point(const Ptr<D> &d) {
    this->push_back(d);
  }

  template <class D>
  void TimeSeries<D>::add_series(const Ptr<TimeSeries<D> > &d) {
    for (uint i = 0; i < d->length(); ++i) {
      add_data_point((*d)[i]);
    }
  }

}  // namespace BOOM

#endif  // BOOM_TIME_SERIES_BASE_CLASS_HPP
