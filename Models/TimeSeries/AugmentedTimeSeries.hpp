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

#ifndef BOOM_AUGMENTED_TIME_SERIES_BASE_CLASS_HPP
#define BOOM_AUGMENTED_TIME_SERIES_BASE_CLASS_HPP

#include "Models/TimeSeries/TimeSeries.hpp"

namespace BOOM {
  template <class D, class FIRST>
  class AugmentedTimeSeries : public TimeSeries<D> {
    /*
      An augmented time series is a TimeSeries<D> augmented with an
      initial extra data point of type FIRST, presumed to be at time
      zero.  The augmenting data point may contain very different
      information than the elements of the time series.
     */

   public:
    typedef D data_point_type;
    typedef TimeSeries<D> ts_type;

    explicit AugmentedTimeSeries(const Ptr<FIRST> &x0, const string &ID = "");
    AugmentedTimeSeries(const Ptr<FIRST> &x0, Ptr<D> proto,
                        const string &ID = "");
    AugmentedTimeSeries(const Ptr<FIRST> &x0, const std::vector<Ptr<D> > &v,
                        bool reset_links = true, const string &ID = "");
    template <class FwdIt>
    AugmentedTimeSeries(const Ptr<FIRST> &, FwdIt Beg, FwdIt End,
                        bool reset_links = true, bool copy_data = false,
                        const string &ID = "");

    AugmentedTimeSeries(const AugmentedTimeSeries &);  // value semantics
    AugmentedTimeSeries<D, FIRST> *clone() const;      // value semantics
    AugmentedTimeSeries<D, FIRST> &operator=(
        const AugmentedTimeSeries &rhs);  // copies pointers
    AugmentedTimeSeries<D, FIRST> &unique_copy(
        const AugmentedTimeSeries &rhs);  // clones pointers

    virtual uint size(bool minimal) const;
    virtual std::ostream &display(std::ostream &out) const;
    //    virtual istream & read(istream & in);

    void set_x0(const Ptr<FIRST> &);
    Ptr<FIRST> x0();
    const Ptr<FIRST> x0() const;

   private:
    Ptr<FIRST> x0_;  // initial augmenting data point
  };
  //======================================================================

  template <class D, class F>
  AugmentedTimeSeries<D, F>::AugmentedTimeSeries(const Ptr<F> &first,
                                                 const string &ID)
      : TimeSeries<D>(ID), x0_(first) {}

  template <class D, class F>
  AugmentedTimeSeries<D, F>::AugmentedTimeSeries(const Ptr<F> &first, Ptr<D> d,
                                                 const string &ID)
      : TimeSeries<D>(d, ID), x0_(first) {}

  template <class D, class F>
  AugmentedTimeSeries<D, F>::AugmentedTimeSeries(const Ptr<F> &x0,
                                                 const std::vector<Ptr<D> > &v,
                                                 bool reset_links,
                                                 const string &ID)
      : TimeSeries<D>(v, reset_links, ID), x0_(x0) {}

  template <class D, class F>
  template <class FwdIt>
  AugmentedTimeSeries<D, F>::AugmentedTimeSeries(const Ptr<F> &first, FwdIt Beg,
                                                 FwdIt End, bool reset_links,
                                                 bool copy_data,
                                                 const string &ID)
      : TimeSeries<D>(Beg, End, reset_links, copy_data, ID), x0_(first) {}

  template <class D, class F>
  AugmentedTimeSeries<D, F>::AugmentedTimeSeries(const AugmentedTimeSeries &rhs)
      : Data(rhs), TimeSeries<D>(rhs), x0_(rhs.x0_->clone()) {}

  template <class D, class F>
  AugmentedTimeSeries<D, F> &AugmentedTimeSeries<D, F>::operator=(
      const AugmentedTimeSeries<D, F> &rhs) {
    if (&rhs == this) return *this;
    // changed 10/21/2005.  No longer clones underlying data
    //    clone_series(rhs);
    TimeSeries<D>::operator=(rhs);
    x0_ = rhs.x0_;
    return *this;
  }

  template <class D, class F>
  AugmentedTimeSeries<D, F> *AugmentedTimeSeries<D, F>::clone() const {
    return new AugmentedTimeSeries<D, F>(*this);
  }

  template <class D, class F>
  AugmentedTimeSeries<D, F> &AugmentedTimeSeries<D, F>::unique_copy(
      const AugmentedTimeSeries<D, F> &rhs) {
    TimeSeries<D>::unique_copy(rhs);
    x0_ = rhs.x0_->clone();
    return *this;
  }

  template <class D, class F>
  uint AugmentedTimeSeries<D, F>::size(bool minimal) const {
    return x0_->size(minimal) + TimeSeries<D>::size(minimal);
  }

  template <class D, class F>
  std::ostream &AugmentedTimeSeries<D, F>::display(std::ostream &out) const {
    x0_->display(out);
    TimeSeries<D>::display(out);
    return out;
  }

  //   template <class D, class F>
  //   istream & AugmentedTimeSeries<D,F>::read(istream &in){
  //     x0_->read(in);
  //     TimeSeries<D>::read(in);
  //     return in;
  //   }

  template <class D, class F>
  void AugmentedTimeSeries<D, F>::set_x0(const Ptr<F> &x) {
    x0_ = x;
  }

  template <class D, class F>
  Ptr<F> AugmentedTimeSeries<D, F>::x0() {
    return x0_;
  }

  template <class D, class F>
  const Ptr<F> AugmentedTimeSeries<D, F>::x0() const {
    return x0_;
  }

}  // namespace BOOM

#endif  // BOOM_AUGMENTED_TIME_SERIES_BASE_CLASS_HPP
