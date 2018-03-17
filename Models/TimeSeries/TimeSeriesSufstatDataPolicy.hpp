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

#ifndef BOOM_TIME_SERIES_SUFSTAT_DATA_POLICY_HPP
#define BOOM_TIME_SERIES_SUFSTAT_DATA_POLICY_HPP

#include "Models/Sufstat.hpp"
#include "Models/TimeSeries/TimeSeriesDataPolicy.hpp"

namespace BOOM {

  template <class D, class TS, class SUF>
  class TimeSeriesSufstatDataPolicy : public TimeSeriesDataPolicy<D, TS> {
   public:
    typedef TimeSeriesDataPolicy<D, TS> Base;
    typedef D DataPointType;
    typedef TS DataSeriesType;  // should inherit from TimeSeries<D>
    typedef TimeSeriesSufstatDataPolicy<D, TS, SUF> DataPolicy;

    explicit TimeSeriesSufstatDataPolicy(const Ptr<SUF> &s);
    TimeSeriesSufstatDataPolicy(const Ptr<SUF> &s, const DataSeriesType &ds);
    TimeSeriesSufstatDataPolicy(const TimeSeriesSufstatDataPolicy &rhs);
    TimeSeriesSufstatDataPolicy *clone() const = 0;
    TimeSeriesSufstatDataPolicy &operator=(const TimeSeriesSufstatDataPolicy &);

    using Base::DAT;
    using Base::DAT_1;

    template <class FwdIt>
    void set_data(FwdIt Beg, FwdIt End);
    virtual void set_data(const Ptr<DataSeriesType> &d);

    virtual void add_data_series(const Ptr<DataSeriesType> &d);
    virtual void add_data_point(const Ptr<DataPointType> &d);
    virtual void add_data(const Ptr<Data> &d);
    virtual void clear_data();

    using Base::dat;

    const Ptr<SUF> suf() const { return suf_; }
    void clear_suf() { suf_->clear(); }
    void update_suf(const Ptr<DataPointType> &d) { suf_->update(d); }
    void update_suf(const Ptr<DataSeriesType> &d);
    void refresh_suf();

   private:
    Ptr<SUF> suf_;
  };
  //=====================================================================

  template <class D, class TS, class S>
  void TimeSeriesSufstatDataPolicy<D, TS, S>::update_suf(
      const Ptr<DataSeriesType> &pds) {
    const DataSeriesType &d(*pds);
    for (uint i = 0; i < d.size(); ++i) suf_->update(d[i]);
  }

  template <class D, class TS, class S>
  TimeSeriesSufstatDataPolicy<D, TS, S>::TimeSeriesSufstatDataPolicy(
      const TimeSeriesSufstatDataPolicy &rhs)
      : Model(rhs),
        TimeSeriesDataInfoPolicy<D, TS>(rhs),
        Base(rhs),
        suf_(rhs.suf_->clone())

  {}

  template <class D, class TS, class S>
  TimeSeriesSufstatDataPolicy<D, TS, S>::TimeSeriesSufstatDataPolicy(
      const Ptr<S> &s)
      : Base(), suf_(s) {}

  template <class D, class TS, class S>
  TimeSeriesSufstatDataPolicy<D, TS, S>::TimeSeriesSufstatDataPolicy(
      const Ptr<S> &s, const DataSeriesType &ds)
      : Base(ds), suf_(s) {}

  template <class D, class TS, class S>
  TimeSeriesSufstatDataPolicy<D, TS, S>
      &TimeSeriesSufstatDataPolicy<D, TS, S>::operator=(const DataPolicy &rhs) {
    if (&rhs == this) return *this;
    Base::operator=(rhs);
    suf_ = rhs.suf_->clone();
    return *this;
  }

  template <class D, class TS, class S>
  template <class FwdIt>
  void TimeSeriesSufstatDataPolicy<D, TS, S>::set_data(FwdIt Beg, FwdIt End) {
    Base::set_data(Beg, End);
    refresh_suf();
  }

  template <class D, class TS, class S>
  void TimeSeriesSufstatDataPolicy<D, TS, S>::set_data(
      const Ptr<DataSeriesType> &d) {
    Base::set_data(d);
    refresh_suf();
  }

  template <class D, class TS, class S>
  void TimeSeriesSufstatDataPolicy<D, TS, S>::add_data_series(
      const Ptr<DataSeriesType> &d) {
    Base::add_data_series(d);
    update_suf(d);
  }

  template <class D, class TS, class S>
  void TimeSeriesSufstatDataPolicy<D, TS, S>::add_data_point(
      const Ptr<DataPointType> &d) {
    Base::add_data_point(d);
    this->suf()->update(d);
  }

  template <class D, class TS, class S>
  void TimeSeriesSufstatDataPolicy<D, TS, S>::add_data(const Ptr<Data> &d) {
    Base::add_data(d);
    this->suf()->update(d);
  }

  template <class D, class TS, class S>
  void TimeSeriesSufstatDataPolicy<D, TS, S>::clear_data() {
    Base::clear_data();
    this->suf()->clear();
  }

  template <class D, class TS, class S>
  void TimeSeriesSufstatDataPolicy<D, TS, S>::refresh_suf() {
    suf()->clear();
    uint n = this->nseries();
    for (uint i = 0; i < n; ++i) {
      const DataSeriesType &d(this->dat(i));
      for (uint j = 0; j < d.size(); ++j) suf()->update(d[j]);
    }
  }

  //======================================================================

}  // namespace BOOM
#endif  // BOOM_TIME_SERIES_SUFSTAT_DATA_POLICY_HPP
