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

  template <class DATA, class SERIES, class SUF>
  class TimeSeriesSufstatDataPolicy : public TimeSeriesDataPolicy<DATA, SERIES> {
   public:
    typedef TimeSeriesDataPolicy<DATA, SERIES> Base;
    typedef DATA DataPointType;
    typedef SERIES DataSeriesType;  // should inherit from TimeSeries<D>
    typedef TimeSeriesSufstatDataPolicy<DATA, SERIES, SUF> DataPolicy;

    explicit TimeSeriesSufstatDataPolicy(const Ptr<SUF> &s)
        : Base(), suf_(s) {}

    TimeSeriesSufstatDataPolicy(const Ptr<SUF> &s, const DataSeriesType &ds)
        : Base(ds), suf_(s) {}

    TimeSeriesSufstatDataPolicy(const TimeSeriesSufstatDataPolicy &rhs)
        : Model(rhs), Base(rhs), suf_(rhs.suf_->clone()) {}

    TimeSeriesSufstatDataPolicy *clone() const = 0;

    TimeSeriesSufstatDataPolicy &operator=(
        const TimeSeriesSufstatDataPolicy &rhs) {
      if (&rhs == this) return *this;
      Base::operator=(rhs);
      suf_ = rhs.suf_->clone();
      return *this;
    }

    using Base::DAT;
    using Base::DAT_1;

    template <class FwdIt>
    void set_data(FwdIt Beg, FwdIt End) {
      Base::set_data(Beg, End);
      refresh_suf();
    }

    virtual void set_data(const Ptr<DataSeriesType> &d) {
      Base::set_data(d);
      refresh_suf();
    }

    virtual void add_data_series(const Ptr<DataSeriesType> &d) {
      Base::add_data_series(d);
      update_suf(d);
    }

    virtual void add_data_point(const Ptr<DataPointType> &d) {
      Base::add_data_point(d);
      this->suf()->update(d);
    }

    virtual void add_data(const Ptr<Data> &d) {
      Base::add_data(d);
      this->suf()->update(d);
    }

    virtual void clear_data() {
      Base::clear_data();
      this->suf()->clear();
    }

    using Base::dat;

    const Ptr<SUF> suf() const { return suf_; }

    void clear_suf() { suf_->clear(); }

    void update_suf(const Ptr<DataPointType> &d) { suf_->update(d); }

    void update_suf(const Ptr<DataSeriesType> &series_pointer) {
      const DataSeriesType &series(*series_pointer);
      for (uint i = 0; i < series.size(); ++i) {
        suf_->update(series[i]);
      }
    }

    void refresh_suf() {
      suf()->clear();
      uint n = this->nseries();
      for (uint i = 0; i < n; ++i) {
        const DataSeriesType &d(this->dat(i));
        for (uint j = 0; j < d.size(); ++j) suf()->update(d[j]);
      }
    }

   private:
    Ptr<SUF> suf_;
  };

}  // namespace BOOM
#endif  // BOOM_TIME_SERIES_SUFSTAT_DATA_POLICY_HPP
