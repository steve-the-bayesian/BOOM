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

#ifndef BOOM_TIME_SERIES_DATA_POLICY_HPP
#define BOOM_TIME_SERIES_DATA_POLICY_HPP
#include <vector>
#include "Models/ModelTypes.hpp"
#include "Models/TimeSeries/TimeSeries.hpp"
#include "cpputil/Ptr.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  //======================================================================
  template <class D, class SER = TimeSeries<D> >
  class TimeSeriesDataInfoPolicy : virtual public Model {
   public:
    typedef D DataPointType;
    typedef SER DataSeriesType;  // should inherit from TimeSeries<D>
    typedef TimeSeriesDataInfoPolicy<D, SER> DataInfoPolicy;

    TimeSeriesDataInfoPolicy *clone() const override = 0;
    Ptr<DataSeriesType> DAT(const Ptr<Data> &dp) const;
    Ptr<DataPointType> DAT_1(const Ptr<Data> &dp) const;
  };
  //======================================================================
  template <class D, class SER = TimeSeries<D> >
  class TimeSeriesDataPolicy : virtual public TimeSeriesDataInfoPolicy<D, SER> {
    // Data policy for time series models.  Many such models will have
    // a single time series as their data, but some will have many
    // time series.  This class handles both jobs.

   public:
    typedef D DataPointType;
    typedef SER DataSeriesType;  // should inherit from TimeSeries<D>
    typedef TimeSeriesDataInfoPolicy<D, SER> DataInfo;
    typedef TimeSeriesDataPolicy<D, SER> DataPolicy;

    TimeSeriesDataPolicy();
    explicit TimeSeriesDataPolicy(const Ptr<DataSeriesType> &ds);
    TimeSeriesDataPolicy *clone() const = 0;

    template <class FwdIt>
    void set_data(FwdIt Beg, FwdIt End);
    virtual void set_data(const Ptr<DataSeriesType> &d);

    virtual void add_data_series(const Ptr<DataSeriesType> &d);
    virtual void add_data_point(const Ptr<D> &d);
    virtual void add_data(const Ptr<Data> &d);
    virtual void combine_data(const Model &m, bool just_suf = true);

    virtual void clear_data();

    virtual DataSeriesType &dat(uint i = 0) { return *(ts_[i]); }
    virtual const DataSeriesType &dat(uint i = 0) const { return *(ts_[i]); }

    uint nseries() const { return ts_.size(); }

   private:
    std::vector<Ptr<DataSeriesType> > ts_;  // model owns data;
    bool linked(const Ptr<D> &d) const;
    // linked(d) returns true if d's next or prev links have been set
  };

  //======================================================================
  // implementation follows below
  //______________________________________________________________________

  template <class D, class TS>
  TimeSeriesDataPolicy<D, TS>::TimeSeriesDataPolicy() {}

  template <class D, class TS>
  TimeSeriesDataPolicy<D, TS>::TimeSeriesDataPolicy(const Ptr<TS> &ts)
      : ts_(1, ts) {}

  template <class D, class TS>
  template <class FwdIt>
  void TimeSeriesDataPolicy<D, TS>::set_data(FwdIt Beg, FwdIt End) {
    NEW(DataSeriesType, ts)(Beg, End);
    this->set_data(ts);
  }

  template <class D, class TS>
  void TimeSeriesDataPolicy<D, TS>::set_data(const Ptr<DataSeriesType> &ts) {
    ts_.clear();
    add_data_series(ts);
  }

  template <class D, class TS>
  void TimeSeriesDataPolicy<D, TS>::add_data_series(
      const Ptr<DataSeriesType> &ts) {
    ts_.push_back(ts);
  }

  template <class D, class TS>
  void TimeSeriesDataPolicy<D, TS>::add_data_point(const Ptr<D> &dp) {
    if (ts_.empty()) {
      NEW(DataSeriesType, ts)();
      ts_.push_back(ts);
    }
    ts_.back()->add_1(dp);
  }

  template <class D, class TS>
  void TimeSeriesDataPolicy<D, TS>::add_data(const Ptr<Data> &d) {
    Ptr<DataSeriesType> tsp = this->DAT(d);
    if (!!tsp) {
      add_data_series(tsp);
      return;
    }

    Ptr<DataPointType> dp = this->DAT_1(d);
    if (!!dp) {
      add_data_point(dp);
      return;
    }

    ostringstream err;
    err << "data value " << *d << " could not be cast to a "
        << "time series or a time series data point.  " << endl;
    report_error(err.str());
  }

  template <class D, class TS>
  void TimeSeriesDataPolicy<D, TS>::clear_data() {
    ts_.clear();
  }

  template <class D, class TS>
  void TimeSeriesDataPolicy<D, TS>::combine_data(const Model &m, bool) {
    const DataPolicy &other(dynamic_cast<const DataPolicy &>(m));
    ts_.reserve(ts_.size() + other.ts_.size());
    ts_.insert(ts_.end(), other.ts_.begin(), other.ts_.end());
  }

  //============================================================
  template <class D, class TS>
  Ptr<TS> TimeSeriesDataInfoPolicy<D, TS>::DAT(const Ptr<Data> &dp) const {
    if (!dp) return Ptr<DataSeriesType>();
    return dp.dcast<DataSeriesType>();
  }

  template <class D, class TS>
  Ptr<D> TimeSeriesDataInfoPolicy<D, TS>::DAT_1(const Ptr<Data> &dp) const {
    if (!dp) return Ptr<DataPointType>();
    return dp.dcast<DataPointType>();
  }

  //======================================================================

}  // namespace BOOM
#endif  // BOOM_TIME_SERIES_DATA_POLICY_HPP
