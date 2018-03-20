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

#ifndef BOOM_AUGMENTED_TIME_SERIES_DATA_POLICY_HPP
#define BOOM_AUGMENTED_TIME_SERIES_DATA_POLICY_HPP

#include "Models/TimeSeries/AugmentedTimeSeries.hpp"
#include "Models/TimeSeries/TimeSeriesDataPolicy.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  template <class D, class F>
  class AugmentedTimeSeriesDataPolicy
      : public TimeSeriesDataInfoPolicy<D, AugmentedTimeSeries<D, F> > {
   public:
    typedef D DataPointType;
    typedef F InitialDataType;
    typedef AugmentedTimeSeries<D, F>
        DataSeriesType;  // should inherit from TimeSeries<D>
    typedef TimeSeriesDataInfoPolicy<D, DataSeriesType> DataInfo;
    typedef AugmentedTimeSeriesDataPolicy<D, F> DataPolicy;

    AugmentedTimeSeriesDataPolicy();
    explicit AugmentedTimeSeriesDataPolicy(const Ptr<DataSeriesType> &ds);
    AugmentedTimeSeriesDataPolicy *clone() const = 0;

    // extended conversion function
    using DataInfo::DAT;
    using DataInfo::DAT_1;
    Ptr<F> DAT_0(const Ptr<Data> &dp) const {
      if (!dp) return nullptr;
      return dp.dcast<F>();
    }

    virtual void set_data(const Ptr<DataSeriesType> &d);
    virtual void add_data_series(const Ptr<DataSeriesType> &d);
    virtual void start_new_series(const Ptr<F> &);
    virtual void add_data_point(const Ptr<D> &d);
    virtual void add_data(const Ptr<Data> &d);
    // add_data will check whether data is initial data point, regular
    // data point, or new series

    virtual void clear_data();

    virtual DataSeriesType &dat(uint i = 0) { return *(ts_[i]); }
    virtual const DataSeriesType &dat(uint i = 0) const { return *(ts_[i]); }

    uint nseries() const { return ts_.size(); }

   private:
    std::vector<Ptr<AugmentedTimeSeries<D, F> > > ts_;
  };

  //______________________________________________________________________
  // Life would be easier if I could inherit from
  // TimeSeriesDataPolicy, but the
  // TimeSeriesDataPolicy::add_data_point function takes an action
  // that would be illegal if TS=AugmentedTimeSeries

  // If following specialization would compile that would fix the
  // problem, but the compiler complains about using an incomplete
  // type

  //   template <class D, class F>
  //   void TimeSeriesDataPolicy<D, AugmentedTimeSeries<D,F>
  //   >::add_data_point(const Ptr<D> &)
  //   //void TimeSeriesDataPolicy<D, TimeSeries<D> >::add_data_point(const
  //   Ptr<D> &)
  //   {}

  //______________________________________________________________________

  template <class D, class F>
  AugmentedTimeSeriesDataPolicy<D, F>::AugmentedTimeSeriesDataPolicy() {}

  template <class D, class F>
  AugmentedTimeSeriesDataPolicy<D, F>::AugmentedTimeSeriesDataPolicy(
      const Ptr<DataSeriesType> &ds)
      : ts_(1, ds) {}

  template <class D, class F>
  void AugmentedTimeSeriesDataPolicy<D, F>::set_data(
      const Ptr<DataSeriesType> &ts) {
    ts_.clear();
    add_data_series(ts);
  }

  template <class D, class F>
  void AugmentedTimeSeriesDataPolicy<D, F>::add_data_series(
      const Ptr<DataSeriesType> &ts) {
    ts_.push_back(ts);
  }

  template <class D, class F>
  void AugmentedTimeSeriesDataPolicy<D, F>::start_new_series(const Ptr<F> &dp) {
    NEW(DataSeriesType, ts)(dp);
    add_data_series(ts);
  }

  template <class D, class F>
  void AugmentedTimeSeriesDataPolicy<D, F>::add_data_point(const Ptr<D> &dp) {
    uint n = this->nseries();
    if (n == 0) {
      ostringstream err;
      err << "You need at least one data series before you add a data point."
          << endl;
      report_error(err.str());
    }
    this->dat(n - 1).add_1(dp);
  }

  template <class D, class F>
  void AugmentedTimeSeriesDataPolicy<D, F>::add_data(const Ptr<Data> &dp) {
    Ptr<DataSeriesType> ts(DAT(dp));
    if (!!ts) {
      add_data_series(ts);
      return;
    }

    Ptr<D> obs(DAT_1(dp));
    if (!!obs) {
      add_data_point(obs);
      return;
    }

    Ptr<F> init(DAT_0(dp));
    if (!!init) {
      start_new_series(init);
      return;
    }

    ostringstream err;
    err << "data value: " << *dp << " could not be cast to an augmented "
        << "time series, an element of the augmented time series, or "
        << "the initial data for an augmented time series." << endl;
    report_error(err.str());
  }

  template <class D, class F>
  void AugmentedTimeSeriesDataPolicy<D, F>::clear_data() {
    ts_.clear();
  }

}  // namespace BOOM

#endif  // BOOM_AUGMENTED_TIME_SERIES_DATA_POLICY_HPP
