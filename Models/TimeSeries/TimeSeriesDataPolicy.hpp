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
  template <class DATA, class SERIES = TimeSeries<DATA>>
  class TimeSeriesDataPolicy : virtual public Model {
    // Data policy for time series models.  Many such models will have
    // a single time series as their data, but some will have many
    // time series.  This class handles both jobs.

   public:
    typedef DATA DataPointType;
    typedef SERIES DataSeriesType;  // should inherit from TimeSeries<DATA>
    typedef TimeSeriesDataPolicy<DATA, SERIES> DataPolicy;

    TimeSeriesDataPolicy() {}

    explicit TimeSeriesDataPolicy(const Ptr<DataSeriesType> &ts)
        : ts_(1, ts) {}

    TimeSeriesDataPolicy *clone() const = 0;

    template <class FwdIt>
    void set_data(FwdIt Beg, FwdIt End) {
      NEW(DataSeriesType, ts)(Beg, End);
      this->set_data(ts);
    }

    virtual void set_data(const Ptr<DataSeriesType> &ts) {
      ts_.clear();
      add_data_series(ts);
    }

    virtual void add_data_series(const Ptr<DataSeriesType> &ts) {
      ts_.push_back(ts);
    }

    virtual void add_data_point(const Ptr<DataPointType> &d) {
      if (ts_.empty()) {
        NEW(DataSeriesType, ts)();
        ts_.push_back(ts);
      }
      ts_.back()->add_data_point(d);
    }

    virtual void add_data(const Ptr<Data> &d) {
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

    virtual void combine_data(const Model &m, bool just_suf = true) {
      const DataPolicy &other(dynamic_cast<const DataPolicy &>(m));
      ts_.reserve(ts_.size() + other.ts_.size());
      ts_.insert(ts_.end(), other.ts_.begin(), other.ts_.end());
    }

    virtual void clear_data() { ts_.clear(); }

    virtual DataSeriesType &dat(uint i = 0) { return *(ts_[i]); }
    virtual const DataSeriesType &dat(uint i = 0) const { return *(ts_[i]); }

    uint nseries() const { return ts_.size(); }

    Ptr<SERIES> DAT(const Ptr<Data> &dp) const {
      if (!dp) return Ptr<DataSeriesType>();
      return dp.dcast<DataSeriesType>();
    }

    Ptr<DATA> DAT_1(const Ptr<Data> &dp) const {
      if (!dp) return Ptr<DataPointType>();
      return dp.dcast<DataPointType>();
    }
   private:
    std::vector<Ptr<DataSeriesType> > ts_;  // model owns data;
  };

}  // namespace BOOM
#endif  // BOOM_TIME_SERIES_DATA_POLICY_HPP
