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
#ifndef BOOM_SUFSTAT_TYPES_HPP
#define BOOM_SUFSTAT_TYPES_HPP

#include <string>

#include "LinAlg/Vector.hpp"
#include "Models/DataTypes.hpp"
#include "cpputil/Ptr.hpp"
#include "cpputil/report_error.hpp"
#include "uint.hpp"

namespace BOOM {

  // Sufficient statistics.  Having Sufstat inherit from Data helps with
  // exponential family hierarchical models, where each "data point" is really a
  // collection of data for a specific group.  It also makes sense because of
  // the duality between sufficient statistics and model parameters, which
  // inherit from data for the same reason (parameters from one level of a
  // hierarchical model are data for the next level).
  class Sufstat : public Data {
   public:
    virtual void clear() = 0;
    virtual void update(const Ptr<Data> &) = 0;
    virtual void update(const Data &) = 0;
    virtual ~Sufstat() {}
    virtual Sufstat *clone() const override = 0;
    virtual Sufstat *abstract_combine(Sufstat *rhs) = 0;

    virtual Vector vectorize(bool minimal = true) const = 0;
    virtual Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                               bool minimal = true) = 0;
    virtual Vector::const_iterator unvectorize(const Vector &v,
                                               bool minimal = true) = 0;
    virtual std::ostream &print(std::ostream &) const = 0;

    std::ostream &display(std::ostream &out) const override {
      return print(out);
    }

    std::string print_to_string() const;
  };

  inline std::ostream &operator<<(std::ostream &out, const Sufstat &s) {
    return s.print(out);
  }

  Vector vectorize(const std::vector<Ptr<Sufstat>> &v, bool minimal = true);
  void unvectorize(std::vector<Ptr<Sufstat>> &, const Vector &v,
                   bool minimal = true);

  // The following policy helps make concrete Sufstats.  The policy
  // contains richer type information that cannot be included in the
  // abstract base class.

  template <class D>
  class SufstatDetails : virtual public Sufstat {
   public:
    typedef D DataType;
    typedef SufstatDetails<D> SufTraits;
    Ptr<D> DAT(const Ptr<Data> &dp) const { return dp.dcast<DataType>(); }
    virtual void Update(const DataType &) = 0;
    virtual void update(const DataType &d) { Update(d); }
    void update(const Ptr<Data> &dp) override { Update(*(DAT(dp))); }
    virtual void update(const Ptr<D> &dp) { Update(*dp); }
    void update(const Data &d) override {
      Update(dynamic_cast<const DataType &>(d));
    }
  };

  //==================================================================
  template <class D, class SER>
  class TimeSeriesSufstatDetails : virtual public Sufstat {
   public:
    typedef D DataPointType;
    typedef SER DataSeriesType;
    typedef DataPointType DataType;
    typedef TimeSeriesSufstatDetails<D, SER> SufTraits;
    Ptr<D> DAT_1(const Ptr<Data> &dp) const {
      return dp.dcast<DataPointType>();
    }
    Ptr<SER> DAT(const Ptr<Data> &dp) const {
      return dp.dcast<DataSeriesType>();
    }

    virtual void Update(const DataPointType &) = 0;
    virtual void update_series(const DataSeriesType &ds) {
      for (uint i = 0; i < ds.length(); ++i) {
        update(ds[i]);
      }
    }
    virtual void update(const DataPointType &d) { Update(d); }
    void update(const Ptr<Data> &dp) override {
      Ptr<DataPointType> d = DAT_1(dp);
      if (!!d) {
        Update(*d);
        return;
      }

      Ptr<DataSeriesType> ds = DAT(dp);
      if (!!ds) {
        update_series(*ds);
        return;
      }
      report_error(
          "TimeSeriesSfustatDetails::update failed due to "
          "unknown type");
    }
    virtual void update(const Ptr<DataPointType> &dp) { Update(*dp); }

    void update(const Data &d) override {
      // pointer contortions to get around using exceptions resulting
      // from a bad dynamic_cast of a reference
      const Data *data_ptr = &d;
      const D *dp(dynamic_cast<const D *>(data_ptr));
      if (dp) {  // if the dynamic cast failed then this block gets skipped
        Update(*dp);
      } else {
        const SER *ds(dynamic_cast<const SER *>(data_ptr));
        update_series(*ds);
      }
    }
  };

}  // namespace BOOM
#endif  // BOOM_SUFSTAT_TYPES_HPP
