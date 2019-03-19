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
#ifndef BOOM_WEIGHED_DATA_HPP
#define BOOM_WEIGHED_DATA_HPP
#include "Models/DataTypes.hpp"

namespace BOOM {
  template <class DAT, class WGT = DoubleData>
  class WeightedData : virtual public Data {
    Ptr<DAT> dat_;
    Ptr<WGT> w_;

   public:
    typedef typename DAT::value_type value_type;
    typedef typename WGT::value_type weight_type;
    //    WeightedData(const value_type &x);
    WeightedData(const Ptr<DAT> &d, const weight_type &W);
    WeightedData(const Ptr<DAT> &d, Ptr<WGT> W);
    WeightedData(const WeightedData &rhs);
    WeightedData *clone() const override { return new WeightedData(*this); }

    std::ostream &display(std::ostream &out) const override;

    virtual void set_weight(const weight_type &w);
    const weight_type &weight() const;
    virtual void set(const value_type &v) { dat_->set(v); }
    virtual const value_type &value() const { return dat_->value(); }
  };

  typedef WeightedData<VectorData> WeightedVectorData;
  typedef WeightedData<DoubleData> WeightedDoubleData;

  //------------------------------------------------------------

  template <class D, class W>
  WeightedData<D, W>::WeightedData(const Ptr<D> &d, const weight_type &w)
      : dat_(d), w_(new W(w)) {}

  template <class D, class W>
  WeightedData<D, W>::WeightedData(const Ptr<D> &d, Ptr<W> w)
      : dat_(d), w_(w) {}

  template <class D, class W>
  WeightedData<D, W>::WeightedData(const WeightedData &rhs)
      : Data(rhs), dat_(rhs.dat_->clone()), w_(rhs.w_->clone()) {}

  template <class D, class W>
  std::ostream &WeightedData<D, W>::display(std::ostream &out) const {
    w_->display(out);
    out << " ";
    dat_->display(out);
    return out;
  }

  template <class D, class W>
  void WeightedData<D, W>::set_weight(const weight_type &w) {
    w_->set(w);
  }

  template <class D, class W>
  const typename W::value_type &WeightedData<D, W>::weight() const {
    return w_->value();
  }

}  // namespace BOOM
#endif  // BOOM_WEIGHED_DATA_HPP
