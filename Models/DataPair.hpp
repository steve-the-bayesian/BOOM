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
#ifndef BOOM_DATA_PAIR_HPP
#define BOOM_DATA_PAIR_HPP

#include "Models/DataTypes.hpp"
namespace BOOM {

  template <class D1, class D2>
  class DataPair : public Data {
   public:
    DataPair(const Ptr<D1> &d1, const Ptr<D2> &d2) : d1_(d1), d2_(d2) {}
    DataPair(const DataPair &rhs)
        : Data(rhs), d1_(rhs.d1_->clone()), d2_(rhs.d2_->clone()) {}
    DataPair *clone() const override { return new DataPair(*this); }

    std::ostream &display(std::ostream &out) const override {
      return d1_->display(out) << " " << d2_->display(out);
    }
    virtual istream &read(istream &in) {
      d1_->read(in);
      d2_->read(in);
      return (in);
    }
    virtual uint size(bool minimal = true) const {
      return d1_->size(minimal) + d2_->size(minimal);
    }

    Ptr<D1> first() { return d1_; }
    Ptr<D2> second() { return d2_; }
    const Ptr<D1> first() const { return d1_; }
    const Ptr<D2> second() const { return d2_; }

   private:
    Ptr<D1> d1_;
    Ptr<D2> d2_;
  };

}  // namespace BOOM
#endif  // BOOM_DATA_PAIR_HPP
