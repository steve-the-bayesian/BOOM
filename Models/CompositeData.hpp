// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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
#ifndef BOOM_COMPOSITE_DATA_HPP
#define BOOM_COMPOSITE_DATA_HPP

#include "Models/DataTypes.hpp"
namespace BOOM {

  // CompositeData is a multivariate observation (on a single subject)
  // of potentially disparate data types.  This is the data type for a
  // CompositeModel.
  class CompositeData : virtual public Data {
   public:
    // Use this constructor to create a new CompositeData and add() in
    // elements one at a time.  This approach is more flexible than
    // the vector constructor because add() can take advantage of
    // polymorphic arguments.
    CompositeData();

    // Use this constructor to create a new Composite Data when you've
    // already got the component data stored away in a vector.
    explicit CompositeData(const std::vector<Ptr<Data>> &d);

    CompositeData *clone() const override;
    std::ostream &display(std::ostream &) const override;

    // Number of composite data elements.  This can be smaller than
    // size() when some elements are vectors, matrices, regression
    // data, etc.
    uint dim() const;

    Ptr<Data> get_ptr(uint i);
    const Data *get(uint i) const;
    void add(const Ptr<Data> &dp);

   private:
    std::vector<Ptr<Data>> dat_;
  };

}  // namespace BOOM

#endif  // BOOM_COMPOSITE_DATA_HPP
