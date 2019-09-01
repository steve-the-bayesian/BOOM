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

#ifndef BOOM_DATA_INFO_POLICY_HPP
#define BOOM_DATA_INFO_POLICY_HPP

#include "Models/ModelTypes.hpp"
namespace BOOM {
  template <class D>
  class DefaultDataInfoPolicy : virtual public Model {
   public:
    typedef D DataType;
    typedef std::vector<Ptr<DataType> > DatasetType;
    typedef DefaultDataInfoPolicy<D> DataTraits;

    virtual DatasetType &dat() = 0;
    virtual const DatasetType &dat() const = 0;

    Ptr<DataType> DAT(const Ptr<Data> &dp) const {
      if (!!dp) return dp.dcast<DataType>();
      return Ptr<DataType>();
    }

    const DataType *DAT(const Data *dp) const {
      return dp ? dynamic_cast<const DataType *>(dp) : NULL;
    }
  };
  //======================================================================
  template <>
  class DefaultDataInfoPolicy<Data> : virtual public Model {
   public:
    typedef Data DataType;
    typedef std::vector<Ptr<DataType> > DatasetType;
    typedef DefaultDataInfoPolicy<Data> DataTraits;

    virtual DatasetType &dat() = 0;
    virtual const DatasetType &dat() const = 0;

    Ptr<DataType> DAT(const Ptr<Data> &dp) const { return dp; }
  };

}  // namespace BOOM
#endif  // BOOM_DATA_INFO_POLICY_HPP
