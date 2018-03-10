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
#ifndef BOOM_PARAM_HOLDER_HPP
#define BOOM_PARAM_HOLDER_HPP

#include "LinAlg/Vector.hpp"

#include "cpputil/Ptr.hpp"

namespace BOOM {

  class Params;
  class ParamHolder {
   public:
    ParamHolder(const Ptr<Params> &held_prm, Vector &Wsp);
    ParamHolder(const Vector &x, const Ptr<Params> &held_prm, Vector &Wsp);
    ~ParamHolder();

   private:
    Vector &v;
    Ptr<Params> prm;
  };

  class ParamVectorHolder {
   public:
    typedef std::vector<Ptr<Params> > ParamVector;

   private:
    Vector &v;
    ParamVector prm;

   public:
    ParamVectorHolder(const ParamVector &held, Vector &Wsp);
    ParamVectorHolder(const Vector &x, const ParamVector &held, Vector &Wsp);
    ~ParamVectorHolder();
  };
}  // namespace BOOM
#endif  // BOOM_PARAM_HOLDER_HPP
