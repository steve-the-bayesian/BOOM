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

  // An object that will remember a real value, and a memory location where it
  // is stored.  When this object goes out of scope, its destructor resets the
  // variable to the value it had when this object was created.
  class RealValueHolder {
   public:
    explicit RealValueHolder(double &value)
        : value_(value), source_(&value)
    {}
    ~RealValueHolder() {*source_ = value_;}
   private:
    double value_;
    double *source_;
  };

  class Params;
  class ParamHolder {
   public:
    // Store the current value of held_prm in Wsp, to be restored when the
    // ParamHolder goes out of scope.
    ParamHolder(const Ptr<Params> &held_prm, Vector &Wsp);

    // Store teh value of held_prm in Wsp, and replace it with the values in
    // 'x'.  The original value of held_prm will be restored when *this goes out
    // of scope.
    ParamHolder(const Vector &x, const Ptr<Params> &held_prm, Vector &Wsp);

    // Restore the parameter to its initial value.
    ~ParamHolder();

   private:
    Vector &storage_;
    Ptr<Params> prm_;
  };

  // For holding and restoring a vector of parameters.
  class ParamVectorHolder {
   public:
    ParamVectorHolder(const std::vector<Ptr<Params>> &held, Vector &Wsp);
    ParamVectorHolder(const Vector &x, const std::vector<Ptr<Params>> &held, Vector &Wsp);
    ~ParamVectorHolder();

   private:
    Vector &v;
    std::vector<Ptr<Params>> prm;
  };
}  // namespace BOOM
#endif  // BOOM_PARAM_HOLDER_HPP
