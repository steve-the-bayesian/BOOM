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
#ifndef BOOM_MANY_PARAM_POLICY_HPP
#define BOOM_MANY_PARAM_POLICY_HPP

/*======================================================================
  Use this policy when the number of model parameters is not known at
  compile time.  The model owns the paramters.  Compare to
  CompositeParamPolicy, where the parameters are owned by sub-models.
  ======================================================================*/

#include "Models/ModelTypes.hpp"
#include "Models/ParamTypes.hpp"
#include "cpputil/Ptr.hpp"

namespace BOOM {

  class ManyParamPolicy : virtual public Model {
   public:
    typedef ManyParamPolicy ParamPolicy;
    ManyParamPolicy();

    // Copy and assignment are hard, because we don't know how the parameters
    // are organized in the concrete class.  Child classes will to call
    // ParamPolicy::add_params on each of their parameters after assignment.
    ManyParamPolicy(const ManyParamPolicy &rhs);
    ManyParamPolicy &operator=(const ManyParamPolicy &);

    // Moving works by default.
    ManyParamPolicy(ManyParamPolicy &&rhs) = default;
    ManyParamPolicy &operator=(ManyParamPolicy &&rhs) = default;

    void add_params(const Ptr<Params> &p) {t_.push_back(p);}
    void clear() {t_.clear();}

    std::vector<Ptr<Params>> parameter_vector() override {return t_;}
    const std::vector<Ptr<Params>> parameter_vector() const override {
      return t_;
    }

   private:
    std::vector<Ptr<Params>> t_;
  };
}  // namespace BOOM

#endif  // BOOM_MANY_PARAM_POLICY_HPP
