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

#include<cpputil/Ptr.hpp>
#include <Models/ModelTypes.hpp>
#include <Models/ParamTypes.hpp>

namespace BOOM{

  class ManyParamPolicy : virtual public Model{
  public:
    typedef ManyParamPolicy ParamPolicy;
    ManyParamPolicy();
    ManyParamPolicy(const ManyParamPolicy &rhs);  // components not copied
    ManyParamPolicy & operator=(const ManyParamPolicy &);


    // the following functions will need to be called during
    // construction of the inheriting Model object

    template<class Fwd>
    void set_params(Fwd b, Fwd e){t_.assign(b,e);}

    template<class Fwd>
    void add_params(Fwd b, Fwd e){std::copy(b,e,back_inserter(t_));}

    void add_params(const Ptr<Params> & p);
    void clear();

    ParamVector parameter_vector() override;
    const ParamVector parameter_vector()const override;

  protected:
    virtual void setup_params()=0;  // to be called during construction
  private:
    ParamVector t_;
  };




}

#endif// BOOM_MANY_PARAM_POLICY_HPP
