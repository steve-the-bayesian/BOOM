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
#ifndef BOOM_PARAM_POLICY_2_HPP
#define BOOM_PARAM_POLICY_2_HPP

#include "Models/ModelTypes.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  template <class P1, class P2>
  class ParamPolicy_2 : virtual public Model {
   public:
    typedef P1 param_type_1;
    typedef P2 param_type_2;
    typedef ParamPolicy_2<P1, P2> ParamPolicy;

    ParamPolicy_2();
    ParamPolicy_2(const Ptr<P1> &p1, Ptr<P2> p2);
    ParamPolicy_2(const ParamPolicy_2 &rhs);
    ParamPolicy_2(ParamPolicy_2 &&rhs) = default;
    ParamPolicy_2<P1, P2> &operator=(const ParamPolicy_2 &rhs);
    ParamPolicy_2<P1, P2> &operator=(ParamPolicy_2 &&rhs) = default;

    // One can call set_params() to set the parameters of an empty ParamPolicy_2
    // post-construction.
    void set_params(const Ptr<P1> &p1, Ptr<P2> p2);
    
    Ptr<P1> prm1() { return prm1_; }
    const Ptr<P1> prm1() const { return prm1_; }
    P1 &prm1_ref() { return *prm1_; }
    const P1 &prm1_ref() const { return *prm1_; }

    Ptr<P2> prm2() { return prm2_; }
    const Ptr<P2> prm2() const { return prm2_; }
    P2 &prm2_ref() { return *prm2_; }
    const P2 &prm2_ref() const { return *prm2_; }

    // over-rides for abstract base Model
    ParamVector parameter_vector() override { return parameter_vector_; }
    const ParamVector parameter_vector() const override {
      return parameter_vector_;
    }

    // Calling disable_parameter_1 removes parameter_1 from the vector of
    // parameters.  Likewise for disable_parameter_2.
    void disable_parameter_1() {
      parameter_vector_ = ParamVector(1, prm2_);
    }
    void disable_parameter_2() {
      parameter_vector_ = ParamVector(1, prm1_);
    }
    
   private:
    Ptr<P1> prm1_;
    Ptr<P2> prm2_;
    ParamVector parameter_vector_;

    // An implementation utility to help set up the parameter_vector_ during
    // copy construction and assignment.  To be called after prm1_ and prm2_ are
    // constructed.
    void copy_parameter_vector(const ParamPolicy_2 &rhs);
  };
  //------------------------------------------------------------

  template <class P1, class P2>
  ParamPolicy_2<P1, P2>::ParamPolicy_2() : prm1_(), prm2_() {
    parameter_vector_ = ParamVector{prm1_, prm2_};
  }
  template <class P1, class P2>
  ParamPolicy_2<P1, P2>::ParamPolicy_2(const Ptr<P1> &p1, Ptr<P2> p2)
      : prm1_(p1), prm2_(p2), parameter_vector_{prm1_, prm2_} {
  }

  template <class P1, class P2>
  void ParamPolicy_2<P1, P2>::copy_parameter_vector(const ParamPolicy_2 &rhs) {
    if (rhs.parameter_vector_.size() == 2) {
      parameter_vector_ = ParamVector{prm1_, prm2_};
    } else if (rhs.parameter_vector_.size() == 1) {
      if (rhs.parameter_vector_[0] == rhs.prm1_) {
        disable_parameter_2();
      } else if (rhs.parameter_vector_[0] == rhs.prm2_) {
        disable_parameter_1();
      } else {
        report_error("Something went horribly wrong when copying "
                     "ParamPolicy_2.");
      }
    }
  }

  template <class P1, class P2>
  ParamPolicy_2<P1, P2>::ParamPolicy_2(const ParamPolicy_2 &rhs)
      : Model(rhs), prm1_(rhs.prm1_->clone()), prm2_(rhs.prm2_->clone()) {
    copy_parameter_vector(rhs);
  }

  template <class P1, class P2>
  ParamPolicy_2<P1, P2> &ParamPolicy_2<P1, P2>::operator=(
      const ParamPolicy_2 &rhs) {
    if (&rhs != this) {
      prm1_ = rhs.prm1_->clone();
      prm2_ = rhs.prm2_->clone();
      copy_parameter_vector(rhs);
    }
    return *this;
  }

  template <class P1, class P2>
  void ParamPolicy_2<P1, P2>::set_params(const Ptr<P1> &p1, Ptr<P2> p2) {
    prm1_ = p1;
    prm2_ = p2;
    parameter_vector_ = ParamVector{prm1_, prm2_};
  }

}  // namespace BOOM

#endif  // BOOM_PARAM_POLICY_2_HPP
