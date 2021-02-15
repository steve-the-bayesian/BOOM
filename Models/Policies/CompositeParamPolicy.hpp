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
#ifndef BOOM_COMPOSITE_MODEL_PARAM_POLICY
#define BOOM_COMPOSITE_MODEL_PARAM_POLICY

/*======================================================================
  Use this policy when the Model is defined as a composite of several
  sub-models.  E.g. latent variable models.  If the model just happens
  to have many paramters use ManyParamPolicy instead.
  ======================================================================*/

#include "Models/ModelTypes.hpp"
#include "Models/ParamTypes.hpp"
#include "cpputil/Ptr.hpp"
namespace BOOM {

  class CompositeParamPolicy : virtual public Model {
   public:
    typedef CompositeParamPolicy ParamPolicy;
    CompositeParamPolicy();

    template <class FwdIt>  // *FwdIt is a Ptr<Model>
    CompositeParamPolicy(FwdIt b, FwdIt e);

    // When a composite model is copied, the sub models are not copied.  This is
    // because the owning model often needs to take some sort of additional
    // action prior to adding a model.
    CompositeParamPolicy(
        const CompositeParamPolicy &rhs);  // components not copied
    CompositeParamPolicy(CompositeParamPolicy &&rhs) = default;

    CompositeParamPolicy &operator=(const CompositeParamPolicy &);
    CompositeParamPolicy &operator=(CompositeParamPolicy &&) = default;

    void add_model(const Ptr<Model> &);
    void drop_model(const Ptr<Model> &);
    void clear();

    template <class Fwd>
    void set_models(Fwd b, Fwd e);

    std::vector<Ptr<Params>> parameter_vector() override;
    const std::vector<Ptr<Params>> parameter_vector() const override;

    void add_params(const Ptr<Params> &);

   private:
    bool have_model(const Ptr<Model> &) const;
    std::vector<Ptr<Model> > models_;
    std::vector<Ptr<Params>> t_;
  };

  template <class Fwd>
  void CompositeParamPolicy::set_models(Fwd b, Fwd e) {
    models_.clear();
    std::copy(b, e, back_inserter(models_));
    t_.clear();
    for (uint i = 0; i < models_.size(); ++i) {
      std::vector<Ptr<Params>> tmp(models_[i]->parameter_vector());
      std::copy(tmp.begin(), tmp.end(), back_inserter(t_));
    }
  }

  template <class Fwd>
  CompositeParamPolicy::CompositeParamPolicy(Fwd b, Fwd e) : models_(), t_() {
    set_models(b, e);
  }

}  // namespace BOOM
#endif  // BOOM_COMPOSITE_MODEL_PARAM_POLICY
