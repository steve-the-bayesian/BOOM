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
#ifndef BOOM_EM_MIXTURE_COMPONENT_HPP
#define BOOM_EM_MIXTURE_COMPONENT_HPP

#include "Models/DataTypes.hpp"
#include "Models/ModelTypes.hpp"

namespace BOOM {
  class EmMixtureComponent : virtual public MixtureComponent,
                             virtual public MLE_Model,
                             virtual public PosteriorModeModel {
   public:
    // The rule-of-five members are explicitly defined below because the
    // implicit virtual assignment move operator has trouble with virtual base
    // classes and had to be defined explicitly.  Defining one means you need to
    // define all, or you get stupid compiler warnings.
    EmMixtureComponent() = default;
    ~EmMixtureComponent() = default;
    
    EmMixtureComponent(const EmMixtureComponent &rhs)
        : Model(rhs),
          MixtureComponent(rhs),
          MLE_Model(rhs),
          PosteriorModeModel(rhs)
    {}

    EmMixtureComponent(EmMixtureComponent &&rhs)
        : Model(rhs),
          MixtureComponent(rhs),
          MLE_Model(rhs),
          PosteriorModeModel(rhs)
    {}

    EmMixtureComponent *clone() const override = 0;
    virtual void add_mixture_data(const Ptr<Data> &, double weight) = 0;

    EmMixtureComponent &operator=(const EmMixtureComponent &rhs) {
      if (&rhs != this) {
        MixtureComponent::operator=(rhs);
        MLE_Model::operator=(rhs);
        PosteriorModeModel::operator=(rhs);
      }
      return *this;
    }

    EmMixtureComponent &operator=(EmMixtureComponent &&rhs) {
      if (&rhs != this) {
        MixtureComponent::operator=(rhs);
        MLE_Model::operator=(rhs);
        PosteriorModeModel::operator=(rhs);
      }
      return *this;
    }
  };
}  // namespace BOOM
#endif  // BOOM_EM_MIXTURE_COMPONENT_HPP
