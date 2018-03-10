// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#ifndef BOOM_SPD_MODEL_HPP
#define BOOM_SPD_MODEL_HPP

#include "Models/ModelTypes.hpp"

namespace BOOM {

  // A mix-in class indicating that the model is capable of
  class SpdModel : virtual public MixtureComponent {
   public:
    virtual double logp(const SpdMatrix &) const = 0;
    SpdModel *clone() const override = 0;
    double pdf(const Data *dp, bool logscale) const override;
  };

}  // namespace BOOM

#endif  // BOOM_SPD_MODEL_HPP
