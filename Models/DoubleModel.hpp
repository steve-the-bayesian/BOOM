// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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

#ifndef BOOM_DOUBLE_MODEL_HPP
#define BOOM_DOUBLE_MODEL_HPP

#include "Models/ModelTypes.hpp"
namespace BOOM {

  class DoubleModel : virtual public MixtureComponent {
   public:
    virtual double logp(double x) const = 0;
    virtual double sim(RNG &rng = GlobalRng::rng) const = 0;
    DoubleModel *clone() const override = 0;
    virtual double pdf(const Ptr<Data> &dp, bool logscale) const;
    double pdf(const Data *dp, bool logscale) const override;
  };

  class LocationScaleDoubleModel : virtual public DoubleModel {
   public:
    LocationScaleDoubleModel *clone() const override = 0;
    virtual double mean() const = 0;
    virtual double variance() const = 0;
  };

  class dDoubleModel : virtual public DoubleModel {
    // the 'diff' is for differentiable
   public:
    virtual double dlogp(double x, double &g) const = 0;
    dDoubleModel *clone() const override = 0;
  };

  class d2DoubleModel : public dDoubleModel {
   public:
    virtual double d2logp(double x, double &g, double &h) const = 0;
    d2DoubleModel *clone() const override = 0;
  };

  class DiffDoubleModel : public d2DoubleModel {
    // the 'diff' is for differentiable
   public:
    virtual double Logp(double x, double &g, double &h, uint nd) const = 0;
    double logp(double x) const override;
    double dlogp(double x, double &g) const override;
    double d2logp(double x, double &g, double &h) const override;
    DiffDoubleModel *clone() const override = 0;
  };

}  // namespace BOOM

#endif  // BOOM_DOUBLE_MODEL_HPP
