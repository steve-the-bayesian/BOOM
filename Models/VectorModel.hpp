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

#ifndef BOOM_VECTOR_MODEL_HPP
#define BOOM_VECTOR_MODEL_HPP
#include "Models/ModelTypes.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  // Mix-in model classes that supply logp(Vec);

  class VectorModel : virtual public Model {
   public:
    virtual double logp(const Vector &x) const = 0;
    VectorModel *clone() const override = 0;
    virtual Vector sim(RNG &rng = GlobalRng::rng) const = 0;
  };

  class LocationScaleVectorModel : virtual public VectorModel {
   public:
    virtual const Vector &mu() const = 0;
    virtual const SpdMatrix &Sigma() const = 0;
    virtual const SpdMatrix &siginv() const = 0;
    virtual double ldsi() const = 0;
  };

  class dVectorModel : virtual public VectorModel {
   public:
    virtual double dlogp(const Vector &x, Vector &g) const = 0;
    dVectorModel *clone() const override = 0;
  };

  class d2VectorModel : public dVectorModel {
   public:
    virtual double d2logp(const Vector &x, Vector &g, Matrix &h) const = 0;
    d2VectorModel *clone() const override = 0;
  };

  class DiffVectorModel : public d2VectorModel {
   public:
    DiffVectorModel *clone() const override = 0;
    double logp(const Vector &x) const override;
    double dlogp(const Vector &x, Vector &g) const override;
    double d2logp(const Vector &x, Vector &g, Matrix &h) const override;

    // Args:
    //   x: The location where the density is to be evaluated.
    //   g: Gradient of the density at x.
    //   h: Hessian of the density at x.
    //   nd:  Number of derivatives desired.
    // Returns:
    //  The log density at x.  If nd > 0 then 'g' is filled with the
    //  gradient at x.  If nd > 1 then 'h' is filled with the Hessian
    //  at x.  Neither g nor h is used if nd is below the relevant
    //  threshold.
    virtual double Logp(const Vector &x, Vector &g, Matrix &h,
                        uint nd) const = 0;
  };

}  // namespace BOOM

#endif  // BOOM_VECTOR_MODEL_HPP
