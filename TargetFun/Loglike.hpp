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

#ifndef MODEL_TF_H
#define MODEL_TF_H

#include "Models/ModelTypes.hpp"

namespace BOOM {
  class LoglikeTF {
   public:
    explicit LoglikeTF(LoglikeModel *model) : mod(model) {}
    double operator()(const Vector &x) const { return mod->loglike(x); }

   private:
    LoglikeModel *mod;  // provides loglike(x);
  };
  //----------------------------------------------------------------------

  class dLoglikeTF : public LoglikeTF {
   public:
    explicit dLoglikeTF(dLoglikeModel *d) : LoglikeTF(d), dmod(d) {}
    double operator()(const Vector &x) const {
      return LoglikeTF::operator()(x);
    }
    double operator()(const Vector &x, Vector &g) const {
      return dmod->dloglike(x, g);
    }

   private:
    dLoglikeModel *dmod;
  };

  //----------------------------------------------------------------------
  class d2LoglikeTF : public dLoglikeTF {
   public:
    explicit d2LoglikeTF(d2LoglikeModel *d2) : dLoglikeTF(d2), d2mod(d2) {}
    double operator()(const Vector &x) const {
      return LoglikeTF::operator()(x);
    }
    double operator()(const Vector &x, Vector &g) const {
      return dLoglikeTF::operator()(x, g);
    }
    double operator()(const Vector &x, Vector &g, Matrix &h) const {
      return d2mod->d2loglike(x, g, h);
    }

   private:
    d2LoglikeModel *d2mod;
  };
  //------------------------------------------------------------
}  // namespace BOOM
#endif  // MODEL_TF_H
