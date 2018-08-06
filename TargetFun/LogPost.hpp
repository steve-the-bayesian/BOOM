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

#ifndef BOOM_LOG_POST_H
#define BOOM_LOG_POST_H

#include <functional>
#include "TargetFun/Loglike.hpp"
#include "cpputil/Ptr.hpp"
#include "numopt.hpp"

namespace BOOM {
  class VectorModel;
  class dVectorModel;
  class d2VectorModel;

  class LogPostTF {
   public:
    LogPostTF(const Target &loglike, const Ptr<VectorModel> &prior);
    double operator()(const Vector &z) const;

   private:
    Target loglike_;
    Ptr<VectorModel> prior_;
  };
  /*----------------------------------------------------------------------*/
  class dLogPostTF : public LogPostTF {
   public:
    dLogPostTF(const dLoglikeTF &loglike, const Ptr<dVectorModel> &prior);
    dLogPostTF(const Target &loglike, const dTarget &dloglike,
               const Ptr<dVectorModel> &prior);
    double operator()(const Vector &z) const {
      return LogPostTF::operator()(z);
    }
    double operator()(const Vector &z, Vector &g) const;

   private:
    dTarget dloglike_;
    Ptr<dVectorModel> dprior_;
  };

  //----------------------------------------------------------------------
  class d2LogPostTF : public dLogPostTF {
   public:
    d2LogPostTF(const d2LoglikeTF &loglike, const Ptr<d2VectorModel> &prior);
    d2LogPostTF(const Target &loglike, const dTarget &dloglike,
                const d2Target &d2loglike, const Ptr<d2VectorModel> &prior);

    double operator()(const Vector &z) const {
      return LogPostTF::operator()(z);
    }
    double operator()(const Vector &z, Vector &g) {
      return dLogPostTF::operator()(z, g);
    }
    double operator()(const Vector &z, Vector &g, Matrix &h) const;

   private:
    std::function<double(const Vector &x, Vector &g, Matrix &h)> d2loglike_;
    Ptr<d2VectorModel> d2prior_;
  };

}  // namespace BOOM
#endif  // BOOM_LOG_POST_HPP
