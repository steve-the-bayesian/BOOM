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
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
   USA
 */

#ifndef BOOM_MOMENTS_HPP
#define BOOM_MOMENTS_HPP

#include <vector>
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"

namespace BOOM {

  Vector mean(const Matrix &m);
  SpdMatrix var(const Matrix &m);
  SpdMatrix cor(const Matrix &m);

  double mean(const Vector &x);
  double mean(const VectorView &x);
  double mean(const ConstVectorView &x);

  double var(const Vector &x);
  double var(const VectorView &x);
  double var(const ConstVectorView &x);

  double sd(const Vector &x);
  double sd(const VectorView &x);
  double sd(const ConstVectorView &x);

  double mean(const std::vector<double> &x);
  double var(const std::vector<double> &x);
  double sd(const std::vector<double> &x);
  double cor(const std::vector<double> &x, const std::vector<double> &y);

  double mean(const std::vector<double> &x, double missing_value_code);
  double var(const std::vector<double> &x, double missing_value_code);
  double sd(const std::vector<double> &x, double missing_value_code);

  double mean(const std::vector<double> &x, const std::vector<bool> &observed);
  double var(const std::vector<double> &x, const std::vector<bool> &observed);
  double sd(const std::vector<double> &x, const std::vector<bool> &observed);
}  // namespace BOOM
#endif  // BOOM_MOMENTS_HPP
