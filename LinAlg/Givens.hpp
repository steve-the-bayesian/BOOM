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

#ifndef BOOM_GIVENS_HPP
#define BOOM_GIVENS_HPP
#include <iosfwd>

namespace BOOM {
  class Selector;
  using std::ostream;
  class Matrix;

  class GivensRotation {
   public:
    GivensRotation(const Matrix &A, int I, int J);
    GivensRotation(int i, int j, double c, double s);

    std::ostream &print(std::ostream &out) const;
    GivensRotation trans() const;

   private:
    friend Matrix &operator*(const GivensRotation &, Matrix &);
    friend Matrix &operator*(Matrix &, const GivensRotation &);
    int i, j;
    double c, s;
  };

  std::ostream &operator<<(std::ostream &out, const GivensRotation &G);
  Matrix &operator*(const GivensRotation &G, Matrix &A);
  Matrix &operator*(Matrix &A, const GivensRotation &G);
  Matrix triangulate(const Matrix &U, const Selector &inc,
                     bool chop_zero_rows = false);
}  // namespace BOOM
#endif  // BOOM_GIVENS_HPP
