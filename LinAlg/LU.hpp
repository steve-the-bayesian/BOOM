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

// class for computing the LU factorization of a Matrix

#ifndef BOOM_LU_HPP
#define BOOM_LU_HPP

#include <LinAlg/Matrix.hpp>
#include <vector>

namespace BOOM{
  class Vector;
  class LU{             // LU decomposition of the Matrix A
    Matrix dcmp;
    typedef std::vector<int> ivec;
    ivec pivots;
    bool sing_;
   public:
    LU(const Matrix &m);
    Matrix getL()const;
    Matrix getU()const;
    ivec get_pivots()const;
    double det()const;
    Matrix solve(const Matrix &B)const;   // find X such that AX = B
    Vector solve(const Vector &b)const;   // find x such that Ax = b
    Matrix solveT(const Matrix &B)const;  // find X such that A^TX = B
    Vector solveT(const Vector &b)const;  // find x such that A^Tx = b
    bool singular()const{return sing_;}   // is the Matrix A singular?
  };
}
#endif // BOOM_LU_HPP
