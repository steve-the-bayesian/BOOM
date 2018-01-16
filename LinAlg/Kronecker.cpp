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
#include <LinAlg/Kronecker.hpp>

namespace BOOM{
  Matrix Kronecker(const Matrix &A, const Matrix &B){
    uint nra = A.nrow();
    uint nca = A.ncol();

    Matrix tmp = A(0,0)*B;
    Matrix ans(tmp);
    for(uint j=1; j<nca; ++j){
      tmp = A(0,j)*B;
      ans = cbind(ans,tmp);
    }

    for(uint i=1; i<nra; ++i){
      tmp = A(i,0)*B;
      Matrix row(tmp);
      for(uint j = 1; j<nca; ++j){
        tmp = A(i,j)*B;
        row = cbind(row, tmp);
      }
      ans = rbind(ans,row);
    }
    return ans;
  }
}
