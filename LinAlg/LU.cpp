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

#include <LinAlg/Vector.hpp>
#include <LinAlg/LU.hpp>
#include <cpputil/report_error.hpp>

extern "C"{
  void dgetrf_(int *, int *, double *, int *, int *, int *);
  void dgetrs_(const char *, int *, const double *, int *, int *,
               const int *, double *, int *, int *);
}


namespace BOOM{
  LU::LU(const Matrix &mat)
    : dcmp(mat),
      pivots(std::min(mat.nrow(), mat.ncol())),
      sing_(false)
  {
    int m = mat.nrow();
    int n = mat.ncol();
    int info;
    dgetrf_(&m, &n, dcmp.data(), &m, &pivots[0], &info);
    if(info!=0) sing_ = true;
  }

  //----------------------------------------------------------------------

  Matrix LU::getL()const{
    uint n = dcmp.nrow();
    Matrix ans(n,n, 0.0);
    for(uint i=0; i<n; ++i){
      for(uint j=0; j<i; ++j){
        ans(i,j) = dcmp(i,j);}
      ans(i,i) = 1.0;}
    return ans; }

  Matrix LU::getU()const{
    uint m = dcmp.ncol();
    Matrix ans(m,m, 0.0);
    for(uint i=0; i<m; ++i){
      for(uint j=i; j<m; ++j){
        ans(i,j) = dcmp(i,j);}}
    return ans;}

  std::vector<int> LU::get_pivots()const{
    return pivots;}

  double LU::det()const{
    if(dcmp.nrow()!=dcmp.ncol()) {
      report_error("LU:: determinant of non-square Matrix attempted.");
    }
    double ans = 1.0;  // not right!  might be -1... how to tell from pivots?
    int sign = 1;
    for(uint i=0; i<dcmp.nrow(); ++i){
      if(pivots[i]!=i+1) sign = -sign;
      ans*= dcmp.unchecked(i,i);
    }
    return sign*ans;
  }


  Matrix LU::solve(const Matrix &B)const{
    assert(dcmp.nrow()==dcmp.ncol());
    Matrix ans(B);
    int n = dcmp.ncol();
    int ncol_b = ans.ncol();
    int info=0;
    dgetrs_("N", &n, dcmp.data(), &n, &ncol_b, &pivots[0], ans.data(),
            &n, &info);
    if(info!=0) {
      report_error("LU::solve illegal argument to dgetrs_");
    }
    return ans;
  }

  Matrix LU::solveT(const Matrix &B)const{
    assert(dcmp.nrow()==dcmp.ncol());
    Matrix ans(B);
    int n = dcmp.ncol();
    int ncol_b = ans.ncol();
    int info=0;
    dgetrs_("T", &n, dcmp.data(), &n, &ncol_b, &pivots[0], ans.data(), &n,
            &info);
    if(info!=0) {
      report_error("LU::solveT illegal argument to dgetrs_");
    }
    return ans;
  }

  Vector LU::solve(const Vector &B)const{
    assert(dcmp.nrow()==dcmp.ncol());
    Vector ans(B);
    int n = dcmp.ncol();
    int ncol_b = 1;
    int info=0;
    dgetrs_("N", &n, dcmp.data(), &n, &ncol_b, &pivots[0], ans.data(), &n,
            &info);
    if(info!=0) {
      report_error("LU::solve illegal argument to dgetrs_");
    }
    return ans;
  }

  Vector LU::solveT(const Vector &B)const{
    assert(dcmp.nrow()==dcmp.ncol());
    Vector ans(B);
    int n = dcmp.ncol();
    int ncol_b = 1;
    int info=0;
    dgetrs_("T", &n, dcmp.data(), &n, &ncol_b, &pivots[0], ans.data(), &n,
            &info);
    if(info!=0) {
      report_error("LU::solveT illegal argument to dgetrs_");
    }
    return ans;
  }

}
