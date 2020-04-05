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
#ifndef BOOM_SPD_STORAGE_HPP
#define BOOM_SPD_STORAGE_HPP

#include <functional>
#include "Models/DataTypes.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Cholesky.hpp"

namespace BOOM {
  // An SpdMatrix matrix (though of as a variance matrix), its inverse (ivar),
  // and the lower Cholesky triangles of matrix and its inverse.
  class SpdData : virtual public Data {
   public:
    explicit SpdData(uint n, double diag = 1.0, bool ivar = false);
    explicit SpdData(const SpdMatrix &S, bool ivar = false);
    SpdData(const SpdData &rhs) = default;
    SpdData(SpdData &&rhs) = default;
    SpdData *clone() const override;

    // The number of elements in the matrix.
    // Args:
    //   minimal: If true then only the elements in the diagonal and the upper
    //     triangle are counted.  Otherwise all elements (including elements
    //     duplicated by symmetry) are counted.
    virtual uint size(bool minimal = true) const;

    // The number of rows in the matrix (which is the same as the number of
    // columns).
    uint dim() const;

    // Show the variance matrix.
    std::ostream &display(std::ostream &out) const override;

    const SpdMatrix &value() const {return var();}
    void set(const SpdMatrix &var, bool signal = true) {
      set_var(var, signal);
    }

    const SpdMatrix &var() const;
    const SpdMatrix &ivar() const;
    Matrix var_chol() const;
    Matrix ivar_chol() const;

    // The log determinant of sigma-inverse.  This name was chosen because it
    // appears as an argument in the multivariate normal probability
    // distribution functions in the distributions library.
    double ldsi() const;

    void set_var(const SpdMatrix &, bool signal = true);
    void set_ivar(const SpdMatrix &, bool signal = true);
    void set_var_chol(const Matrix &L, bool signal = true);
    void set_ivar_chol(const Matrix &L, bool signal = true);

   private:
    // Report an error message stating that nothing is current.
    void nothing_current() const;

    void ensure_var_current() const;
    void ensure_ivar_current() const;
    void ensure_var_chol_current() const;
    void ensure_ivar_chol_current() const;

    // Check that the desired representation contains the desired
    // data.  If not then find and refresh from the current
    // representation.
    mutable SpdMatrix var_;
    mutable SpdMatrix ivar_;
    mutable Cholesky ivar_chol_;
    mutable Cholesky var_chol_;

    mutable bool var_current_;
    mutable bool ivar_current_;
    mutable bool var_chol_current_;
    mutable bool ivar_chol_current_;
  };
}  // namespace BOOM
#endif  // BOOM_SPD_STORAGE_HPP
