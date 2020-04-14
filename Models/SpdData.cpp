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

#include "Models/SpdData.hpp"
#include "LinAlg/Cholesky.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  SpdData::SpdData(uint n, double diag, bool ivar)
      : SpdData(SpdMatrix(n, diag), ivar) {}

  SpdData::SpdData(const SpdMatrix &S, bool ivar)
      : var_(ivar ? SpdMatrix(0) : S),
        ivar_(ivar ? S : SpdMatrix(0)),
        var_current_(!ivar),
        ivar_current_(ivar),
        var_chol_current_(false),
        ivar_chol_current_(false) {}

  SpdData *SpdData::clone() const { return new SpdData(*this); }

  uint SpdData::size(bool min) const {
    if (var_current_) {
      return var_.size();
    } else if (ivar_current_) {
      return ivar_.size();
    } else if (ivar_chol_current_) {
      int dim = ivar_chol_.dim();
      return dim * dim;
    } else if (var_chol_current_) {
      int dim = var_chol_.dim();
      return dim * dim;
    } else {
      nothing_current();
    }
    return 0;
  }

  uint SpdData::dim() const {
    if (var_current_) {
      return var_.nrow();
    } else if (ivar_current_) {
      return ivar_.nrow();
    } else if (ivar_chol_current_) {
      return ivar_chol_.nrow();
    } else if (var_chol_current_) {
      return var_chol_.nrow();
    } else {
      nothing_current();
    }
    return 0;
  }

  std::ostream &SpdData::display(std::ostream &out) const {
    out << var() << endl;
    return out;
  }

  const SpdMatrix &SpdData::var() const {
    ensure_var_current();
    return var_;
  }

  const SpdMatrix &SpdData::ivar() const {
    ensure_ivar_current();
    return ivar_;
  }

  Matrix SpdData::ivar_chol() const {
    ensure_ivar_chol_current();
    return ivar_chol_.getL();
  }

  Matrix SpdData::var_chol() const {
    ensure_var_chol_current();
    return var_chol_.getL();
  }

  double SpdData::ldsi() const {
    ensure_ivar_chol_current();
    return ivar_chol_.logdet();
  }

  void SpdData::set_var(const SpdMatrix &var, bool signal) {
    var_ = var;
    var_current_ = true;
    ivar_current_ = false;
    var_chol_current_ = false;
    ivar_chol_current_ = false;
    if (signal) {
      Data::signal();
    }
  }

  void SpdData::set_ivar(const SpdMatrix &ivar, bool signal) {
    ivar_ = ivar;
    ivar_current_ = true;
    var_current_ = false;
    var_chol_current_ = false;
    ivar_chol_current_ = false;
    if (signal) {
      Data::signal();
    }
  }

  void SpdData::set_ivar_chol(const Matrix &L, bool signal) {
    ivar_chol_.setL(L);
    ivar_chol_current_ = true;
    var_current_ = false;
    ivar_current_ = false;
    var_chol_current_ = false;
    if (signal) {
      Data::signal();
    }
  }

  void SpdData::set_var_chol(const Matrix &L, bool signal) {
    var_chol_.setL(L);
    var_chol_current_ = true;
    var_current_ = false;
    ivar_current_ = false;
    ivar_chol_current_ = false;
    if (signal) {
      Data::signal();
    }
  }

  void SpdData::nothing_current() const {
    report_error("Nothing is current in SpdData.  That should not happen.");
  }

  void SpdData::ensure_var_current() const {
    if (var_current_) {
      return;
    } else if (var_chol_current_) {
      var_ = var_chol_.original_matrix();
    } else if (ivar_chol_current_) {
      var_ = ivar_chol_.inv();
    } else if (ivar_current_) {
      ivar_chol_ = Cholesky(ivar_);
      ivar_chol_current_ = true;
      var_ = ivar_chol_.inv();
    } else {
      nothing_current();
    }
    var_current_ = true;
  }

  void SpdData::ensure_ivar_current() const {
    if (ivar_current_) {
      return;
    } else if (ivar_chol_current_) {
      ivar_ = ivar_chol_.original_matrix();
    } else if (var_chol_current_) {
      ivar_ = var_chol_.inv();
    } else if (var_current_) {
      var_chol_ = Cholesky(var_);
      var_chol_current_ = true;
      ivar_ = var_chol_.inv();
    } else {
      nothing_current();
    }
    ivar_current_ = true;
  }

  void SpdData::ensure_var_chol_current() const {
    if (var_chol_current_) {
      return;
    } else if (var_current_) {
      var_chol_ = Cholesky(var_);
    } else if (ivar_chol_current_) {
      var_ = ivar_chol_.inv();
      var_current_ = true;
      var_chol_ = Cholesky(var_);
    } else if (ivar_current_) {
      ivar_chol_ = Cholesky(ivar_);
      ivar_chol_current_ = true;
      var_ = ivar_chol_.inv();
      var_current_ = true;
      var_chol_ = Cholesky(var_);
    } else {
      nothing_current();
    }
    var_chol_current_ = true;
  }

  void SpdData::ensure_ivar_chol_current() const {
    if (ivar_chol_current_) {
      return;
    } else if (ivar_current_) {
      ivar_chol_ = Cholesky(ivar_);
    } else if (var_chol_current_) {
      ivar_ = var_chol_.inv();
      ivar_current_ = true;
      ivar_chol_ = Cholesky(ivar_);
    } else if (var_current_) {
      var_chol_ = Cholesky(var_);
      var_chol_current_ = true;
      ivar_ = var_chol_.inv();
      ivar_current_ = true;
      ivar_chol_ = Cholesky(ivar_);
    }
    ivar_chol_current_ = true;
  }

}  // namespace BOOM
