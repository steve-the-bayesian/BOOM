/*
  Copyright (C) 2005-2023 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/PositiveSemidefiniteData.hpp"
#include "LinAlg/Eigen.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  PositiveSemidefiniteData::PositiveSemidefiniteData(
      const SpdMatrix &S)
      : value_(S)
  {
    update();
  }

  PositiveSemidefiniteData::PositiveSemidefiniteData(
      const PositiveSemidefiniteData &rhs)
      : Data(rhs),
        value_(rhs.value_),
        root_(rhs.root_),
        generalized_inverse_(rhs.generalized_inverse_),
        ldsi_(rhs.ldsi_)
  {}

  PositiveSemidefiniteData &PositiveSemidefiniteData::operator=(
      const PositiveSemidefiniteData &rhs) {
    if (&rhs != this) {
      value_ = rhs.value_;
      root_ = rhs.root_;
      generalized_inverse_ = rhs.generalized_inverse_;
      ldsi_ = rhs.ldsi_;
    }
    return *this;
  }

  PositiveSemidefiniteData::PositiveSemidefiniteData(
      PositiveSemidefiniteData &&rhs)
      : Data(std::move(rhs)),
        value_(std::move(rhs.value_)),
        root_(std::move(rhs.root_)),
        generalized_inverse_(std::move(rhs.generalized_inverse_)),
        ldsi_(rhs.ldsi_)
  {}

  PositiveSemidefiniteData &PositiveSemidefiniteData::operator=(
      PositiveSemidefiniteData &&rhs) {
    if (&rhs != this) {
      value_ = std::move(rhs.value_);
      root_ = std::move(rhs.root_);
      generalized_inverse_ = std::move(rhs.generalized_inverse_);
      ldsi_ = rhs.ldsi_;
    }
    return *this;
  }

  PositiveSemidefiniteData * PositiveSemidefiniteData::clone() const {
    return new PositiveSemidefiniteData(*this);
  }

  uint PositiveSemidefiniteData::size(bool minimal) const {
    size_t nrow = value_.nrow();
    return minimal ? nrow * (nrow + 1) / 2 : nrow * nrow;
  }

  ostream &PositiveSemidefiniteData::display(std::ostream &out) const {
    out << value_;
    return out;
  }

  void PositiveSemidefiniteData::set(const SpdMatrix &value, bool signal) {
    value_ = value;
    update();
    if (signal) {
      Data::signal();
    }
  }

  void PositiveSemidefiniteData::update() {
    SymmetricEigen eigen(value_, true);

    root_ = eigen.eigenvectors();
    const Vector &eigenvalues(eigen.eigenvalues());
    for (int i = 0; i < dim(); ++i) {
      double v = eigenvalues[i];
      if (v < 0) {
        if (fabs(v) > 1e-8) {
          std::ostringstream err;
          Vector values = eigenvalues;
          err << "A significant positive eigenvalue was found in what "
              "was supposed to be a positive semidefinite matrix.\n"
              << values.sort() << "\n";
          report_error(err.str());
        } else {
          v = 0.0;
        }
      }
      root_.col(i) *= sqrt(v);
    }
    generalized_inverse_ = eigen.generalized_inverse(1e-8);
    ldsi_ = eigen.generalized_inverse_logdet();
  }

  //
  PositiveSemidefiniteParams::PositiveSemidefiniteParams(const SpdMatrix &S)
      : PositiveSemidefiniteData(S)
  {}

  PositiveSemidefiniteParams::PositiveSemidefiniteParams(
      const PositiveSemidefiniteParams &rhs)
      : Params(rhs),
        PositiveSemidefiniteData(rhs)
  {}

  PositiveSemidefiniteParams &PositiveSemidefiniteParams::operator=(
      const PositiveSemidefiniteParams &rhs) {
    if (&rhs != this) {
      PositiveSemidefiniteData::operator=(rhs);
    }
    return *this;
  }

  PositiveSemidefiniteParams::PositiveSemidefiniteParams(
      PositiveSemidefiniteParams &&rhs)
      : Params(std::move(rhs)),
        PositiveSemidefiniteData(std::move(rhs))
  {}

  PositiveSemidefiniteParams &PositiveSemidefiniteParams::operator=(
      PositiveSemidefiniteParams &&rhs) {
    if (&rhs != this) {
      PositiveSemidefiniteData::operator=(rhs);
    }
    return *this;
  }

  PositiveSemidefiniteParams *PositiveSemidefiniteParams::clone() const {
    return new PositiveSemidefiniteParams(*this);
  }

  uint PositiveSemidefiniteParams::size(bool minimal) const {
    return nrow() * (minimal ? (nrow() + 1) / 2 : nrow());
  }

  Vector PositiveSemidefiniteParams::vectorize(bool minimal) const {
    return value().vectorize(minimal);
  }

  Vector::const_iterator PositiveSemidefiniteParams::unvectorize(
      Vector::const_iterator &v, bool minimal) {
    SpdMatrix value(nrow());
    Vector::const_iterator ans = value.unvectorize(v, minimal);
    set(value);
    return ans;
  }

}  // namespace BOOM
