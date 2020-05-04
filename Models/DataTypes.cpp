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

#include "Models/DataTypes.hpp"
#include <sstream>
#include <stdexcept>
#include <vector>
#include "cpputil/report_error.hpp"

namespace BOOM {

  std::ostream &operator<<(std::ostream &out, const Ptr<Data> &dp) {
    return dp->display(out);
  }

  std::ostream &operator<<(std::ostream &out, const Data &d) {
    d.display(out);
    return out;
  }

  void print_data(const Data &d) { std::cout << d << std::endl; }

  void intrusive_ptr_add_ref(Data *d) { d->up_count(); }
  void intrusive_ptr_release(Data *d) {
    d->down_count();
    if (d->ref_count() == 0) {
      delete d;
    }
  }

  Data::missing_status Data::missing() const { return missing_flag; }
  void Data::set_missing_status(missing_status m) { missing_flag = m; }

  //------------------------------------------------------------
  VectorData::VectorData(uint n, double X) : data_(n, X) {}
  VectorData::VectorData(const Vector &y) : data_(y) {}
  VectorData::VectorData(const VectorData &rhs)
      : Data(rhs), Traits(rhs), data_(rhs.data_) {}
  VectorData *VectorData::clone() const { return new VectorData(*this); }

  std::ostream &VectorData::display(std::ostream &out) const {
    out << data_;
    return out;
  }

  void VectorData::set(const Vector &rhs, bool signal_change) {
    data_ = rhs;
    if (signal_change) {
      signal();
    }
  }

  void VectorData::set_element(double value, int position, bool sig) {
    data_[position] = value;
    if (sig) {
      signal();
    }
  }

  void VectorData::set_subset(const Vector &subset, int start, bool signal) {
    VectorView view(data_, start, subset.size());
    view = subset;
    if (signal) {
      this->signal();
    }
  }

  double VectorData::operator[](uint i) const { return data_[i]; }

  double &VectorData::operator[](uint i) {
    signal();
    return data_[i];
  }
  //------------------------------------------------------------
  PartiallyObservedVectorData::PartiallyObservedVectorData(
      const Vector &y, const Selector &obs)
      : VectorData(y),
        obs_(obs) {
    if (obs.empty()) {
      obs_ = Selector(y.size(), true);
    }
    if (obs_.nvars() == obs_.nvars_possible()) {
      set_missing_status(observed);
    } else if (obs_.nvars() > 0) {
      set_missing_status(partly_missing);
    } else {
      set_missing_status(completely_missing);
    }
  }

  PartiallyObservedVectorData * PartiallyObservedVectorData::clone() const {
    return new PartiallyObservedVectorData(*this);
  }

  void PartiallyObservedVectorData::set(const Vector &rhs, bool signal_change) {
    if (rhs.size() != obs_.nvars_possible()) {
      report_error("Dimension changes are not possible with "
                   "PartiallyObservedVectorData");
    }
    VectorData::set(rhs, signal_change);
  }

  //------------------------------------------------------------
  MatrixData::MatrixData(int r, int c, double val) : x(r, c, val) {}

  MatrixData::MatrixData(const Matrix &y) : x(y) {}

  MatrixData::MatrixData(const MatrixData &rhs)
      : Data(rhs), Traits(rhs), x(rhs.x) {}

  MatrixData *MatrixData::clone() const { return new MatrixData(*this); }
  std::ostream &MatrixData::display(std::ostream &out) const {
    out << x << std::endl;
    return out;
  }

  void MatrixData::set(const Matrix &rhs, bool sig) {
    x = rhs;
    if (sig) {
      signal();
    }
  }

  void MatrixData::set_element(double value, int row, int col, bool sig) {
    x(row, col) = value;
    if (sig) {
      signal();
    }
  }

}  // namespace BOOM
