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

#include "Models/Glm/Glm.hpp"
#include "distributions.hpp"

#include <sstream>
#include "cpputil/report_error.hpp"

namespace BOOM {

  GlmBaseData::GlmBaseData(const Vector &x) : x_(new VectorData(x)) {}

  GlmBaseData::GlmBaseData(const Ptr<VectorData> &x) : x_(x) {}

  GlmBaseData::GlmBaseData(const GlmBaseData &rhs) : x_(rhs.x_->clone()) {}

  uint GlmBaseData::xdim() const { return x_->dim(); }

  const Vector &GlmBaseData::x() const { return x_->value(); }

  void GlmBaseData::set_x(const Vector &X, bool allow_any) {
    if (allow_any || x_->dim() == X.size()) {
      x_->set(X);
    } else {
      std::ostringstream err;
      err << "Vector sizes are incompatible in set_x." << endl
          << "New vector is " << X << endl
          << "Old vector is " << x() << endl;
      report_error(err.str());
    }
    signal();
  }

  GlmModel::GlmModel() {}
  GlmModel::GlmModel(const GlmModel &rhs) : Model(rhs) {}
  uint GlmModel::xdim() const { return coef().nvars_possible(); }
  void GlmModel::add(uint p) { coef().add(p); }
  void GlmModel::add_all() {
    for (int i = 0; i < xdim(); ++i) add(i);
  }
  void GlmModel::drop(uint p) { coef().drop(p); }
  void GlmModel::drop_all() {
    for (int i = 0; i < xdim(); ++i) drop(i);
  }
  void GlmModel::drop_all_but_intercept() {
    drop_all();
    add(0);
  }
  void GlmModel::flip(uint p) { coef().flip(p); }
  const Selector &GlmModel::inc() const { return coef().inc(); }
  bool GlmModel::inc(uint p) const { return coef().inc(p); }

  double GlmModel::predict(const Vector &x) const { return coef().predict(x); }
  double GlmModel::predict(const VectorView &x) const {
    return coef().predict(x);
  }
  double GlmModel::predict(const ConstVectorView &x) const {
    return coef().predict(x);
  }

  Vector GlmModel::included_coefficients() const {
    return coef().included_coefficients();
  }
  void GlmModel::set_included_coefficients(const Vector &b) {
    coef().set_included_coefficients(b);
  }

  // reports 0 for excluded positions
  const Vector &GlmModel::Beta() const { return coef().Beta(); }
  void GlmModel::set_Beta(const Vector &B) { coef().set_Beta(B); }
  double GlmModel::Beta(uint I) const { return coef().Beta(I); }

}  // namespace BOOM
