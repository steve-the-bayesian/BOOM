// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include "Models/Glm/BinomialLogitModel.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  typedef BinomialRegressionData BRD;
  BRD::BinomialRegressionData(double y, double n, const Vector &x)
      : GlmData<DoubleData>(y, x), n_(n) {
    check();
  }

  BRD::BinomialRegressionData(double y, double n, const Ptr<VectorData> &x)
      : GlmData<DoubleData>(Ptr<DoubleData>(new DoubleData(y)), x), n_(n) {
    check();
  }

  BRD *BRD::clone() const { return new BRD(*this); }

  void BRD::set_n(double n, bool check_n) {
    n_ = n;
    if (check_n) check();
  }

  void BRD::set_y(double y, bool check_n) {
    GlmData<DoubleData>::set_y(y);
    if (check_n) check();
  }

  void BRD::increment(double incremental_y, double incremental_n) {
    if (incremental_y > incremental_n) {
      report_error("Incremental successes cannot exceed incremental trials.");
    }
    if (incremental_y < 0 || incremental_n < 0) {
      report_error("Both incremental successes and incremental trials must "
                   "be non-negative.");
    }
    n_ += incremental_n;
    GlmData<DoubleData>::set_y(incremental_y + y());
  }
  
  double BRD::n() const { return n_; }

  void BRD::check() const {
    if (n_ < y() || n_ < 0 || y() < 0) {
      ostringstream err;
      err << "error in BinomialRegressionData:  n < y" << endl
          << "  n = " << n_ << endl
          << "  y = " << y() << endl;
      report_error(err.str());
    }
  }

  std::ostream &BRD::display(std::ostream &out) const {
    out << n_ << " ";
    return GlmData<DoubleData>::display(out);
  }

}  // namespace BOOM
