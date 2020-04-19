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
#include "stats/NaturalSpline.hpp"
#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include "uint.hpp"
#include "LinAlg/Matrix.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  typedef NaturalSpline NS;

  QR NS::make_qr_const(double lo, double hi) const {
    Vector tmplo = basis(lo, 2);
    Vector tmphi = basis(hi, 2);
    Matrix tmp = rbind(tmplo, tmphi);
    QR qr(tmp.transpose());
    return qr;
  }

  NS::NaturalSpline(double lo, double hi, uint nIntKnots, bool intercept)
      : curs(0),
        boundary(false),
        knots_(nIntKnots),
        rdel(3),
        ldel(3),
        a(4),
        offsets(0),
        wsp(order_),
        icpt(intercept) {
    if (nIntKnots <= 0) too_few_knots();
    if (hi < lo) std::swap(hi, lo);
    double dx = (hi - lo) / (1 + nIntKnots);
    knots_[0] = lo + dx;
    for (uint i = 1; i < nIntKnots; ++i) knots_[i] = knots_[i - 1] + dx;
    for (int j = 0; j < order_; ++j) {
      knots_.push_back(lo);
      knots_.push_back(hi);
    }
    std::sort(knots_.begin(), knots_.end());

    basis_left = basis(lo);
    deriv_left = basis(lo, 1);
    basis_right = basis(hi);
    deriv_right = basis(hi, 1);

    qr_const = make_qr_const(lo, hi);
  }

  void NS::too_few_knots() const {
    std::string msg = "you must have at least one knot to use a NaturalSpline";
    report_error(msg);
  }

  NS::NaturalSpline(const Vector &Knots, double lo, double hi, bool intercept)
      : curs(0),
        boundary(false),
        knots_(Knots),
        rdel(3),
        ldel(3),
        a(4),
        offsets(0),
        wsp(order_),
        icpt(intercept) {
    if (nknots() <= 0) too_few_knots();
    std::sort(knots_.begin(), knots_.end());
    if (lo > knots_.front() || hi < knots_.back()) {
      std::ostringstream err;
      err << "in NaturalSpline constructor:" << endl
          << "boundary knots must be outside interior knots:" << endl
          << "you supplied: " << endl
          << "[" << lo << "] " << knots_ << " [" << hi << "]" << endl;
      report_error(err.str());
    }
    if (lo > hi) std::swap(hi, lo);
    for (int j = 0; j < order_; ++j) {
      knots_.push_back(lo);
      knots_.push_back(hi);
    }
    std::sort(knots_.begin(), knots_.end());

    basis_left = basis(lo);
    deriv_left = basis(lo, 1);
    basis_right = basis(hi);
    deriv_right = basis(hi, 1);

    qr_const = make_qr_const(lo, hi);
  }

  //======================================================================
  Vector NS::operator()(double x) const {
    Vector ans(order_);
    if (x < knots_.front()) {
      double dx = x - knots_.front();
      ans = basis_left + dx * deriv_left;
    } else if (x > knots_.back()) {
      double dx = x - knots_.back();
      ans = basis_right + dx * deriv_right;
    } else {
      ans = basis(x);
    }

    Vector tmp(qr_const.Qty(ans));
    Vector final_ans(tmp.begin() + 2, tmp.end());
    return final_ans;
  }

  //======================================================================

  int NS::nknots() const { return static_cast<int>(knots_.size()); }

  Vector NS::knots() const { return knots_; }

  //======================================================================

  void NS::set_cursor(double x) const {
    // do not assume xs are sorted
    curs = -1;  // Wall
    boundary = false;
    for (int i = 0; i < nknots(); ++i) {
      if (knots_[i] >= x) curs = i;
      if (knots_[i] > x) break;
    }

    if (curs > nknots() - order_) {
      int lastLegit = nknots() - order_;
      if (x == knots_[lastLegit]) {
        boundary = true;
        curs = lastLegit;
      }
    }
  }

  //======================================================================

  bool NS::in_outer_knots(double x) const {
    if (x >= knots_[0] && x < knots_[order_]) return true;
    if (x >= knots_[nknots() - ordm1_ - 1] && x <= knots_.back()) return true;
    return false;
  }

  //======================================================================

  Vector NS::basis(double x, uint nder) const {
    if (x < knots_.front() || x > knots_.back()) return basis_exterior(x, nder);
    return basis_interior(x, nder);
  }

  //======================================================================
  uint NS::basis_dim() const {
    uint ans = nknots() - order_;
    if (!icpt) --ans;
    return ans - 2;
  }

  Vector NS::basis_exterior(double, uint) const {
    return Vector(basis_dim(), 0.0);
  }

  //======================================================================

  Vector NS::basis_interior(double x, uint nd) const {
    int nk = nknots();
    assert(nk >= order_);
    uint sz = nk - order_;
    Vector ans(sz, 0.0);
    minimal_basis(x, nd);
    std::copy(wsp.begin(), wsp.end(), ans.begin() + offsets);
    if (icpt) return ans;
    return Vector(ans.begin() + 1, ans.end());
  }

  //======================================================================

  const Vector &NS::minimal_basis(double x, uint nd) const {
    set_cursor(x);
    int j = curs - order_;
    offsets = j;
    int nk = nknots();
    if (j < 0 || j > nk) {
      std::ostringstream err;
      err << "a bad bad thing happened in NS::minimal_basis()" << endl
          << " you can't have x inside the left or right " << order_
          << " knots." << endl
          << "x = " << x << endl;
      report_error(err.str());
    }
    if (nd > 0) {
      for (int ii = 0; ii < order_; ++ii) {
        a = 0;
        a[ii] = 1;
        wsp[ii] = eval_derivs(x, nd);
      }
    } else {
      basis_funcs(x, wsp);
    }
    return wsp;
  }
  //======================================================================

  void NS::diff_table(double x, int ndiff) const {
    for (int i = 0; i < ndiff; i++) {
      rdel[i] = knots_[curs + i] - x;
      ldel[i] = x - knots_[curs - (i + 1)];
    }
  }

  //======================================================================

  /* fast evaluation of basis functions */
  void NS::basis_funcs(double x, Vector &b) const {
    diff_table(x, ordm1_);
    b[0] = 1.;
    for (int j = 1; j <= ordm1_; j++) {
      double saved = 0.;
      for (int r = 0; r < j; r++) {
        double term = b[r] / (rdel[r] + ldel[j - 1 - r]);
        b[r] = saved + rdel[r] * term;
        saved = ldel[j - 1 - r] * term;
      }
      b[j] = saved;
    }
  }
  //======================================================================

  double NS::predict(double x, const Vector &beta) const {
    set_cursor(x);
    double ans(0);
    if (curs < order_ || curs > (nknots() - order_)) {
      report_error("a bad bad thing happened in NaturalSpline::predict");
    } else {
      memcpy(a.data(), beta.data() + curs - order_, order_);
      ans = eval_derivs(x, 0);
    }
    return ans;
  }

  //======================================================================

  double NS::eval_derivs(double x, int nder) const {
    const double *lpt, *rpt, *ti = knots_.data() + curs;
    double *apt;
    int inner, outer = ordm1_;

    if (boundary && nder == ordm1_) { /* value is arbitrary */
      return 0.0;
    }
    while (nder--) {
      for (inner = outer, apt = a.data(), lpt = ti - outer; inner--;
           apt++, lpt++)
        *apt = outer * (*(apt + 1) - *apt) / (*(lpt + outer) - *lpt);
      outer--;
    }
    diff_table(x, outer);
    while (outer--)
      for (apt = a.data(), lpt = ldel.data() + outer, rpt = rdel.data(),
          inner = outer + 1;
           inner--; lpt--, rpt++, apt++)
        *apt = (*(apt + 1) * *lpt + *apt * *rpt) / (*rpt + *lpt);
    return a[0];
  }

}  // namespace BOOM
