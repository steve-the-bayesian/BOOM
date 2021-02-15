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

#include "stats/IQagent.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

using std::vector;
typedef vector<double>::iterator IT;
typedef vector<double>::const_iterator CIT;

namespace BOOM {

  IQagent::IQagent(uint Bufsize)
      : max_buffer_size_(Bufsize), nobs_(0), ecdf_(Vector(1, 0.0)) {
    set_default_probs();
    quantiles_.resize(probs_.size());
  }
  //-----------------------------------------------------------------------
  IQagent::IQagent(const Vector& probs, uint Bufsize)
      : max_buffer_size_(Bufsize), nobs_(0), probs_(probs) {
    std::sort(probs_.begin(), probs_.end());
    quantiles_.resize(probs_.size());
  }
  //-----------------------------------------------------------------------
  IQagent::IQagent(const IqAgentState &state) {
    restore_from_state(state);
  }
  //-----------------------------------------------------------------------
  void IQagent::set_default_probs() {
    probs_.clear();
    //    for(double x = .01; x<=.99; x+=.005) probs_.push_back(x);
    probs_.reserve(11);
    probs_.push_back(.01);
    probs_.push_back(.025);
    probs_.push_back(.05);
    probs_.push_back(.10);
    probs_.push_back(.25);
    probs_.push_back(.5);
    probs_.push_back(.75);
    probs_.push_back(.9);
    probs_.push_back(.95);
    probs_.push_back(.975);
    probs_.push_back(.99);
  }
  //----------------------------------------------------------------------
  void IQagent::add(double x) {
    data_buffer_.push_back(x);
    if (data_buffer_.size() > max_buffer_size_) {
      update_cdf();
    }
  }
  //----------------------------------------------------------------------
  void IQagent::add(const Vector &x) {
    std::copy(x.begin(), x.end(), std::back_inserter(data_buffer_));
    if (data_buffer_.size() > max_buffer_size_) {
      update_cdf();
    }
  }
  //----------------------------------------------------------------------
  inline double interp_q(double p, double p0, double p1, double q0, double q1) {
    double slope = (q1 - q0) / (p1 - p0);
    return q0 + (p - p0) * slope;
  }
  //----------------------------------------------------------------------
  double IQagent::cdf(double x) const { return Fq(x); }
  //----------------------------------------------------------------------
  double IQagent::quantile(double prob) const {
    //    if(data_buffer_.size()>0) update_cdf();
    CIT b = probs_.begin();
    CIT e = probs_.end();
    CIT lower = lower_bound(b, e, prob);
    if (lower == probs_.end()) return quantiles_.back();
    CIT upper = upper_bound(b, e, prob);

    uint lo = lower - b;
    uint hi = upper - b;
    if (lo == hi) return quantiles_[lo];
    return interp_q(prob, probs_[lo], probs_[hi], quantiles_[lo],
                    quantiles_[hi]);
  }
  //----------------------------------------------------------------------
  inline double interp(double x, double x0, double x1, double p0, double p1) {
    assert(x <= x1 && x >= x0);
    if (x0 == x1) return p0;
    return p0 + (p1 - p0) * (x - x0) / (x1 - x0);
  }
  //----------------------------------------------------------------------
  inline double pm_med(double pm, double T) {
    // An implementation detail of the interpolation scheme between successive
    // quantile points.
    //
    // Args:
    //   pm: One of the stored probability values.  The probability label for
    //     quantiles_[m].
    //   T: A sample size.
    //
    // Returns:
    //   The median of pm, 1/(2T) and 1 - 1/(2T).  The latter two are the lower
    //   and upper extremes of a data set with 2T entries.
    if (T < 1) return pm;
    double lo = .5 / T;
    double hi = 1 - .5 / T;
    assert(hi > lo);
    if (lo > pm)
      return lo;  // pm lo hi
    else if (hi < pm)
      return hi;  // lo hi pm
    return pm;    // lo pm hi
  }

  double IQagent::Fq(double x) const {
    if (x < quantiles_[0]) return 0;
    if (x >= quantiles_.back()) return 1;
    CIT it = std::upper_bound(quantiles_.begin(), quantiles_.end(), x);
    uint pos = it - quantiles_.begin();
    double T = nobs_;
    double pm = pm_med(probs_[pos - 1], T);
    double pmp1 = pm_med(probs_[pos], T);
    return interp(x, quantiles_[pos - 1], quantiles_[pos], pm, pmp1);
  }
  //----------------------------------------------------------------------
  double IQagent::F(double x, bool plus) const {
    double T = nobs_;
    double N = ecdf_.sorted_data().size();
    return (T * Fq(x) + N * ecdf_(x, plus)) / (T + N);
  }
  //----------------------------------------------------------------------
  double IQagent::find_xplus(double p) const {
    uint n = Fplus_.size();
    uint i = 0;
    while (i < n && Fplus_[i] < p) ++i;
    if (i == n) {
      // report_error("find_xplus failed in IQagent");
      i = n - 1;  /////// troubling???
    }
    return data_buffer_[i];
  }
  //----------------------------------------------------------------------
  double IQagent::find_xminus(double p) const {
    uint n = Fminus_.size();
    uint i = n - 1;
    while (Fminus_[i] > p) {
      if (i == 0) {
        //      report_error("find_xminus failed in IQagent");
        break;
      }
      --i;
    }
    return data_buffer_[i];
  }
  //----------------------------------------------------------------------
  void IQagent::update_cdf() {
    if (data_buffer_.empty()) return;
    ecdf_ = ECDF(data_buffer_);
    const Vector &sorted_data(ecdf_.sorted_data());
    data_buffer_.reserve(sorted_data.size() + quantiles_.size());
    data_buffer_.clear();
    std::merge(sorted_data.begin(), sorted_data.end(), quantiles_.begin(),
               quantiles_.end(), back_inserter(data_buffer_));
    // now data buffer includes quantiles and new data
    uint n = data_buffer_.size();
    Fplus_.resize(n);
    Fminus_.resize(n);
    for (uint i = 0; i < n; ++i) {
      double x = data_buffer_[i];
      Fplus_[i] = F(x, true);
      Fminus_[i] = F(x, false);
    }

    n = probs_.size();
    for (uint i = 0; i < n; ++i) {
      double xplus = find_xplus(probs_[i]);
      double xminus = find_xminus(probs_[i]);
      if (xplus == xminus)
        quantiles_[i] = xminus;
      else {
        double Fplus = F(xplus, true);
        double Fminus = F(xminus, false);
        if (Fplus == Fminus)
          quantiles_[i] = xminus;
        else {
          double rho = (Fplus - probs_[i]) / (Fplus - Fminus);
          assert(std::isfinite(rho));
          quantiles_[i] = rho * xminus + (1 - rho) * xplus;
        }
      }
    }

    // clean up
    nobs_ += sorted_data.size();
    data_buffer_.clear();
  }

  IqAgentState IQagent::save_state() const {
    IqAgentState ans;
    ans.max_buffer_size = max_buffer_size_;
    ans.nobs = nobs_;
    ans.data_buffer = data_buffer_;
    ans.probs = probs_;
    ans.quantiles = quantiles_;
    ans.ecdf_sorted_data = ecdf_.sorted_data();
    ans.fplus = Fplus_;
    ans.fminus = Fminus_;
    return ans;
  };

  void IQagent::restore_from_state(const IqAgentState &state) {
    max_buffer_size_ = state.max_buffer_size;
    nobs_ = state.nobs;
    data_buffer_ = std::move(state.data_buffer);
    probs_ = std::move(state.probs);
    quantiles_ = std::move(state.quantiles);
    ecdf_.restore(state.ecdf_sorted_data);
    Fplus_ = state.fplus;
    Fminus_ = state.fminus;
  }

}  // namespace BOOM
