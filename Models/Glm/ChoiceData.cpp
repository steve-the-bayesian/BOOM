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
#include "Models/Glm/ChoiceData.hpp"
#include "LinAlg/Vector.hpp"

#include "cpputil/report_error.hpp"
#include <sstream>

namespace BOOM {

  typedef ChoiceData CHD;
  typedef VectorData VD;

  ChoiceData::ChoiceData(const CategoricalData &y,
                         const Ptr<VectorData> &subject_x,
                         const std::vector<Ptr<VectorData> > &choice_x)
      : CategoricalData(y),
        xsubject_(subject_x),
        xchoice_(choice_x),
        avail_(y.nlevels(), true),
        big_x_current_(false) {
    if (!subject_x) {
      xsubject_.reset(new VectorData(Vector(0)));
    }
  }

  CHD::ChoiceData(const CHD &rhs)
      : Data(rhs),
        CategoricalData(rhs),
        xsubject_(rhs.xsubject_->clone()),
        xchoice_(rhs.xchoice_.size()),
        avail_(rhs.avail_),
        bigX_(rhs.bigX_),
        big_x_current_(rhs.big_x_current_) {
    int n = rhs.xchoice_.size();
    for (int i = 0; i < n; ++i) {
      xchoice_[i] = rhs.xchoice_[i]->clone();
    }
  }

  CHD *CHD::clone() const { return new CHD(*this); }

  //======================================================================

  std::ostream &CHD::display(std::ostream &out) const {
    CategoricalData::display(out) << " " << *xsubject_ << " ";
    for (int i = 0; i < xchoice_.size(); ++i) out << Xchoice(i) << " ";
    return out;
  }

  int CHD::nchoices() const { return CategoricalData::nlevels(); }
  int CHD::n_avail() const { return avail_.nvars(); }
  bool CHD::avail(int i) const { return avail_[i]; }

  int CHD::subject_nvars() const { return xsubject_->dim(); }
  int CHD::choice_nvars() const {
    if (xchoice_.empty()) return 0;
    return xchoice_[0]->dim();
  }

  const uint &CHD::value() const { return CategoricalData::value(); }
  void CHD::set_y(int y) { CategoricalData::set(y); }

  const Vector &CHD::Xsubject() const { return xsubject_->value(); }

  const Vector &CHD::Xchoice(int i) const {
    if (!xchoice_.empty()) {
      return xchoice_[i]->value();
    } else
      return null_;
  }

  void CHD::set_Xsubject(const Vector &x) { xsubject_->set(x); }

  void CHD::set_Xchoice(const Vector &x, int i) { xchoice_[i]->set(x); }

  const Matrix &CHD::write_x(Matrix &X, bool inc_zero) const {
    bool inc = inc_zero;
    int pch = choice_nvars();
    int psub = subject_nvars();
    int M = nchoices();
    int nc = pch + (inc ? M : M - 1) * psub;
    X.resize(M, nc);
    X = 0;

    const Vector &xcu(Xsubject());
    for (int m = 0; m < M; ++m) {
      const Vector &xch(Xchoice(m));
      VectorViewIterator it = X.row_begin(m);
      if (inc || m > 0) {
        it += (inc ? m : m - 1) * psub;
        std::copy(xcu.begin(), xcu.end(), it);
      }
      it = X.row_begin(m) + (inc ? M : M - 1) * psub;
      std::copy(xch.begin(), xch.end(), it);
    }
    big_x_current_ = true;
    return X;
  }

  const Matrix &CHD::X(bool inc_zeros) const {
    if (!check_big_x(inc_zeros)) {
      write_x(bigX_, inc_zeros);
    }
    return bigX_;
  }

  bool CHD::check_big_x(bool include_zeros) const {
    if (!big_x_current_) return false;
    return bigX_.size() ==
           choice_nvars() + subject_nvars() * (nchoices() - 1 + include_zeros);
  }

  int ChoiceDataPredictorMap::long_subject_index(
      int subject_predictor,
      int choice_level) const {
    if (choice_level >= num_choices_ || choice_level < 0) {
      std::ostringstream err;
      err << "Choice level " << choice_level << " out of bounds.\n";
      report_error(err.str());
    }
    if (choice_level == 0 && !include_zeros_) {
      report_error("Choice level 0 was requested from a "
                   "ChoiceDataPredictorMap considering choice level 0 "
                   "to be implicit.");
    }
    return (choice_level - 1 + include_zeros_) * subject_dim_ + subject_predictor;
  }

  int ChoiceDataPredictorMap::long_choice_index(int choice_index) const {
    return (num_choices_ - 1 + include_zeros_) * subject_dim_ + choice_index;
  }

  std::pair<int, int> ChoiceDataPredictorMap::subject_index(
      int long_index) const {
    int choice = long_index / subject_dim_;
    if (!include_zeros_) {
      ++choice;
    }
    int index = long_index % subject_dim_;
    return std::pair<int, int>(index, choice);
  }

  int ChoiceDataPredictorMap::choice_index(int long_index) const {
    return long_index - (num_choices_ - 1 + include_zeros_) * subject_dim_;
  }

  bool ChoiceDataPredictorMap::is_choice(int long_index) const {
    return long_index >= (num_choices_ - 1 + include_zeros_) * subject_dim_;
  }

}  // namespace BOOM
