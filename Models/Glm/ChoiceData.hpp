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

#ifndef BOOM_CHOICE_DATA_HPP
#define BOOM_CHOICE_DATA_HPP

#include "LinAlg/Selector.hpp"
#include "Models/CategoricalData.hpp"
#include "Models/DataTypes.hpp"

namespace BOOM {

  class ChoiceData : public CategoricalData {
   public:
    // Args:
    //   y:  The response
    //   subject_x: A vector of predictors describing subject level
    //     characteristics.  Should contain an explicit intercept if
    //     an intercept term is desired for the model.  If the vector
    //     is empty then no subject-level predictors will be
    //     considered.
    //   choice_x: A vector of vectors describing characteristics of
    //     the object being chosen.  May be empty if all data is
    //     subject-level data.
    ChoiceData(const CategoricalData &y, const Ptr<VectorData> &subject_x,
               const std::vector<Ptr<VectorData> > &choice_x);

    ChoiceData(const ChoiceData &rhs);

    ChoiceData *clone() const override;

    //--------- virtual function over-rides ----
    std::ostream &display(std::ostream &) const override;

    //--------- choice information ----
    uint nchoices() const;     // number of possible choices
    uint n_avail() const;      // number of choices actually available
    bool avail(uint i) const;  // is choice i available to the subject?

    uint subject_nvars() const;
    uint choice_nvars() const;

    const uint &value() const override;
    void set_y(uint y);

    // The Vector of subject-level predictor variables.
    const Vector &Xsubject() const;

    // The Vector of choice-level predictor variables for choice i.
    const Vector &Xchoice(uint i) const;

    // Fills the matrix given in the first argument with the "design
    // matrix" corresponding to this observation.  Let Sx denote the
    // vector of subject level predictors, and Cx the vector of choice
    // level predictors.  The matrix X is resized and over-written with
    //
    // Sx 0 0 0 ... 0 Cx
    // 0 Sx 0 0 ... 0 Cx
    // 0 0 Sx 0 ... 0 Cx
    // ...
    //
    // The matrix has Nchoices() rows.  If include_zero is true then
    // it has (Nchoices() * subject_nvars()) + choice_nvars() columns.
    // If include_zero is false then the first subject_nvars() columns
    // are omitted (so the first row is zero until you get to Cx).
    //
    // The return value is the final value of the first argument,
    // after it is modified.
    const Matrix &write_x(Matrix &X, bool include_zero) const;

    // Returns the matrix written by write_x().  If this function is
    // never called then the space for X is never allocated.  The
    // first time this function is called space for X is allocated and
    // the correct values are stored, so it is faster to call this
    // function many times than to call write_x many times.  If memory
    // becomes an issue, then it may be prefereable to call write_x
    // instead.
    const Matrix &X(bool include_zero = true) const;

    void set_Xsubject(const Vector &x);
    void set_Xchoice(const Vector &x, uint i);

   private:
    Ptr<VectorData> xsubject_;               // age of car buyer
    std::vector<Ptr<VectorData> > xchoice_;  // price of car

    Selector avail_;  // which choices are available
    Vector null_;     // zero length.  return for null reference.

    // All subject and choice predictors stretched out into a single
    // predictor vector
    mutable Matrix bigX_;
    mutable bool big_x_current_;

    bool check_big_x(bool include_zeros) const;
  };

}  // namespace BOOM
#endif  // BOOM_CHOICE_DATA_HPP
