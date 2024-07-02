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

  // ChoiceData is a type of categorical data where a subject makes a choice
  // from a set of available options.  The elements of ChoiceData are
  //
  //   - The item (categorical level) that was chosen, which can be represented
  //     as an integer 0, ..., Nchoices - 1.
  //
  //   - The characteristics of the subject making the choice, which can be
  //     represented as a numeric vector (like the predictor vector in a
  //     logistic regression).
  //
  //   - The characteristics of the items in the choice set.  This can be
  //     thought of as a matrix of predictors, with rows corresponding to choice
  //     levels, and columns to characteristics.  More generally, this may be a
  //     collection of vectors, if some characteristics don't make sense for
  //     some choice levels.
  //
  // As an example, consider a person making a purchasing decision among several
  // types of cars.  Examples of subject level predictors are the person's age,
  // sex, and income.  Among the choice level predictors are the car's gas
  // mileage, horsepower, and number of cup holders.
  //
  // TODO(steve): One complicating factor in choice data is that some choice
  // occasions may offer a different choice set than others.  One way to resolve
  // this issue os to store the choice_x predictors in a map keyed by the
  // available choice levels.  This would involve updating the models that use
  // ChoiceData to account for assumptions around fixed numbers of choices.
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
    ChoiceData(const CategoricalData &y,
               const Ptr<VectorData> &subject_x,
               const std::vector<Ptr<VectorData>> &choice_x
               = std::vector<Ptr<VectorData>>());

    ChoiceData(const ChoiceData &rhs);

    ChoiceData *clone() const override;

    //--------- virtual function over-rides ----
    std::ostream &display(std::ostream &) const override;

    //--------- choice information ----
    int nchoices() const;     // number of possible choices
    int n_avail() const;      // number of choices actually available
    bool avail(int i) const;  // is choice i available to the subject?

    int subject_nvars() const;
    int choice_nvars() const;

    const uint &value() const override;
    void set_y(int y);

    // The Vector of subject-level predictor variables.
    const Vector &Xsubject() const;

    // The Vector of choice-level predictor variables for choice i.
    const Vector &Xchoice(int i) const;

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
    void set_Xchoice(const Vector &x, int i);

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

  //===========================================================================
  // A bivalent mapping between the "single-vector (expanded)" form of the
  // predictors in a choice data point and the "natural form" where data are
  // broken down into separate subject-level and choice-level predictors.
  //
  // In the "expanded" form, a predictor variable is the concatenation of
  //   *) Several 0-vectors masking the subject variables, with
  //   *) The subject-level predictors in a position corresponding to a given
  //      choice level, and
  //   *) The choice-level predictors (if any) bringing up the rear.
  //
  // In many settings (e.g. the typical handling of multinomial logit models)
  // the subject level predictors for choice level 0 are assigned 0 regression
  // coefficients.  In that case the leading 0's in the coefficient vector are
  // typically left implicit.
  class ChoiceDataPredictorMap {
   public:
    // Args:
    //   subject_dim:  The number of subject-level predictors.
    //   choice_dim: The dimension of the choice-level predictors (for a single
    //     value of the choice variable).
    //   num_choices:  The number of choice levels.
    //   include_zeros: If true, then explicitly create positions for the
    //     (normally implicit) set of zero coefficients describing choice level
    //     0.  If false then those variables are omitted.
    ChoiceDataPredictorMap(int subject_dim,
                           int choice_dim,
                           int num_choices,
                           bool include_zeros=false)
        : subject_dim_(subject_dim),
          choice_dim_(choice_dim),
          num_choices_(num_choices),
          include_zeros_(include_zeros)
    {}

    // The dimension of the expanded predictor.
    int xdim() const {
      return (num_choices_ - 1 + include_zeros_) * subject_dim_ + choice_dim_;
    }
    
    // Return the position in the expanded predictor vector of the subject-level
    // coefficient.
    //
    // Args:
    //   subject_predictor:  The index of the subject-level predictor variable.
    //   choice_level:  The
    int long_subject_index(int subject_predictor, int choice_level) const;

    // Return the index of a specific choice predictor in the expanded predictor
    // variable.
    //
    // Args:
    //   choice_index: The index of the choice variable (in the decomposed
    //     structure).
    //
    // Returns:
    //   The index of the same variable in the "one long vector" structure.
    int long_choice_index(int choice_index) const;

    // Given an index in the "expanded" form, return the (index, choice_level)
    // of the subject level predictors.
    //
    // Args: long_index: The index of a variable in the expanded form.  It is
    //   the caller's responsibility to ensure this index is less than the
    //   cutoff for choice indices.
    //
    // Returns:
    //   .first: The subject-level predictor index.
    //   .second: The choice level to which the predictor variable belongs.
    std::pair<int, int> subject_index(int long_index) const;

    // Given an index in the "expanded" form, return the choice-level predictor
    // index to which it corresponds.  It is the caller's responsibility to
    // ensure that 'long_index' is in the 'choice' portion of the index space.
    int choice_index(int long_index) const;

    // Returns true iff long_index points to a spot in the expanded predictor
    // vector corresponding to a choice variable.
    bool is_choice(int long_index) const;

   private:
    int subject_dim_;
    int choice_dim_;
    int num_choices_;
    bool include_zeros_;
  };

}  // namespace BOOM
#endif  // BOOM_CHOICE_DATA_HPP
