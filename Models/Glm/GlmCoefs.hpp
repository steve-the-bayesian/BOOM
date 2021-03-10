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
#ifndef BOOM_GLM_COEFS_HPP
#define BOOM_GLM_COEFS_HPP

#include "LinAlg/Selector.hpp"
#include "Models/ParamTypes.hpp"

namespace BOOM {
  class GlmCoefs : public VectorParams {
   public:
    explicit GlmCoefs(uint p, bool all = true);  // beta is 0..p

    // If infer_model_selection is true then zero-valued coefficients
    // will be marked as excluded from the model.
    explicit GlmCoefs(const Vector &b, bool infer_model_selection = false);
    explicit GlmCoefs(const Vector &b, const Selector &Inc);
    GlmCoefs(const GlmCoefs &rhs);
    GlmCoefs *clone() const override;

    //---     model selection  -----------
    const Selector &inc() const;
    bool inc(uint p) const;
    void set_inc(const Selector &);
    void add(uint i);
    void drop(uint i);
    void flip(uint i);
    void drop_all();
    void add_all();

    //---- size querries...
    // Args:
    //   minimal: If true, return the number of included coefficients.
    //     Otherwise return the number of available coefficients
    uint size(bool minimal = true) const override;

    uint nvars() const;
    uint nvars_possible() const;
    uint nvars_excluded() const;

    // GlmCoefs can call predict on a vector of dimension nvars (i.e. the set of
    // included variables) or nvars_possible (all variables).
    double predict(const Vector &x) const;
    double predict(const VectorView &x) const;
    double predict(const ConstVectorView &x) const;

    //
    Vector predict(const Matrix &design_matrix) const;
    void predict(const Matrix &design_matrix, Vector &result) const;
    void predict(const Matrix &design_matrix, VectorView result) const;

    //------ operations for only included variables --------
    Vector included_coefficients() const;

    // Set all coefficients to zero except the subset given by
    // 'nonzero_positions', which are set to 'nonzero_values'.  These two
    // vectors must be the same size.
    void set_sparse_coefficients(const Vector &nonzero_values,
                                 const std::vector<uint> &nonzero_positions);
    void set_sparse_coefficients(const Vector &nonzero_values,
                                 const std::vector<int> &nonzero_positions);

    // Set the included coefficients to b.  The dimension of b must match
    // nvars().
    void set_included_coefficients(const Vector &b);

    //----- operations for both included and excluded variables ----
    const Vector &Beta() const;  // reports 0 for excluded positions

    // Set the dense vector of coefficients to beta.  If any elements are
    // excluded, then those elements will be set to zero.
    void set_Beta(const Vector &beta);

    // Set the a subset of beta to the requested value.  If any elements of the
    // subset are excluded, those values will be set to zero, regardless of
    // their value in beta_subset.
    //
    // Args:
    //   beta_subset:  The vector of values to be assigned.
    //   start: The position in the dense vector "Beta" where the assignment
    //     should be made.
    //   signal:  Should observers be signalled about the assignment.
    void set_subset(const Vector &beta_subset, int start,
                    bool signal = true) override;

    double Beta(uint dense_index) const;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;

   private:
    Selector inc_;

    // included_coefficients_ is a view into the full coefficient vector.
    // included_coefficients_current_ is a flag indicating whether or not the
    // view needs to be refreshed.
    mutable Vector included_coefficients_;
    mutable bool included_coefficients_current_;

    void set_excluded_coefficients_to_zero();
    void inc_from_beta(const Vector &v);
    uint indx(uint i) const { return inc_.indx(i); }
    void wrong_size_beta(const Vector &b) const;
    void fill_beta() const;
    void setup_obs();

    double &operator[](uint i);
    double operator[](uint i) const;
  };

  //===========================================================================
  class MatrixGlmCoefs : public MatrixParams {
   public:
    // All coefficients included.  Initial value is zero.
    explicit MatrixGlmCoefs(int nrow, int ncol);

    // All coefficients are included.
    explicit MatrixGlmCoefs(const Matrix &coefficients);

    // Specify which coefficients are included.
    explicit MatrixGlmCoefs(const Matrix &coefficients,
                            const SelectorMatrix &included);

    MatrixGlmCoefs *clone() const override {return new MatrixGlmCoefs(*this);}

    int nrow() const {return value().nrow();}
    int ncol() const {return value().ncol();}

    // The rows of the coefficient matrix correspond to predictors.  The columns
    // correspond to different outcomes.
    int xdim() const {return nrow();}
    int ydim() const {return ncol();}

    Vector predict(const Vector &predictors) const;

    // Args:
    //   values: The full matrix of coefficients, including all 0's for excluded
    //     variables.  Nonzero values for coefficients that have been excluded
    //     will be
    void set(const Matrix &values, bool signal = true) override;

    void set_inclusion_pattern(const SelectorMatrix &included);

    void add_all() {included_.add_all();}
    void drop_all() {included_.drop_all();}
    void add(int i, int j) {included_.add(i, j);}
    void drop(int i, int j) {included_.drop(i, j);}
    void flip(int i, int j) {included_.flip(i, j);}

    const SelectorMatrix &included_coefficients() const { return included_; }

   private:
    // Keeps track of which coefficients are included (i.e. not forced to zero).
    SelectorMatrix included_;

    // Throws an error if the dimension of the selector matrix does not match
    // the dimension of the matrix parameter.
    void check_dimension(const SelectorMatrix &included) const;

    // Set all excluded coefficients to zero.
    void set_zeros();
  };

}  // namespace BOOM
#endif  // BOOM_GLM_COEFS_HPP
