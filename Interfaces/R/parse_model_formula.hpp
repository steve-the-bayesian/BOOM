/*
  Copyright (C) 2005-2014 Steven L. Scott

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

#ifndef BOOM_PARSE_MODEL_FORMULA_HPP_
#define BOOM_PARSE_MODEL_FORMULA_HPP_

#include <string>
#include "stats/Design.hpp"
#include "r_interface/boom_r_tools.hpp"

namespace BOOM {
  namespace RInterface {

    // Use BOOM to construct a design matrix.  This is a limited
    // version of R's model.matrix but using BOOM to do the
    // processing.
    // Args:
    //   r_formula_rhs: The right hand side of an R model formula.
    //   r_data_frame_containing_only_factors: An R data frame where
    //     all variables are factors.
    //   add_intercept: If true an intercept term will be added to the
    //     returned RowBuilder object.  If false no intercept will be
    //     added.
    // Returns:
    //   The R design matrix corresponding to the inputs.
    SEXP BuildDesignMatrix(
        SEXP r_formula_rhs,
        SEXP r_data_frame_containing_only_factors,
        SEXP r_add_intercept);

    SEXP BuildContextualDesignMatrix(
        SEXP r_formula_rhs,
        SEXP r_experiment_data_only_factors,
        SEXP r_context_data_only_factors,
        SEXP r_add_intercept);

    // Args:
    //   r_formula_rhs: The right hand side of an R model formula.
    //   r_data_frame_containing_only_factors: An R data frame where
    //     all variables are factors.
    //   add_intercept: If true an intercept term will be added to the
    //     returned RowBuilder object.  If false no intercept will be
    //     added.
    //
    // Returns:
    //   A RowBuilder object that takes a row of a data.frame as input
    //   (as a vector of integers), and produces the appropriate row
    //   of a design matrix as output.
    BOOM::RowBuilder ParseModelFormulaRHS(
        SEXP r_formula_rhs,
        SEXP r_data_frame_containing_only_factors,
        bool add_intercept);

    // Args:
    //   r_formula_rhs: The right hand side of an R model formula.
    //   r_experimental_factors_data_frame: An R data frame containing
    //     the factors describing the levels of the experiment.
    //   r_contextual_factors_data_frame: An R data frame containing
    //     factors describing the context in which each observation is
    //     observed.
    //   add_intercept: If true an intercept term will be added to the
    //     returned RowBuilder object.  If false no intercept will be
    //     added.
    //
    // Returns:
    //   A ContextualRowBuilder object that takes a pair of rows of a
    //   data.frame as input (as a vector of integers), and produces
    //   the appropriate row of a design matrix as output.
    BOOM::ContextualRowBuilder ParseContextualModelFormulaRHS(
        SEXP r_formula_rhs,
        SEXP r_experimental_factors_data_frame,
        SEXP r_contextual_factors_data_frame,
        bool add_intercept);

    // Compute the dimension of the design matrix that would result
    // from this combination formula and data frame.
    //   r_formula_rhs: The right hand side of an R model formula.
    //   r_data_frame_containing_only_factors: An R data frame where
    //     all variables are factors.
    //   add_intercept: If true an intercept term will be added to the
    //     returned RowBuilder object.  If false no intercept will be
    //     added.
    inline int ComputeNumberOfDesignMatrixColumns(
        SEXP r_formula_rhs,
        SEXP r_data_frame_containing_only_factors,
        bool add_intercept) {
      return ParseModelFormulaRHS(r_formula_rhs,
                                  r_data_frame_containing_only_factors,
                                  add_intercept).dimension();
    }

  }  // namespace RInterface
}  // namespace BOOM

#endif  // BOOM_PARSE_MODEL_FORMULA_HPP_
