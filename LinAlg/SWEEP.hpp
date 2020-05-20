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

#ifndef BOOM_SWEEP_HPP
#define BOOM_SWEEP_HPP

#include <vector>
#include "uint.hpp"

#include "LinAlg/Matrix.hpp"
#include "LinAlg/Selector.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"

namespace BOOM {

  // A SweptVarianceMatrix is a matrix that has been operated on by
  // the SWEEP operator.  The SWEEP operator operates on multivariate
  // normal parameters (or sufficient statistics).  If Sigma is the
  // variance matrix of a multivariate normal, then SWP[k](Sigma)
  // moves element k from the unobserved, random 'Y' side of the
  // equation, to the observed, conditional, 'X' side.  The inverse
  // operation RSW[k](Sigma) moves it back.
  //
  // Suppose the matrix A is  A = (A_11  A_12)
  //                              (A_21  A_22)
  //
  // Then sweeping on the _1 elements of A (you only ever sweep on
  // diagonal elements) yields
  //    SWP[1](A) = ( -A_11^{-1}             A_11^{-1} * A_{12}              )
  //                ( A_{21} * A_{11}^{-1}   A_22 - A_{21} A_{11}^{-1} A_{12})
  //

  class SweptVarianceMatrix {
    // Sweeping a variable is equivalent to conditioning on it.
    // i.e. when a variable is swept it changes from 'y' to 'x'.
   public:
    // Args:
    //   m:  The matrix to be swept, or unswept.
    //   inverse: If true, then consider m to be a precision matrix,
    //     and consider all of its variables to be knowns.  Otherwise,
    //     consider m to be a variance matrix, and consider all its
    //     variables to be unknowns.
    explicit SweptVarianceMatrix(const SpdMatrix &m, bool inverse = false);

    // Sweep the given index, or set of indices into the "known" component.
    // Calling SWP on an already swept index is a no-op.
    void SWP(uint index_to_sweep);
    void SWP(const Selector &variables_to_sweep);

    // Sweep the given index from the "known" component back to the "unknown"
    // component.  Calling RSW on an unswept index is a no-op.
    void RSW(uint index_to_unsweep);

    // The matrix of regression coefficients for E(unknown | known).  The
    // dimension of Beta is unswept-rows by swept-columns, so it is to be called
    // as Beta() * known, not known * Beta().
    Matrix Beta() const;  // to compute E(unswept | swept)

    // Compute the conditional mean of the unknowns given the knowns.
    // Args:
    //   known_subset: Vector of known values (x's, in the regression
    //     context), of dimension xdim().
    //   unconditional_mean:  Overall mean, of both knowns and unknowns.
    // Returns:
    //   The conditional mean of the y's (unknowns) given the x's (knowns).
    //   mu[Y | X] = mu[Y] + Beta * (x - mu[X])
    Vector conditional_mean(const Vector &known_subset,
                            const Vector &unconditional_mean) const;

    // Conditional variance of the unknowns (of dimension ydim()).
    SpdMatrix residual_variance() const;

    // Precision matrix of the known component.
    SpdMatrix precision_of_swept_elements() const;

    // Dimension of the unknowns y.
    uint ydim() const;

    // Dimension of the knowns x.  xdim() + ydim() = nrow(S_).
    uint xdim() const;

    const SpdMatrix &swept_matrix() const { return S_; }

   private:
    SpdMatrix S_;

    // swept_[i] is true iff variable i is in the known set.
    Selector swept_;

  };
}  // namespace BOOM

#endif  // BOOM_SWEEP_HPP
