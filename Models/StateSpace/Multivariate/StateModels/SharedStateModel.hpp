# ifndef BOOM_STATE_SPACE_SHARED_STATE_MODEL_HPP_
# define BOOM_STATE_SPACE_SHARED_STATE_MODEL_HPP_

/*
  Copyright (C) 2005-2021 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/StateSpace/Filters/SparseMatrix.hpp"

namespace BOOM {
  // State models for dynamic factor models and similar multivariate time series
  // with fixed dimension.
  class SharedStateModel : virtual public StateModelBase {
   public:
    SharedStateModel * clone() const override = 0;

    // The coefficients (Z) in the observation equation.  The coefficients are
    // arranged so that y = Z * state + error.  Thus columns of the observation
    // coefficients Z correspond to the state dimension.
    //
    // Args:
    //   t:  The time index of the observation.
    //   observed: Indicates which elements of the outcome variable are observed
    //     at time t.  Rows of Z corresponding to unobserved variables are
    //     omitted.
    virtual Ptr<SparseMatrixBlock> observation_coefficients(
        int t, const Selector &observed) const = 0;
  };

}


# endif //  BOOM_STATE_SPACE_SHARED_STATE_MODEL_HPP_
