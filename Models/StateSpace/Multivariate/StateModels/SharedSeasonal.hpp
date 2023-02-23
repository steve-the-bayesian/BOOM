#ifndef BOOM_STATE_SPACE_MULTIVARIATE_SHARED_SEASONAL_HPP_
#define BOOM_STATE_SPACE_MULTIVARIATE_SHARED_SEASONAL_HPP_

/*
  Copyright (C) 2005-2023 Steven L. Scott

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

#include "Models/Policies/NullDataPolicy.hpp"

namespace BOOM {

  //===========================================================================
  // A seasonal state model for multivariate outcomes.  The latent state
  // consists of K seasonal factors, each with the same period (S) and season
  // duration.
  //
  // Each factor comprises a vector of size S-1, with the first element
  // containing the seasonal effect for the current period, and the remaining
  // S-2 elements containing the lag-1, lag-2, etc seasonal effects.
  //
  // Series i relates to factor k through a coefficient beta[i, k]
  //
  // ===========================================================================


  class SharedSeasonalStateModelBase
      : public SharedStateModel,
        public CompositeParamPolicy,
        public NullDataPolicy,
        public PriorPolicy
  {
   public:
    SharedSeasonalStateModelBase(int number_of_factors, int nseries);
    SharedSeasonalStateModelBase(const SharedSeasonalStateModelBase &rhs);
    SharedSeasonalStateModelBase(SharedSeasonalStateModelBase &&rhs);
    SharedSeasonalStateModelBase & operator=(
        const SharedSeasonalStateModelBase &rhs);
    SharedSeasonalStateModelBase & operator=(
        SharedSeasonalStateModelBase &&rhs);

    SharedSeasonalStateModelBase * clone() const override = 0;

    //---------------------------------------------------------------------------
    // Sizes of things.
    uint state_dimension() const override;
    uint state_error_dimension() const override;
    virtual int nseries const = 0;

    int number_of_factors() const;

    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;
    void simulate_initial_state(RNG &rhg, VectorView eta) const override;

    virtual MultivariateStateSpaceModelBase *host() = 0;
    virtual const MultivariateStateSpaceModelBase *host() const = 0;


    //---------------------------------------------------------------------------
    // Model matrices.  Implementation TBD.
    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_error_expander(int t) const override;
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override;
  };

}



#endif  // BOOM_STATE_SPACE_MULTIVARIATE_SHARED_SEASONAL_HPP_
