#ifndef BOOM_MVREG_COPULA_DATA_IMPUTER_HPP_
#define BOOM_MVREG_COPULA_DATA_IMPUTER_HPP_
/*
  Copyright (C) 2005-2020 Steven L. Scott

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

#include "Policies/CompositeParamPolicy.hpp"
#include "Policies/IID_DataPolicy.hpp"
#include "Policies/PriorPolicy.hpp"
#include "Models/Glm/MvReg.hpp"

#include "Models/WishartModel.hpp"
#include "Models/ChisqModel.hpp"

namespace BOOM {

  // Given a collection of continuous variables Y, which contain missing values,
  // and a collection of fully observed predictors X (which might be empty),
  // impute the missing values in Y from their posterior distribution using the
  // following algorithm. Let Z ~ Mvn(X * Beta, Sigma), where Z is the Gaussian
  // copula transformed, fully observed, version of Y.
  //
  // 1) Simulate Beta, Sigma ~ p(Beta, Sigma | Z).
  // 2) Simulate Zmis | Zobs, Beta, Sigma.
  // 3) Transform Z -> Y, and recompute the copula transform.
  // 4) Transform back to Z, then repeat.
  class MvRegCopulaDataImputer
      : public IID_DataPolicy<PartiallyMissingDataTable>,
        public CompositeParamPolicy,
        public PriorPolicy
  {
   public:

    void impute(RNG &rng);

    void impute_atoms(RNG &rng);

   private:
    Ptr<DataFrame> original_data_;
    Ptr<DataFrame> imputed_data_;

    std::vector<Vector> atoms_;

    // Describes the component to which each observation belongs.  This model
    // controls the sharing of information across variables.
    //
    // The "emission distribution" for this model is the
    Ptr<FiniteMixtureModel> observation_class_model_;

    std::vector<Ptr<FiniteMixtureModel>> atom_models_;

    Ptr<MultivariateRegressionModel> complete_data_model_;
  };

}  // namespace BOOM

#endif  // BOOM_MVREG_COPULA_DATA_IMPUTER_HPP_
