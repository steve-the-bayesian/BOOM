#ifndef BOOM_MODELS_LOW_RANK_MVN_MODEL_HPP_
#define BOOM_MODELS_LOW_RANK_MVN_MODEL_HPP_

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

#include "Models/ParamTypes.hpp"
#include "Models/PositiveSemidefiniteData.hpp"
#include "Models/MvnBase.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"


// A multivariate normal model with a less than full rank covariance matrix.
// This is sometimes called the "degenerate" model or "rank deficient" model.
//
// Let the dimension of the model be n.  The model is defined in terms of an
// m-dimensional standard normal random vector z (where m <= n).  The random
// variable y = mu + A * z, where A is an n x m matrix and mu is an n-vector.
// The mean of y is mu, and the variance is A * A', which is non-negative
// definite but less than full rank.
//
// The low rank MVN model does not in general have a density function (though
// linear transformations of it to the full-rank space will have densities), and
// the inverse variance is not defined.
//
namespace BOOM {

  class LowRankMvnModel
      : public MvnBase,
        public ParamPolicy_2<VectorParams, PositiveSemidefiniteParams>,
        public SufstatDataPolicy<VectorData, MvnSuf>,
        public PriorPolicy
  {
   public:
    LowRankMvnModel(const Vector &mu, const SpdMatrix &Sigma);
    LowRankMvnModel(const LowRankMvnModel &rhs);
    LowRankMvnModel(LowRankMvnModel &&rhs);
    LowRankMvnModel &operator=(const LowRankMvnModel &rhs);
    LowRankMvnModel &operator=(LowRankMvnModel &&rhs);
    LowRankMvnModel *clone() const override;

    const Vector &mu() const override;
    const SpdMatrix &Sigma() const override;
    const SpdMatrix &siginv() const override;
    double ldsi() const override;
    Vector sim(RNG &rng = GlobalRng::rng) const override;
  };

}  // namespace BOOM


#endif //  BOOM_MODELS_LOW_RANK_MVN_MODEL_HPP_
