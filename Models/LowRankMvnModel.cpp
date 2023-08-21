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

#include "Models/LowRankMvnModel.hpp"
#include "distributions.hpp"

namespace BOOM {

  LowRankMvnModel::LowRankMvnModel(const Vector &mu, const SpdMatrix &Sigma)
      : ParamPolicy(new VectorParams(mu),
                    new PositiveSemidefiniteParams(Sigma)),
        DataPolicy(new MvnSuf(mu.size()))
  {}

  LowRankMvnModel::LowRankMvnModel(const LowRankMvnModel &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs)
  {}

  LowRankMvnModel::LowRankMvnModel(LowRankMvnModel &&rhs)
      : Model(std::move(rhs)),
        ParamPolicy(std::move(rhs)),
        DataPolicy(std::move(rhs)),
        PriorPolicy(std::move(rhs))
  {}

  LowRankMvnModel &LowRankMvnModel::operator=(const LowRankMvnModel &rhs) {
    if (&rhs != this) {
      Model::operator=(rhs);
      ParamPolicy::operator=(rhs);
      DataPolicy::operator=(rhs);
      PriorPolicy::operator=(rhs);
    }
    return *this;
  }

  LowRankMvnModel &LowRankMvnModel::operator=(LowRankMvnModel &&rhs) {
    if (&rhs != this) {
      Model::operator=(std::move(rhs));
      ParamPolicy::operator=(std::move(rhs));
      DataPolicy::operator=(std::move(rhs));
      PriorPolicy::operator=(std::move(rhs));
    }
    return *this;
  }

  LowRankMvnModel * LowRankMvnModel::clone() const {
    return new LowRankMvnModel(*this);
  }

  const Vector &LowRankMvnModel::mu() const {
    return prm1_ref().value();
  }

  const SpdMatrix &LowRankMvnModel::Sigma() const {
    return prm2_ref().value();
  }

  const SpdMatrix &LowRankMvnModel::siginv() const {
    return prm2_ref().generalized_inverse();
  }

  double LowRankMvnModel::ldsi() const {
    return prm2_ref().generalized_ldsi();
  }

  Vector LowRankMvnModel::sim(RNG &rng) const {
    const Matrix &root(prm2_ref().root());
    int zdim = root.ncol();
    Vector standard(zdim);
    for (int i = 0; i < zdim; ++i) {
      standard[i] = rnorm_mt(rng);
    }
    return mu() + root * standard;
  }

}  // namespace BOOM
