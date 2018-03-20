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

#include "Models/MvnModel.hpp"
#include <cmath>
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/MvnGivenSigma.hpp"
#include "Models/PosteriorSamplers/HierarchicalPosteriorSampler.hpp"
#include "Models/PosteriorSamplers/MvnConjSampler.hpp"
#include "Models/WishartModel.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  double MvnModel::loglike(const Vector &mu_siginv) const {
    const ConstVectorView mu(mu_siginv, 0, dim());
    SpdMatrix siginv(dim());
    Vector::const_iterator b = mu_siginv.cbegin() + dim();
    siginv.unvectorize(b, true);
    return MvnBase::log_likelihood(mu, siginv, *suf());
  }

  void MvnModel::add_raw_data(const Vector &y) {
    NEW(VectorData, dp)(y);
    this->add_data(dp);
  }

  double MvnModel::pdf(const Ptr<Data> &dp, bool logscale) const {
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double MvnModel::pdf(const Data *dp, bool logscale) const {
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double MvnModel::pdf(const Vector &x, bool logscale) const {
    double ans = logp(x);
    return logscale ? ans : exp(ans);
  }

  Vector MvnModel::sim(RNG &rng) const {
    return rmvn_L_mt(rng, mu(), Sigma_chol());
  }

  //======================================================================

  MvnModel::MvnModel(uint p, double mu, double sigma)
      : Base(p, mu, sigma), DataPolicy(new MvnSuf(p)) {}

  MvnModel::MvnModel(const Vector &mean, const SpdMatrix &V, bool ivar)
      : Base(mean, V, ivar), DataPolicy(new MvnSuf(mean.size())) {}

  MvnModel::MvnModel(const Ptr<VectorParams> &mu, const Ptr<SpdParams> &Sigma)
      : Base(mu, Sigma), DataPolicy(new MvnSuf(mu->dim())) {}

  MvnModel::MvnModel(const std::vector<Vector> &data)
      : Base(data[0].size()),
        DataPolicy(new MvnSuf(data[0].size())),
        PriorPolicy() {
    set_data_raw(data.begin(), data.end());
    mle();
  }

  MvnModel::MvnModel(const MvnModel &rhs)
      : Model(rhs),
        VectorModel(rhs),
        Base(rhs),
        LoglikeModel(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        EmMixtureComponent(rhs) {}

  MvnModel *MvnModel::clone() const { return new MvnModel(*this); }

  void MvnModel::mle() {
    set_mu(suf()->ybar());
    set_Sigma(suf()->var_hat());
  }

  void MvnModel::initialize_params() { mle(); }

  void MvnModel::add_mixture_data(const Ptr<Data> &dp, double prob) {
    suf()->add_mixture_data(DAT(dp)->value(), prob);
  }

  void MvnModel::remove_data(const Ptr<Data> &dp) {
    if (DataPolicy::is_raw_data_kept()) {
      DataPolicy::remove_data(dp);
    }
    suf()->remove_data(DAT(dp)->value());
  }

  std::set<Ptr<Data>> MvnModel::abstract_data_set() const {
    return std::set<Ptr<Data>>(dat().begin(), dat().end());
  }

}  // namespace BOOM
