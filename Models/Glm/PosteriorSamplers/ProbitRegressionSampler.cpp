// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#include "Models/Glm/PosteriorSamplers/ProbitRegressionSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef ProbitRegressionSampler PRS;
  }

  PRS::ProbitRegressionSampler(ProbitRegressionModel *model,
                               const Ptr<MvnBase> &prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        prior_(prior),
        xtx_(model_->xdim()),
        xtz_(model_->xdim()) {
    refresh_xtx();
  }

  double PRS::logpri() const { return prior_->logp(model_->Beta()); }

  void PRS::draw() {
    impute_latent_data();
    draw_beta();
  }

  void PRS::draw_beta() {
    model_->set_Beta(rmvn_suf_mt(rng(), xtx_ + prior_->siginv(),
                                 xtz_ + prior_->siginv() * prior_->mu()));
  }

  void PRS::impute_latent_data() {
    const ProbitRegressionModel::DatasetType &data(model_->dat());
    int n = data.size();
    const Vector &beta(model_->Beta());
    xtz_ = 0;
    for (int i = 0; i < n; ++i) {
      const Vector &x(data[i]->x());
      double z = imputer_.impute(rng(), 1, data[i]->y(), x.dot(beta));
      xtz_.axpy(x, z);
    }
  }

  const Vector &PRS::xtz() const { return xtz_; }
  const SpdMatrix &PRS::xtx() const { return xtx_; }

  void PRS::refresh_xtx() {
    int p = model_->xdim();
    xtx_.resize(p);
    xtx_ = 0;
    const ProbitRegressionModel::DatasetType &data(model_->dat());
    int n = data.size();
    for (int i = 0; i < n; ++i) {
      const Vector &x(data[i]->x());
      xtx_.add_outer(x, 1, false);
    }
    xtx_.reflect();
  }

}  // namespace BOOM
