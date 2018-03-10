// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

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

#include "Models/Glm/PosteriorSamplers/BinomialProbitSpikeSlabSampler.hpp"

namespace BOOM {

  namespace {
    typedef BinomialProbitSpikeSlabSampler BPSSS;
  }  // namespace

  BPSSS::BinomialProbitSpikeSlabSampler(
      BinomialProbitModel *model, const Ptr<MvnBase> &slab_prior,
      const Ptr<VariableSelectionPrior> &spike_prior, int clt_threshold,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        slab_prior_(slab_prior),
        spike_prior_(spike_prior),
        spike_slab_(model_, slab_prior_, spike_prior_),
        imputer_(clt_threshold) {}

  void BPSSS::draw() {
    impute_latent_data();
    spike_slab_.draw_model_indicators(rng(),
                                      complete_data_sufficient_statistics());
    spike_slab_.draw_beta(rng(), complete_data_sufficient_statistics());
  }

  double BPSSS::logpri() const { return spike_slab_.logpri(); }

  void BPSSS::allow_model_selection(bool tf) {
    spike_slab_.allow_model_selection(tf);
  }

  void BPSSS::limit_model_selection(int max_flips) {
    spike_slab_.limit_model_selection(max_flips);
  }

  void BPSSS::impute_latent_data() {
    if (nrow(xtx_) != model_->xdim()) {
      refresh_xtx();
    }

    xtz_.resize(model_->xdim());
    xtz_ = 0.0;
    const std::vector<Ptr<BinomialRegressionData>> &data(model_->dat());
    for (int i = 0; i < data.size(); ++i) {
      const Vector &x(data[i]->x());
      double sum_of_z = imputer_.impute(rng(), data[i]->n(), data[i]->y(),
                                        model_->predict(x));
      xtz_.axpy(x, sum_of_z);
    }
  }

  void BPSSS::refresh_xtx() {
    xtx_.resize(model_->xdim());
    const std::vector<Ptr<BinomialRegressionData>> &data(model_->dat());
    for (int i = 0; i < data.size(); ++i) {
      xtx_.add_outer(data[i]->x(), data[i]->n());
    }
  }

  WeightedRegSuf BPSSS::complete_data_sufficient_statistics() const {
    WeightedRegSuf suf(model_->xdim());
    suf.set_xtwx(xtx_);
    suf.set_xtwy(xtz_);
    return suf;
  }

}  // namespace BOOM
