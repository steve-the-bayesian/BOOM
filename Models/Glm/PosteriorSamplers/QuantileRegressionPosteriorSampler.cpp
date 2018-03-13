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

#include "Models/Glm/PosteriorSamplers/QuantileRegressionPosteriorSampler.hpp"
#include "distributions/inverse_gaussian.hpp"

namespace BOOM {
  namespace {
    typedef QuantileRegressionPosteriorSampler QRPS;
    typedef QuantileRegressionImputeWorker QRIW;
    typedef QuantileRegressionSpikeSlabSampler QRSSS;
  }  // namespace

  void QRIW::impute_latent_data_point(const RegressionData &observed,
                                      WeightedRegSuf *suf, RNG &rng) {
    double residual = fabs(observed.y() - coefficients_->predict(observed.x()));
    if (residual > 0) {
      double lambda_inv = rig_mt(rng, 1.0 / residual, 1.0);
      double lambda = 1.0 / lambda_inv;
      suf->add_data(observed.x(), adjusted_observation(observed.y(), lambda),
                    lambda_inv);
    }
  }

  //======================================================================
  QRPS::QuantileRegressionPosteriorSampler(QuantileRegressionModel *model,
                                           const Ptr<MvnBase> &prior, RNG &rng)
      : PosteriorSampler(rng),
        model_(model),
        prior_(prior),
        suf_(model->xdim()) {
    set_number_of_workers(1);
  }

  void QRPS::draw() {
    impute_latent_data();
    draw_params();
  }

  double QRPS::logpri() const { return prior_->logp(model_->Beta()); }

  void QRPS::draw_params() {
    SpdMatrix ivar = prior_->siginv() + suf_.xtx();
    Vector ivar_mu = suf_.xty() + prior_->siginv() * prior_->mu();
    Vector draw = rmvn_suf_mt(rng(), ivar, ivar_mu);
    model_->set_Beta(draw);
  }

  Ptr<QRIW> QRPS::create_worker(std::mutex &suf_mutex) {
    return new QRIW(model_->coef_prm().get(), model_->quantile(), suf_,
                    suf_mutex, nullptr, rng());
  }

  void QRPS::clear_latent_data() { suf_.clear(); }

  void QRPS::assign_data_to_workers() {
    BOOM::assign_data_to_workers(model_->dat(), workers());
  }

  //======================================================================
  QRSSS::QuantileRegressionSpikeSlabSampler(
      QuantileRegressionModel *model, const Ptr<MvnBase> &slab,
      const Ptr<VariableSelectionPrior> &spike, RNG &seeding_rng)
      : QuantileRegressionPosteriorSampler(model, slab, seeding_rng),
        sam_(model, slab, spike),
        slab_prior_(slab),
        spike_prior_(spike) {}

  void QRSSS::draw() {
    impute_latent_data();
    sam_.draw_model_indicators(rng(), suf());
    sam_.draw_beta(rng(), suf());
  }

  double QRSSS::logpri() const { return sam_.logpri(); }

}  // namespace BOOM
