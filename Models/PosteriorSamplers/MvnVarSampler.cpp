// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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
#include "Models/PosteriorSamplers/MvnVarSampler.hpp"
#include "Models/MvnModel.hpp"
#include "Models/WishartModel.hpp"
#include "distributions.hpp"
namespace BOOM {

  MvnVarSampler::MvnVarSampler(MvnModel *m, double df,
                               const SpdMatrix &variance_estimate,
                               RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(m),
        prior_(new WishartModel(df, variance_estimate)) {}

  MvnVarSampler::MvnVarSampler(MvnModel *m,
                               const Ptr<WishartModel> &siginv_prior,
                               RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), model_(m), prior_(siginv_prior) {}

  double MvnVarSampler::logpri() const {
    return prior_->logp(model_->siginv());
  }

  MvnVarSampler *MvnVarSampler::clone_to_new_host(Model *new_host) const {
    return new MvnVarSampler(
        dynamic_cast<MvnModel*>(new_host),
        prior_->clone(),
        rng());
  }

  void MvnVarSampler::draw() {
    Ptr<MvnSuf> suf = model_->suf();
    model_->set_siginv(draw_precision(
        rng(), suf->n(), suf->center_sumsq(model_->mu()), *prior_));
  }

  SpdMatrix MvnVarSampler::draw_precision(
      RNG &rng, double data_sample_size,
      const SpdMatrix &data_centered_sum_of_squares,
      const WishartModel &precision_prior) {
    return rWish_mt(
        rng, precision_prior.nu() + data_sample_size,
        (data_centered_sum_of_squares + precision_prior.sumsq()).inv(), false);
  }

  SpdMatrix MvnVarSampler::draw_variance(
      RNG &rng, double data_sample_size,
      const SpdMatrix &data_centered_sum_of_squares,
      const WishartModel &precision_prior) {
    return rWish_mt(
        rng, precision_prior.nu() + data_sample_size,
        (data_centered_sum_of_squares + precision_prior.sumsq()).inv(), true);
  }

  //======================================================================

  MvnConjVarSampler::MvnConjVarSampler(MvnModel *m, double df,
                                       const SpdMatrix &sumsq, RNG &seeding_rng)
      : MvnVarSampler(m, df, sumsq, seeding_rng) {}

  MvnConjVarSampler::MvnConjVarSampler(MvnModel *m,
                                       const Ptr<WishartModel> &prior,
                                       RNG &seeding_rng)
      : MvnVarSampler(m, prior, seeding_rng) {}

  MvnConjVarSampler *MvnConjVarSampler::clone_to_new_host(
      Model *new_host) const {
    return new MvnConjVarSampler(
        dynamic_cast<MvnModel *>(new_host),
        prior()->clone(),
        rng());
  }

  void MvnConjVarSampler::draw() {
    Ptr<MvnSuf> suf = model()->suf();
    model()->set_siginv(MvnVarSampler::draw_precision(
        rng(), suf->n() - 1, suf->center_sumsq(suf->ybar()), *prior()));
  }

}  // namespace BOOM
