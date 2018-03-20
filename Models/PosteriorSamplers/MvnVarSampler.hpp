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
#ifndef BOOM_MVN_VAR_SAMPLER_HPP
#define BOOM_MVN_VAR_SAMPLER_HPP
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/SpdParams.hpp"

namespace BOOM {
  class MvnModel;
  class WishartModel;

  class MvnVarSampler : public PosteriorSampler {
    // assumes y~N(mu, Sigma), with Sigma^-1~W(df, SS).  The prior on
    // mu may or may not be conjugate.  The sampling step will
    // condition on mu.  Use MvnConjVarSampler if you want to
    // integrate out mu.
   public:
    MvnVarSampler(MvnModel *, double df, const SpdMatrix &SS,
                  RNG &seeding_rng = GlobalRng::rng);
    MvnVarSampler(MvnModel *, const Ptr<WishartModel> &siginv_prior,
                  RNG &seeding_rng = GlobalRng::rng);
    double logpri() const override;
    void draw() override;

    // Returns a draw of the precision matrix for an MvnModel.
    static SpdMatrix draw_precision(
        RNG &rng, double data_sample_size,
        const SpdMatrix &data_centered_sum_of_squares,
        const WishartModel &precision_prior);

    // Returns a draw of the variance matrix for an MvnModel.
    static SpdMatrix draw_variance(
        RNG &rng, double data_sample_size,
        const SpdMatrix &data_centered_sum_of_squares,
        const WishartModel &precision_prior);

   private:
    MvnModel *model_;
    Ptr<WishartModel> prior_;

   protected:
    MvnModel *model() { return model_; }
    const WishartModel *prior() const { return prior_.get(); }
  };

  class MvnConjVarSampler : public MvnVarSampler {
    // assumes y~N(mu, Sigma), with mu|Sigma \norm(mu0, Sigma/kappa)
    // and Sigma^-1~W(df, SS)
   public:
    MvnConjVarSampler(MvnModel *, double df, const SpdMatrix &SS,
                      RNG &seeding_rng = GlobalRng::rng);
    MvnConjVarSampler(MvnModel *, const Ptr<WishartModel> &siginv_prior,
                      RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
  };

}  // namespace BOOM
#endif  // BOOM_MVN_VAR_SAMPLER_HPP
