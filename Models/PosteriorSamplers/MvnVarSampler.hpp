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
    // assumes y~N(mu, Sigma), with Sigma^-1~W(df, SS).  The prior on mu may or
    // may not be conjugate.  The sampling step will condition on mu.  Use
    // MvnConjVarSampler if you want to integrate out mu.
   public:
    // Args:
    //   model:  The model to be sampled.
    //   df: The degrees of freedom parameter for the Wishart distribution.  The
    //     distribution is proper if df > dim - 1, where 'dim' is the dimension.
    //   sum_of_squares: The sum of squares matrix.  The mean of the
    //     distribution is sum_of_squares * df.
    //   seeding_rng: The random number generator used to seed the RNG owned by
    //     this object.
    MvnVarSampler(MvnModel *model, double df, const SpdMatrix &sum_of_squares,
                  RNG &seeding_rng = GlobalRng::rng);

    // Args:
    //   model:  The model to be sampled.
    //   precision_prior:  The Wishart distribution to use as a prior.
    //   seeding_rng: The random number generator used to seed the RNG owned by
    //     this object.
    MvnVarSampler(MvnModel *model, const Ptr<WishartModel> &precision_prior,
                  RNG &seeding_rng = GlobalRng::rng);

    MvnVarSampler *clone_to_new_host(Model *new_host) const override;

    double logpri() const override;
    void draw() override;

    // Args:
    //   rng:  The random number generator to use for the draw.
    //   data_sample_size: The sample size to use when computing the posterior
    //     distribution.  This number gets added to the prior sample size
    //     obtained from precision_prior.
    //   data_centered_sum_of_squares: The sum of squares matrix from the data.
    //     This gets added to the prior sum of squares obtained from
    //     precision_prior.
    //
    // Returns:
    //   A posterior draw of the precision matrix.
    static SpdMatrix draw_precision(
        RNG &rng,
        double data_sample_size,
        const SpdMatrix &data_centered_sum_of_squares,
        const WishartModel &precision_prior);

    // Args:
    //   rng:  The random number generator to use for the draw.
    //   data_sample_size: The sample size to use when computing the posterior
    //     distribution.  This number gets added to the prior sample size
    //     obtained from precision_prior.
    //   data_centered_sum_of_squares: The sum of squares matrix from the data.
    //     This gets added to the prior sum of squares obtained from
    //     precision_prior.
    //
    // Returns:
    //   A posterior draw of the variance matrix.  This is equivalent to (but
    //   faster than) draw_precision(...).inv().
    static SpdMatrix draw_variance(
        RNG &rng,
        double data_sample_size,
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
    MvnConjVarSampler(MvnModel *,
                      double df,
                      const SpdMatrix &SS,
                      RNG &seeding_rng = GlobalRng::rng);
    MvnConjVarSampler(MvnModel *,
                      const Ptr<WishartModel> &siginv_prior,
                      RNG &seeding_rng = GlobalRng::rng);
    MvnConjVarSampler *clone_to_new_host(Model *new_host) const override;
    void draw() override;
  };

}  // namespace BOOM
#endif  // BOOM_MVN_VAR_SAMPLER_HPP
