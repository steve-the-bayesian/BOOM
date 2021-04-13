// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_MULTINOMIAL_DIRICHLET_SAMPLER_HPP
#define BOOM_MULTINOMIAL_DIRICHLET_SAMPLER_HPP
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
namespace BOOM {

  class MultinomialModel;
  class DirichletModel;

  class MultinomialDirichletSampler : public PosteriorSampler {
   public:
    MultinomialDirichletSampler(MultinomialModel *mod,
                                const Vector &nu,
                                RNG &seeding_rng = GlobalRng::rng);

    MultinomialDirichletSampler(MultinomialModel *mod,
                                const Ptr<DirichletModel> &prior,
                                RNG &seeding_rng = GlobalRng::rng);

    MultinomialDirichletSampler(const MultinomialDirichletSampler &rhs);
    MultinomialDirichletSampler *clone_to_new_host(
        Model *new_host) const override;

    void draw() override;
    double logpri() const override;
    void find_posterior_mode(double epsilon = 1e-5) override;
    bool can_find_posterior_mode() const override { return true; }

   private:
    MultinomialModel *mod_;
    Ptr<DirichletModel> pri_;
  };


  // A constrained multinomial Dirichlet sampler allows some of the
  // probabilities in the multinomial to be set to zero.
  class ConstrainedMultinomialDirichletSampler
      : public PosteriorSampler {
   public:
    // Args:
    //   model:  The model to be sampled.
    //   prior_counts: The vector of prior counts as in the
    //     MultinomialDirichletSampler.  If any prior counts are negative, the
    //     corresponding probabilities will be set to zero.
    //   seeding_rng: The random number generator used to seed the RNG for this
    //     PosteriorSampler object.
    ConstrainedMultinomialDirichletSampler(
        MultinomialModel *model,
        const Vector &prior_counts,
        RNG &seeding_rng = GlobalRng::rng);

    ConstrainedMultinomialDirichletSampler *clone_to_new_host(
        Model *new_host) const override;
    void draw() override;
    double logpri() const override;

   private:
    MultinomialModel *model_;
    Vector prior_counts_;
    void check_at_least_one_positive(const Vector &counts);
  };

}  // namespace BOOM
#endif  // BOOM_MULTINOMIAL_DIRICHLET_SAMPLER_HPP
