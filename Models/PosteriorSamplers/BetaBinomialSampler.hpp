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

#ifndef BOOM_BETA_BINOMIAL_SAMPLER_HPP
#define BOOM_BETA_BINOMIAL_SAMPLER_HPP

#include "Models/BetaModel.hpp"
#include "Models/BinomialModel.hpp"
#include "Models/PosteriorSamplers/HierarchicalPosteriorSampler.hpp"

namespace BOOM {

  class BetaBinomialSampler : public ConjugateHierarchicalPosteriorSampler {
   public:
    BetaBinomialSampler(BinomialModel *model, const Ptr<BetaModel> &prior,
                        RNG &seeding_rng = GlobalRng::rng);
    BetaBinomialSampler *clone_to_new_host(Model *new_host) const override;
    void draw() override;
    double logpri() const override;

    void find_posterior_mode(double epsilon = 1e-5) override;

    void draw_model_parameters(Model &model) override;
    void draw_model_parameters(BinomialModel &model);

    double log_prior_density(const ConstVectorView &parameters) const override;
    double log_prior_density(const Model &model) const override;
    double log_prior_density(const BinomialModel &model) const;

    // The generic log_marginal_density throws an error.
    double log_marginal_density(const Ptr<Data> &dp,
                                const ConjugateModel *model) const override;
    double log_marginal_density(const BinomialData &data,
                                const BinomialModel *model) const;

   private:
    BinomialModel *model_;
    Ptr<BetaModel> prior_;
  };

}  // namespace BOOM

#endif  // BOOM_BETA_BINOMIAL_SAMPLER_HPP
