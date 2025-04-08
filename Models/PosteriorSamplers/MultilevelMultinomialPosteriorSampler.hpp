#ifndef BOOM_MODELS_MULTILEVEL_MULTINOMIAL_POSTERIOR_SAMPLER_HPP_
#define BOOM_MODELS_MULTILEVEL_MULTINOMIAL_POSTERIOR_SAMPLER_HPP_
/*
  Copyright (C) 2005-2025 Steven L. Scott

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

#include "Models/MultilevelMultinomialModel.hpp"
#include "Models/DirichletModel.hpp"
#include "Models/MultinomialModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  // Posterior sampler for the MultilevelMultinomialModel based on conjugate
  // independent Dirichlet priors for each of the multinomial sub-models inside
  // the MultilevelMultinomialModel object.
  class MultilevelMultinomialPosteriorSampler : public PosteriorSampler {
   public:
    // Args:
    //   model:  The model whose posterior distribution is to be sampled.
    //   default_prior_sum:  See below.
    //   seeding_rng: The random number generator used to seed this sampler's
    //     RNG.
    //
    // Upon construction, each model in *model is assigned a Dirichlet prior
    // distribution.  The "prior counts" in that distribution are all equal,
    // with a sum of 'default_prior_sum'.  Thus if a level has 4 categories and
    // 'default_prior_sum == 3.2' then the prior assigned to that model will be
    // (.25, .25, .25, .25) * 3.2.
    MultilevelMultinomialPosteriorSampler(
        MultilevelMultinomialModel *model,
        double default_prior_sum = 1.0,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;
    bool can_find_posterior_mode() const override {return false;}

    // Users can set specific priors for specific models.  In each case the
    // vector of prior_counts must contain all positive numbers and be of the
    // correct dimension.
    void set_top_level_prior_parameters(const Vector &prior_counts);

    // Priors for conditional models can be accessed by taxonomy level in one of
    // the three standard ways:
    //   (a) A string of the form "shopping/clothes/shoes", with a character
    //       field separator (in this example '/').
    //   (b) A vector of strings giving the taxonomy level (e.g. ["shopping",
    //       "clothes", "shoes"].
    //   (c) A TaxonomyNode pointing to the taxonomy category.
    //
    // In each of the cases, the prior distribution is for the model describing
    // the child levels beneath the specified category.
    void set_conditional_prior_parameters(
        const std::string &taxonomy_level,
        const Vector &prior_counts,
        char sep = '/');

    void set_conditional_prior_parameters(
        const std::vector<std::string> &taxonomy_level,
        const Vector &prior_counts);

    void set_conditional_prior_parameters(
        const TaxonomyNode *node,
        const Vector &prior_counts);

   private:
    // The taxonomy describing the data supported by the model.
    Ptr<Taxonomy> taxonomy_;

    // The model being managed.
    MultilevelMultinomialModel *model_;

    // Prior for the top level multinomial model inside model_.
    Ptr<DirichletModel> top_level_prior_;

    // Priors for each of the conditional models held inside model_.
    std::map<const TaxonomyNode *,
             Ptr<DirichletModel>> conditional_model_priors_;

    void create_default_priors(double prior_sum);
  };

}  // namespace BOOM
#endif  // BOOM_MODELS_MULTILEVEL_MULTINOMIAL_POSTERIOR_SAMPLER_HPP_
