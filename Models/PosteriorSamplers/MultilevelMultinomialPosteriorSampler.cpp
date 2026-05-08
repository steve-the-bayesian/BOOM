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
#include "Models/PosteriorSamplers/MultilevelMultinomialPosteriorSampler.hpp"
#include "Models/MultilevelMultinomialModel.hpp"
#include "distributions.hpp"
#include "cpputil/report_error.hpp"
#include "Models/DirichletModel.hpp"
#include "Models/MultinomialModel.hpp"
#include "Models/PosteriorSamplers/MultinomialDirichletSampler.hpp"

namespace BOOM {

  MultilevelMultinomialPosteriorSampler::MultilevelMultinomialPosteriorSampler(
      MultilevelMultinomialModel *model,
      double default_prior_count_sum,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        taxonomy_(model->taxonomy()),
        model_(model)
  {
    create_default_priors(default_prior_count_sum);
  }

  void MultilevelMultinomialPosteriorSampler::create_default_priors(
      double prior_sum) {
    MultinomialModel *top_level_model = model_->top_level_model();
    int dim = top_level_model->dim();
    top_level_prior_.reset(new DirichletModel(Vector(dim, prior_sum / dim)));
    NEW(MultinomialDirichletSampler, top_level_sampler)(
        top_level_model, top_level_prior_, rng());
    top_level_model->set_method(top_level_sampler);

    for (auto it = taxonomy_->begin(); it != taxonomy_->end(); ++it) {
      const TaxonomyNode *node = (*it).get();
      if (!node->is_leaf()) {
        MultinomialModel *conditional_model = model_->conditional_model(node);
        dim = conditional_model->dim();
        NEW(DirichletModel, conditional_prior)(Vector(dim, prior_sum / dim));
        conditional_model_priors_[node] = conditional_prior;
        NEW(MultinomialDirichletSampler, conditional_sampler)(
            conditional_model, conditional_prior, rng());
        conditional_model->set_method(conditional_sampler);
      }
    }
  }

  void MultilevelMultinomialPosteriorSampler::draw() {
    model_->top_level_model()->sample_posterior();
    for (auto it = taxonomy_->begin(); it != taxonomy_->end(); ++it) {
      const TaxonomyNode *node = (*it).get();
      if (!node->is_leaf()) {
        model_->conditional_model(node)->sample_posterior();
      }
    }
  }

  double MultilevelMultinomialPosteriorSampler::logpri() const {
    double ans = top_level_prior_->logp(model_->top_level_model()->pi());
    for (auto it = taxonomy_->begin(); it != taxonomy_->end(); ++it) {
      const TaxonomyNode *node = (*it).get();
      if (!node->is_leaf()) {
        ans += conditional_model_priors_.find(node)->second->logp(
            model_->conditional_model(node)->pi());
      }
    }
    return ans;
  }

  void MultilevelMultinomialPosteriorSampler::set_top_level_prior_parameters(
      const Vector &prior_counts) {
    top_level_prior_->set_nu(prior_counts);
  }

  void MultilevelMultinomialPosteriorSampler::set_conditional_prior_parameters(
      const std::string &taxonomy_level,
      const Vector &prior_counts,
      char sep) {
    set_conditional_prior_parameters(taxonomy_->node(taxonomy_level, sep),
                                     prior_counts);
  }

  void MultilevelMultinomialPosteriorSampler::set_conditional_prior_parameters(
      const std::vector<std::string> &taxonomy_level,
      const Vector &prior_counts) {
    set_conditional_prior_parameters(taxonomy_->node(taxonomy_level),
                                     prior_counts);
  }

  void MultilevelMultinomialPosteriorSampler::set_conditional_prior_parameters(
      const TaxonomyNode *node,
      const Vector &prior_counts) {
    auto it = conditional_model_priors_.find(node);
    if (it == conditional_model_priors_.end()) {
      if (!node) {
        report_error("NULL node encountered when setting priors for "
                     "MultilevelMultinomialPosteriorSampler.");
      } else {
        std::ostringstream err;
        err << "Error when setting conditional prior parameters for "
            "MultilevelMultinomialPosteriorSampler.  "
            "No prior for taxonomy level "
            << node->path_from_root()
            << " was found.";
        report_error(err.str());
      }
    } else {
      it->second->set_nu(prior_counts);
    }
  }


}  // namespace BOOM
