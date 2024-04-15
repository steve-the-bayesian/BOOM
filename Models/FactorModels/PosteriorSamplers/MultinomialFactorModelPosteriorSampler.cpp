/*
  Copyright (C) 2005-2024 Steven L. Scott

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

#include "Models/FactorModels/PosteriorSamplers/MultinomialFactorModelPosteriorSampler.hpp"
#include "distributions.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/math_utils.hpp"
#include <sstream>

namespace BOOM {

  namespace {
    using Sampler = MultinomialFactorModelPosteriorSampler;
    using Visitor = FactorModels::MultinomialVisitor;
    using Site = FactorModels::MultinomialSite;
  }  // namespace

  Sampler::MultinomialFactorModelPosteriorSampler(
      MultinomialFactorModel *model,
      const Vector &default_prior_class_probabilities,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        visitor_prior_(default_prior_class_probabilities)
  {}

  void Sampler::draw() {
    impute_visitors();
    draw_site_parameters();
    // draw_hyperprior();
  }


  void Sampler::impute_visitors() {
    for (auto &visitor_it : model_->visitors()) {
      Ptr<Visitor> &visitor(visitor_it.second);
      impute_visitor(*visitor);
    }
  }

  void Sampler::impute_visitor(Visitor &visitor) {
    const Vector &prob(prior_class_probabilities(visitor.id()));
    if (prob.max() > .999) {
      visitor.set_class_probabilities(prob);
      visitor.set_class_member_indicator(prob.imax());
    } else {
      Vector logprob = log(prob);
      for (const auto &it : visitor.sites_visited()) {
        const Ptr<Site> &site(it.first);
        logprob += site->logprob();
        check_logprob(logprob);
      }
      Vector post = logprob.normalize_logprob();
      visitor.set_class_probabilities(post);
      visitor.set_class_member_indicator(rmulti_mt(rng(), post));
    }
  }

  void Sampler::draw_site_parameters() {

    // Assemble the site map and reverse site map.  It would be faster to make
    // these member variables, because they won't change across iterations.
    std::map<std::string, Int> site_map;
    std::vector<std::string> reverse_site_map;
    int number_of_classes = model_->number_of_classes();
    Int number_of_sites = model_->number_of_sites();
    Int counter = 0;
    for (const auto &site_it : model_->sites()) {
      const std::string &site_id(site_it.first);
      site_map[site_id] = counter++;
      reverse_site_map.push_back(site_id);
    }

    // Assemble the vectors of counts.  One Vector for each category.  Start by
    // creating the data structures and putting in the priors.
    std::vector<Vector> counts;
    for (int k = 0; k < number_of_classes; ++k) {
      counts.push_back(Vector(number_of_sites, 0.1));
    }

    // Now add the observed data from the visitors.
    for (const auto &visitor_it : model_->visitors()) {
      const Ptr<Visitor> &visitor(visitor_it.second);
      int category = visitor->imputed_class_membership();
      for (const auto &site_it : visitor->sites_visited()) {
        const Ptr<Site> &site(site_it.first);
        Int site_index = site_map[site->id()];
        ++counts[category][site_index];
      }
    }

    // Ready to draw the model parameters.  These are structured the same way as
    // the counts.
    std::vector<Vector> probs;
    for (int k = 0; k < number_of_classes; ++k) {
      probs.push_back(rdirichlet_mt(rng(), counts[k]));
    }

    for (Int i = 0; i < number_of_sites; ++i) {
      Ptr<Site> site = model_->site(reverse_site_map[i]);
      Vector site_probs(number_of_classes);
      for (int k = 0; k < number_of_classes; ++k) {
        site_probs[k] = probs[k][i];
      }
      site->set_probs(site_probs);
    }
  }

  double Sampler::logpri() const {
    return negative_infinity();
  }

  void Sampler::check_logprob(const Vector &logprob) const {
    for (size_t i = 0; i < logprob.size(); ++i) {
      if (!std::isfinite(logprob[i])) {
        std::ostringstream err;
        err << "Element " << i << " is non-finite in:\n"
            << logprob;
        report_error(err.str());
      }
    }
  }

}  // namespace BOOM
