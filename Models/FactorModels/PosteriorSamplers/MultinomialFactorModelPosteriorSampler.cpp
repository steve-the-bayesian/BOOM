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
#include "cpputil/DefaultMap.hpp"
#include <sstream>
#include <ctime>

namespace BOOM {
  using Visitor = FactorModels::MultinomialVisitor;
  using Site = FactorModels::MultinomialSite;
  inline void increment_counts(const Ptr<Visitor> &visitor,
                               int category,
                               Matrix &counts,
                               const DefaultMap<std::string, Int> &site_index_map) {
    for (const auto &site_el : visitor->sites_visited()) {
      const Ptr<Site> &site(site_el.first);
      ++counts(site_index_map[site->id()], category);
    }
  }

  void check_logprob(const Vector &logprob) {
    for (size_t i = 0; i < logprob.size(); ++i) {
      if (!std::isfinite(logprob[i])) {
        std::ostringstream err;
        err << "Element " << i << " is non-finite in:\n"
            << logprob;
        report_error(err.str());
      }
    }
  }

  namespace MfmThreading {
    using Visitor = FactorModels::MultinomialVisitor;
    using Site = FactorModels::MultinomialSite;

    void VisitorImputer::impute_visitors() {
      for (auto &visitor : visitors_) {
        impute_visitor(*visitor);
      }
    }

    void VisitorImputer::impute_visitor(Visitor &visitor) {

      const Vector &prob(prior_manager_->prior_class_probabilities(visitor.id()));
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
        visitor.set_class_member_indicator(rmulti_mt(rng_, post));
      }
    }

  }  // namespace MfmThreading

  namespace {
    using Sampler = MultinomialFactorModelPosteriorSampler;
    using Visitor = FactorModels::MultinomialVisitor;
    using Site = FactorModels::MultinomialSite;

    using VisitorImputer = MfmThreading::VisitorImputer;
  }  // namespace

  Sampler::MultinomialFactorModelPosteriorSampler(
      MultinomialFactorModel *model,
      const Vector &default_prior_class_probabilities,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        visitor_prior_(default_prior_class_probabilities),
        unknown_visitors_(),
        known_site_visit_counts_()
  {
    set_num_threads(1);
  }

  void Sampler::set_num_threads(int num_threads) {
    if (num_threads < 1) {
      num_threads = 1;
    }
    visitor_imputers_.clear();

    for (int i = 0; i < num_threads; ++i) {
      visitor_imputers_.push_back(VisitorImputer(
          rng(), &visitor_prior_));
    }

    // If the "known_visitors" optimization has been set, then only impute the
    // unknown visitors.  Otherwise impute all the visitors.
    size_t counter = 0;
    if (unknown_visitors_.empty()) {
      for (const auto &el : model_->visitors()) {
        visitor_imputers_[counter++ % num_threads].add_visitor(el.second);
      }
    } else {
      for (const Ptr<Visitor> &visitor : unknown_visitors_) {
        visitor_imputers_[counter++ % num_threads].add_visitor(visitor);
      }
    }

    if (num_threads <= 1) {
      pool_.set_number_of_threads(0);
    } else {
      pool_.set_number_of_threads(num_threads);
    }

  }

  void Sampler::draw() {
    fill_unknown_visitors();
    impute_visitors();
    draw_site_parameters();
  }

  void Sampler::impute_visitors() {
    MoveTimer timer = accounting_.start_time("impute_visitors");
    if (visitor_imputers_.size() == 1) {
      visitor_imputers_[0].impute_visitors();
    } else {
      std::vector<std::future<void>> futures;
      for (size_t i = 0; i < visitor_imputers_.size(); ++i) {
        VisitorImputer *imputer = &visitor_imputers_[i];
        futures.emplace_back(
            pool_.submit(
                [imputer]() {imputer->impute_visitors();}));
      }

      for (int i = 0; i < futures.size(); ++i) {
        futures[i].get();
      }
    }
  }

  void Sampler::fill_unknown_visitors() {
    if (unknown_visitors_.empty()) {
      known_site_visit_counts_ = Matrix(
          model_->number_of_sites(),
          model_->number_of_classes(),
          0.0);

      std::map<std::string, Int> raw_site_index_map = model_->site_index_map();
      DefaultMap<std::string, Int> site_index_map(&raw_site_index_map,
                                                  model_->default_site_name());
      for (const auto &visitor_el : model_->visitors()) {
        const Ptr<Visitor> &visitor(visitor_el.second);
        const Vector &prior(prior_class_probabilities(visitor->id()));
        int category = prior.imax();
        if (prior[category] < .999) {
          unknown_visitors_.insert(visitor);
        } else {
          increment_counts(visitor,
                           category,
                           known_site_visit_counts_,
                           site_index_map);
        }
      }
      if (visitor_imputers_.size() > 1) {
        int num_threads = visitor_imputers_.size();
        set_num_threads(1);
        set_num_threads(num_threads);
      }
    }
  }


  // The posterior distribution across site parameters is independent across
  // categories, but not across sites.
  void Sampler::draw_site_parameters() {
    int number_of_classes = model_->number_of_classes();
    Int number_of_sites = model_->number_of_sites();

    // Assemble the vectors of counts.  These are the number of visits by
    // distinct visitors to each site, in each category.
    //
    // Start by creating the data structures and putting in the priors.
    //
    // Counts starts off with a Dirichlet prior in each category with 0.1 prior
    // observations per site for that category.
    Matrix counts(number_of_sites, number_of_classes, 0.1);
    std::map<std::string, Int> raw_site_index_map = model_->site_index_map();
    DefaultMap<std::string, Int> site_index_map(&raw_site_index_map,
                                                model_->default_site_name());
    // Now add the observed data from the visitors.
    if (unknown_visitors_.empty()) {

      for (const auto &visitor_el : model_->visitors()) {
        const Ptr<Visitor> &visitor(visitor_el.second);
        int category = visitor->imputed_class_membership();
        increment_counts(visitor, category, counts, site_index_map);
      }
    } else {
      counts += known_site_visit_counts_;
      for (const Ptr<Visitor> &visitor: unknown_visitors_) {
        increment_counts(visitor,
                         visitor->imputed_class_membership(),
                         counts,
                         site_index_map);
      }
    }

    // Ready to draw the model parameters.  These are structured the same way as
    // the counts.
    Matrix probs(number_of_sites, number_of_classes);
    for (int k = 0; k < number_of_classes; ++k) {
      probs.col(k) = rdirichlet_mt(rng(), counts.col(k));
    }

    Int row_counter = 0;
    for (auto &el : model_->sites()) {
      el.second->set_probs(probs.row(row_counter++));
    }
  }

  double Sampler::logpri() const {
    return negative_infinity();
  }

}  // namespace BOOM
