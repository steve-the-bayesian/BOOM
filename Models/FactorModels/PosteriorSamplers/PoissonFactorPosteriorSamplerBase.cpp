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

#include "Models/FactorModels/PosteriorSamplers/PoissonFactorPosteriorSamplerBase.hpp"
#include <sstream>
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using Sampler = PoissonFactorPosteriorSamplerBase;
    using Visitor = PoissonFactor::Visitor;
    using Site = PoissonFactor::Site;
  }

  Sampler::PoissonFactorPosteriorSamplerBase(
      PoissonFactorModel *model,
      const Vector &default_prior_class_probabilities,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        default_prior_class_probabilities_(default_prior_class_probabilities),
        exposure_counts_(model->number_of_classes(), 0.0)
  {
    check_probabilities(default_prior_class_probabilities);
  }

  const Vector &Sampler::prior_class_probabilities(
      const std::string &visitor_id) const {
    auto it = prior_class_probabilities_.find(visitor_id);
    if (it == prior_class_probabilities_.end()) {
      return default_prior_class_probabilities_;
    } else {
      return it->second;
    }
  }

  void Sampler::set_prior_class_probabilities(
      const std::string &visitor_id,
      const Vector &probs) {
    check_probabilities(probs);
    prior_class_probabilities_[visitor_id] = probs;
  }

  void Sampler::check_probabilities(const Vector &probs) const {
    if (probs.size()
        != model_->number_of_classes()) {
      std::ostringstream err;
      err << "PoissonFactorModel had " << model_->number_of_classes()
          << " latent classes, but the prior class membership "
          << "probabilities had " << probs.size() << " elements.";
      report_error(err.str());
    }
    if (fabs(probs.sum() - 1.0) > 1e-8) {
      std::ostringstream err;
      err << "Prior class probabilities must sum to 1.  They sum to "
          << probs.sum()
          << ".";
      report_error(err.str());
    }
    int min_pos = probs.imin();
    if (probs[min_pos] < 0.0) {
      std::ostringstream err;
      err << "probs[" << min_pos
          << "] = " << probs[min_pos]
          << ".  All probabilities must be non-negative.";
      report_error(err.str());
    }
  }

  void Sampler::impute_visitors() {
    exposure_counts_ *= 0.0;
    for (auto &visitor_it : model_->visitors()) {
      Ptr<Visitor> &visitor(visitor_it.second);
      Vector prob = prior_class_probabilities(
          visitor->id());
      if (prob.max() > .9999) {
        visitor->set_class_probabilities(prob);
        visitor->set_class_member_indicator(prob.imax());
      } else {
        Vector logprob = log(prob);
        logprob -= model_->sum_of_lambdas();
        for (const auto &it : visitor->sites_visited()) {
          int visit_counts = it.second;
          const Ptr<Site> &site(it.first);
          logprob += visit_counts * site->log_lambda();
          check_logprob(logprob, visit_counts, site);
        }
        prob = logprob.normalize_logprob();
        visitor->set_class_probabilities(prob);
        visitor->set_class_member_indicator(rmulti_mt(rng(), prob));
      }
      ++exposure_counts_[visitor->imputed_class_membership()];
    }
  }

  Vector Sampler::compute_visit_counts(const Site &site) const {
    Vector counts(model()->number_of_classes(), 0.0);
    for (const auto &visitor_it : site.observed_visitors()) {
      const Ptr<Visitor> &visitor(visitor_it.first);
      Int visit_count = visitor_it.second;
      int class_id = visitor->imputed_class_membership();
      counts[class_id] += visit_count;
    }
    return counts;
  }

  void Sampler::check_logprob(const Vector &logprob,
                              int visit_counts,
                              const Ptr<Site> &site) const {
    for (double el : logprob) {
      if (!std::isfinite(el)) {
        std::ostringstream err;
        err << "infinite value in logprob: \n"
            << "logprob = " << logprob << ".\n"
            << "visit_counts = " << visit_counts << "\n"
            << "site->log_lambda() = " << site->log_lambda() << "\n"
            ;
        report_error(err.str());
      }
    }
  }

}  // namespace BOOM
