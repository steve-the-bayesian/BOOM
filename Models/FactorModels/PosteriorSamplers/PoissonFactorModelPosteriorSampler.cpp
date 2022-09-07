/*
  Copyright (C) 2005-2022 Steven L. Scott

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

#include "Models/FactorModels/PosteriorSamplers/PoissonFactorModelPosteriorSampler.hpp"
#include "distributions.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  namespace {
    using Visitor = PoissonFactor::Visitor;
    using Site = PoissonFactor::Site;
  }

  PoissonFactorModelPosteriorSampler::PoissonFactorModelPosteriorSampler(
      PoissonFactorModel *model,
      const Vector &prior_class_membership_probabilities,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        prior_class_membership_probabilities_(
            prior_class_membership_probabilities)
  {}

  void PoissonFactorModelPosteriorSampler::draw() {
    impute_visitors();
    draw_site_parameters();
  }

  void PoissonFactorModelPosteriorSampler::impute_visitors() {
    Vector log_prior = log(prior_class_membership_probabilities());

    for (Ptr<Visitor> &visitor : model_->visitors()) {
      Vector logprob = log_prior;
      logprob -= model_->sum_of_lambdas();
      for (const auto &it : visitor->sites_visited()) {
        int site_visits = it.second;
        const Ptr<Site> &site(it.first);
        logprob += site_visits * site->log_lambda();
        for (double el : logprob) {
          if (!std::isfinite(el)) {
            report_error("inf in logprob");
          }
        }
      }
      Vector prob = logprob.normalize_logprob();
      visitor->set_class_probabilities(prob);
      visitor->set_class_member_indicator(rmulti_mt(rng(), prob));
    }
  }

  void PoissonFactorModelPosteriorSampler::draw_site_parameters() {
    Vector sum_of_lambdas(model_->number_of_classes());
    for (auto &site : model_->sites()) {
      Vector counts = site->prior_a();
      Vector exposures = site->prior_b();
      for (const auto &it : site->observed_visitors()) {
        const Ptr<Visitor> &visitor(it.first);
        int visit_count = it.second;
        int mix = visitor->imputed_class_membership();
        counts[mix] += visit_count;
        exposures[mix] += 1;
      }

      Vector lambdas(counts.size());
      for (int k = 0; k < counts.size(); ++k) {
        lambdas[k] = rgamma_mt(rng(), counts[k], exposures[k]);
      }
      site->set_lambda(lambdas);
      sum_of_lambdas += lambdas;
    }
    model_->set_sum_of_lambdas(sum_of_lambdas);
  }


}  // namespace BOOM
