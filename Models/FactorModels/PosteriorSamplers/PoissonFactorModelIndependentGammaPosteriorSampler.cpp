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

#include "Models/FactorModels/PosteriorSamplers/PoissonFactorModelIndependentGammaPosteriorSampler.hpp"
#include "distributions.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  namespace {
    using Visitor = FactorModels::PoissonVisitor;
    using Site = FactorModels::PoissonSite;
    using Sampler = PoissonFactorModelIndependentGammaPosteriorSampler;
  }

  Sampler::PoissonFactorModelIndependentGammaPosteriorSampler(
      PoissonFactorModel *model,
      const Vector &default_prior_class_probabilities,
      const std::vector<Ptr<GammaModelBase>> &default_intensity_prior,
      RNG &seeding_rng)
      : PoissonFactorPosteriorSamplerBase(
            model,
            default_prior_class_probabilities,
            seeding_rng),
        default_intensity_prior_(default_intensity_prior),
        iteration_(0)
  {}

  void Sampler::draw() {
    impute_visitors();
    draw_site_parameters();
    ++iteration_;
  }

  void Sampler::draw_site_parameters() {
    for (auto &site_it : model()->sites()) {
      Ptr<Site> &site(site_it.second);
      const std::vector<Ptr<GammaModelBase>> &site_prior(
          intensity_prior(site->id()));
      Vector visit_counts = compute_visit_counts(*site);
      for (int i = 0; i < site_prior.size(); ++i) {
        visit_counts[i] += site_prior[i]->a();
      }
      Vector lambdas(visit_counts.size());
      for (int k = 0; k < visit_counts.size(); ++k) {
        double b = exposure_counts()[k] + site_prior[k]->b();
        if (!std::isfinite(visit_counts[k]) || !std::isfinite(b)) {
          std::ostringstream err;
          err << "site " << site->id()
              << " had an infinite value in position " << k
              << " in either visit_counts "
              << visit_counts[k] << "\n"
              << " exposure_counts " << exposure_counts()[k]
              << " or prior: ("
              << site_prior[k]->a()
              << ", " << site_prior[k]->b()
              << ".\n";
          report_error(err.str());
        }
        lambdas[k] = rgamma_mt(rng(), visit_counts[k], b);
        if (lambdas[k] <= 0.0) {
          std::ostringstream err;
          err << "site " << site->id()
              << " generated a zero value for lambda.\n"
              << "visit_counts[" << k << "] = " << visit_counts[k]
              << ", b = " << b << "\n";
          report_error(err.str());
        }
      }
      site->set_lambda(lambdas);
    }
  }

  void Sampler::set_intensity_prior(
      const std::string &site_id,
      const std::vector<Ptr<GammaModelBase>> &prior) {
    if (prior.size() != model()->number_of_classes()) {
      std::ostringstream err;
      err << "The model has " << model()->number_of_classes()
          << " latent classes, but the supplied intensity prior had "
          << prior.size() << " elements.";
      report_error(err.str());
    }
    intensity_parameter_priors_[site_id] = prior;
  }

  const std::vector<Ptr<GammaModelBase>> & Sampler::intensity_prior(
      const std::string &site_id) const {
    auto it = intensity_parameter_priors_.find(site_id);
    if (it == intensity_parameter_priors_.end()) {
      return default_intensity_prior_;
    } else {
      return it->second;
    }
  }

}  // namespace BOOM
