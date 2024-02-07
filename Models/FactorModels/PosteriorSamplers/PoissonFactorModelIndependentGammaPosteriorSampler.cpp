/*
  Copyright (C) 2005-2023 Steven L. Scott

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
    using Visitor = PoissonFactor::Visitor;
    using Site = PoissonFactor::Site;
    using Sampler = PoissonFactorModelIndependentGammaPosteriorSampler;
  }

  Sampler::PoissonFactorModelIndependentGammaPosteriorSampler(
      PoissonFactorModel *model,
      const Vector &default_prior_class_probabilities,
      const std::vector<Ptr<GammaModelBase>> &default_intensity_prior,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        default_prior_class_probabilities_(
            default_prior_class_probabilities),
        default_intensity_prior_(default_intensity_prior),
        exposure_counts_(model->number_of_classes(), 0.0),
        sum_of_lambdas_(model->number_of_classes(), negative_infinity()),
        iteration_(0)
  {}

  void Sampler::draw() {
    if (!std::isfinite(sum_of_lambdas_[0])) {
      initialize_sum_of_lambdas();
    }
    impute_visitors();
    draw_site_parameters();
    ++iteration_;
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
        logprob -= sum_of_lambdas_;
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

  void Sampler::draw_site_parameters() {
    sum_of_lambdas_ = 0.0;
    for (auto &site_it : model_->sites()) {
      Ptr<Site> &site(site_it.second);
      const std::vector<Ptr<GammaModelBase>> &site_prior(
          intensity_prior(site->id()));
      Vector counts(site_prior.size());
      for (int i = 0; i < site_prior.size(); ++i) {
        counts[i] = site_prior[i]->a();
      }
      for (const auto &it : site->observed_visitors()) {
        const Ptr<Visitor> &visitor(it.first);
        int visit_count = it.second;
        int mix = visitor->imputed_class_membership();
        counts[mix] += visit_count;
      }
      Vector lambdas(counts.size());
      for (int k = 0; k < counts.size(); ++k) {
        double b = exposure_counts_[k] + site_prior[k]->b();
        if (!std::isfinite(counts[k]) || !std::isfinite(b)) {
          std::ostringstream err;
          err << "site " << site->id()
              << " had an infinite value in position " << k
              << " in either counts "
              << counts[k] << "\n"
              << " exposure_counts_ " << exposure_counts_[k]
              << " or prior: ("
              << site_prior[k]->a()
              << ", " << site_prior[k]->b()
              << ".\n";
          report_error(err.str());
        }
        lambdas[k] = rgamma_mt(rng(), counts[k], b);
        if (lambdas[k] <= 0.0) {
          std::ostringstream err;
          err << "site " << site->id()
              << " generated a zero value for lambda.\n"
              << "counts[" << k << "] = " << counts[k]
              << ", b = " << b << "\n";
          report_error(err.str());
        }
      }
      site->set_lambda(lambdas);
      sum_of_lambdas_ += lambdas;
    }
  }

  Vector Sampler::prior_class_probabilities(
      const std::string &visitor_id) const {
    const auto it = prior_class_probabilities_.find(
        visitor_id);
    if (it == prior_class_probabilities_.end()) {
      return default_prior_class_probabilities_;
    } else {
      return it->second;
    }
  }

  void Sampler::set_prior_class_probabilities(
      const std::string &visitor_id,
      const Vector &probs) {
    if (probs.size() != model_->number_of_classes()) {
      std::ostringstream err;
      err << "Prior class membership probabilities have dimeension "
          << probs.size()
          << " but there are " << model_->number_of_classes()
          << " latent classes.";
      report_error(err.str());
    }
    if (fabs(probs.sum() - 1.0) > 1e-8) {
      report_error("Probabilities must sum to 1.");
    }
    prior_class_probabilities_[visitor_id] = probs;
  }

  void Sampler::set_intensity_prior(
      const std::string &site_id,
      const std::vector<Ptr<GammaModelBase>> &prior) {
    if (prior.size() !=- model_->number_of_classes()) {
      std::ostringstream err;
      err << "The model has " << model_->number_of_classes()
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

  void Sampler::initialize_sum_of_lambdas() {
    sum_of_lambdas_ = 0.0;
    for (const auto &it : model_->sites()) {
      const Ptr<Site> &site(it.second);
      sum_of_lambdas_ += site->lambda();
    }
  }

}  // namespace BOOM
