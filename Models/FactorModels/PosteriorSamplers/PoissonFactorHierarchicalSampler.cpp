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

#include "Models/FactorModels/PosteriorSamplers/PoissonFactorHierarchicalSampler.hpp"

namespace BOOM {

  namespace {
    using Sampler = PoissonFactorHierarchialSampler;
    using Site = PoissonFactor::Site;
    using Visitor = PoissonFactor::Visitor;
  }  // namespace

  Sampler::PoissonFactorHierarchialSampler(
      PoissonFactorModel *model,
      const Vector &default_prior_class_probabilities,
      const Ptr<MvnModel> &profile_hyperprior,
      RNG &seeding_rng)
      : PoissonFactorPosteriorSamplerBase(
            model,
            default_prior_class_probabilities,
            seeding_rng),
        profile_hyperprior_(profile_hyperprior)
  {
    check_dimension(profile_hyperprior);
  }

  void Sampler::draw() {
    impute_visitors();
    draw_site_parameters();
    draw_hyperparameters();
  }

  void Sampler::draw_site_parameters() {
    for (auto &site_it : model()->sites()) {
      Ptr<Site> &site(site_it.second);
      Vector counts = compute_visit_counts(*site);
      if (counts.min() >= 10) {
        draw_site_parameters_MH(*site);
      } else {
        double total_count = sum(counts);
        const Vector &lambda = site->lambda();

        double total_intensity = draw_total_intensity(
            total_count, *site);

        Vector profile = lambda / sum(lambda);
        MultinomialLogitTransform mlogit;
        Vector mlogit_profile = mlogit.to_logits(profile, false);
      }
    }
  }

  class LambdaLogpost {
   public:
    LambdaLogpost(const Site &site, int i)
        site_(site),
        pos_(i)
    {}

    double operator()(double lambda) const {


    }

   private:
    Ptr<Site> site_;
    int pos_;
  };


  // A Metropolis-Hastings update to be used when all categories are observed at
  // or above a threshold that makes it likely that the MH update will be
  // accepted.
  //
  // Args:
  //   counts:  The observation counts for each latent cateogory.
  //   site:  The Site to be updated.
  void draw_site_parameters_MH(const Vector &counts, Ptr<Site> &site) {
    Vector lambda = site.lambda();
    for (size_t i = 0; i < lambda_candidate.size(); ++i) {
      LambdaLogpost logpost(site, i);
      double lambda_candidate = rgamma_mt(counts[i], exposure_counts()[i]);
      double log_alpha_numerator = logpost(lambda_candidate)
          - dgamma(lambda_candidate, counts[i], exposure_counts()[i]);
      double log_alpha_denominator = logpost(lambda[i])
          - dgamma(lambda[i], counts[i], exposure_counts()[i]);
      double log_alpha = log_alpha_numerator - log_alpha_denominator;
      double logu;
      do {
        logu = log(runif_mt(rng, 0, 1));
      } while (!std::isfinite(logu));
      if (logu < log_alpha) {
        lambda[i] = lambda_candidate;
      } else {
        // Do  nothing.
      }
    }
    site.set_lambda(lambda);
  }

  // The total intensity parameter is the sum of a bunch of Poisson
  double Sampler::draw_total_intensity(double total_count, double total_exposure) {
    return rgamma_mt(rng(), total_count, total_exposure);
  }

  void Sampler::draw_profile(const Vector &profile, const Vector &counts) {
  }

  void Sampler::check_dimension(const Ptr<MvnModel> &profile_hyperprior) const {
    if (profile_hyperprior->dim() + 1 != model()->number_of_classes()) {
      std::ostringstream err;
      err << "The dimension of the profile hyperprior was "
          << profile_hyperprior->dim()
          << ".  This should be one less than the number of classes, "
          << model()->number_of_classes()
          << ".";
      report_error(err.str());
    }
  }


}  // namespace BOOM
