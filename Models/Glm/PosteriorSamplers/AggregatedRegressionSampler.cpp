// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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
#include "Models/Glm/PosteriorSamplers/AggregatedRegressionSampler.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
namespace BOOM {

  namespace {
    inline void check_positive(double value, const char *name) {
      if (!(value > 0)) {
        ostringstream err;
        err << name << " was " << value << " (must be postive) " << endl;
        report_error(err.str());
      }
    }
  }  // namespace

  AggregatedRegressionSampler::AggregatedRegressionSampler(
      AggregatedRegressionModel *model, double prior_sigma_nobs,
      double prior_sigma_guess, double prior_beta_nobs,
      double prior_diagonal_shrinkage,
      double prior_variable_inclusion_probability, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        sam_(new BregVsSampler(model_->regression_model(), prior_sigma_nobs,
                               prior_sigma_guess, prior_beta_nobs,
                               prior_diagonal_shrinkage,
                               prior_variable_inclusion_probability)) {
    check_positive(prior_sigma_guess, "prior_sigma_guess");
    check_positive(prior_sigma_nobs, "prior_sigma_nobs");
    check_positive(prior_beta_nobs, "prior_beta_nobs");
    check_positive(prior_diagonal_shrinkage, "prior_diagonal_shrinkage");
    check_positive(prior_variable_inclusion_probability,
                   "prior_variable_inclusion_probability");
    model_->set_method(sam_);
  }

  void AggregatedRegressionSampler::draw() {
    model_->distribute_group_totals();
    sam_->draw();
  }

  double AggregatedRegressionSampler::logpri() const { return sam_->logpri(); }
}  // namespace BOOM
