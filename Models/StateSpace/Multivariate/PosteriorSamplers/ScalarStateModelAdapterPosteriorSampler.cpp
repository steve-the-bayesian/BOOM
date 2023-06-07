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

#include "Models/StateSpace/Multivariate/PosteriorSamplers/ScalarStateModelAdapterPosteriorSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using CIS = CiScalarStateAdapterPosteriorSampler;
    using MODEL = ConditionallyIndependentScalarStateModelMultivariateAdapter;
  }  // namespace

  CIS::CiScalarStateAdapterPosteriorSampler(MODEL *model, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model)
  {}

  double CIS::logpri() const {
    double ans = 0;
    for (int s = 0; s < model_->number_of_state_models(); ++s) {
      ans += model_->state_model(s)->logpri();
    }
    return ans;
  }

  void CIS::draw() {
    for (int s = 0; s < model_->number_of_state_models(); ++s) {
      model_->state_model(s)->sample_posterior();
    }
    Vector sigsq = model_->host()->observation_variance_parameter_values();
    Vector slopes(model_->nseries());
    for (int i = 0; i < model_->nseries(); ++i) {
      const ScalarRegressionSuf &suf(model_->sufficient_statistics(i));
      double prior_precision = 1;
      double posterior_precision = (prior_precision + suf.xtx()) / sigsq[i];
      double posterior_mean = (suf.xty() / sigsq[i]) / posterior_precision;
      double posterior_sd = sqrt(1.0 / posterior_precision);
      double slope = rnorm_mt(rng(), posterior_mean, posterior_sd);
      slopes[i] = slope;
    }
    model_->set_observation_coefficient_slopes(slopes);
  }

}  // namespace BOOM
