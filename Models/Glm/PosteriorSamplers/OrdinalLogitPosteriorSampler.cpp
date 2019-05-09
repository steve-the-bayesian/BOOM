/*
  Copyright (C) 2005-2019 Steven L. Scott

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

#include "Models/Glm/PosteriorSamplers/OrdinalLogitPosteriorSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  OrdinalLogitPosteriorSampler::OrdinalLogitPosteriorSampler(
      OrdinalLogitModel *model,
      const Ptr<MvnBase> &coefficient_prior,
      const Ptr<VectorModel> &delta_prior,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        coefficient_prior_(coefficient_prior),
        delta_prior_(delta_prior),
        complete_data_suf_(coefficient_prior_->dim()),
        logit_mixture_(
            Vector(9, 0.0),
            Vector{0.88437229872213, 1.16097607474416, 1.28021991084306,
                  1.3592552924727, 1.67589879794907, 2.20287232043947,
                  2.20507148325819, 2.91944313615144, 3.90807611741308},
            Vector{0.038483985581272, 0.13389889791451, 0.0657842076622429,
                  0.105680086433879, 0.345939491553619, 0.0442261124345564,
                  0.193289780660134, 0.068173066865908, 0.00452437089387876}),
        coefficient_sampler_(model_, coefficient_prior_, nullptr)
  {}

  double OrdinalLogitPosteriorSampler::logpri() const {
    return coefficient_prior_->logp(model_->Beta())
        + delta_prior_->logp(model_->cutpoint_vector());
  }

  void OrdinalLogitPosteriorSampler::draw() {
    impute_latent_data();
    draw_beta();
    draw_delta();
  }

  void OrdinalLogitPosteriorSampler::impute_latent_data() {
    complete_data_suf_.clear();
    for (int i = 0; i < model_->dat().size(); ++i) {
      const OrdinalRegressionData &data_point(*model_->dat()[i]);
      double eta = model_->predict(data_point.x());
      int y = data_point.y();
      double upper_cutpoint = model_->upper_cutpoint(y);
      double lower_cutpoint = model_->lower_cutpoint(y);
      double z = imputer_.impute(rng(), eta, lower_cutpoint, upper_cutpoint);
      double mu, sigsq;
      logit_mixture_.unmix(rng(), z, &mu, &sigsq);
      complete_data_suf_.add_data(data_point.x(), z, 1.0 / sigsq);
    }
  }

  void OrdinalLogitPosteriorSampler::draw_beta() {
    coefficient_sampler_.draw_beta(rng(), complete_data_suf_);
  }

  void OrdinalLogitPosteriorSampler::draw_delta() {
    
  }
  
} // namespace BOOM

