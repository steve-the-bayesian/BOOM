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

#include <cmath>

namespace BOOM {

  OrdinalLogitPosteriorSampler::OrdinalLogitPosteriorSampler(
      OrdinalLogitModel *model,
      const Ptr<MvnBase> &coefficient_prior,
      const Ptr<VectorModel> &cutpoint_prior,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        coefficient_prior_(coefficient_prior),
        cutpoint_prior_(cutpoint_prior),
        complete_data_suf_(coefficient_prior_->dim()),
        coefficient_sampler_(model_, coefficient_prior_, nullptr)
  {
    for (int i = 0; i < model_->cutpoint_vector().size(); ++i) {
      auto cutpoint_i_logpost = [model, cutpoint_prior, i](double x) {
        Vector cutpoints = model->cutpoint_vector();
        cutpoints[i] = x;
        double log_prior = cutpoint_prior->logp(cutpoints);
        if (!std::isfinite(log_prior)) {
          return log_prior;
        }
        Vector bg, dg;
        Matrix bH, dH, cH;
        return log_prior + model->full_loglike(
            model->Beta(), cutpoints, bg, dg, bH, dH, cH, 0, false, false);
      };
      cutpoint_samplers_.push_back(ScalarSliceSampler(
          cutpoint_i_logpost, false, 1.0, &rng()));
    }
  }

  double OrdinalLogitPosteriorSampler::logpri() const {
    return coefficient_prior_->logp(model_->Beta())
        + cutpoint_prior_->logp(model_->cutpoint_vector());
  }

  void OrdinalLogitPosteriorSampler::draw() {
    impute_latent_data();
    draw_beta();
    draw_cutpoints();
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

  void OrdinalLogitPosteriorSampler::draw_cutpoints() {
    for (int i = 0; i < model_->cutpoint_vector().size(); ++i) {
      if (i > 0) {
        cutpoint_samplers_[i].set_lower_limit(model_->cutpoint_vector()[i - 1]);
      } else {
        cutpoint_samplers_[i].set_lower_limit(0);
      }

      if (i + 1 < model_->cutpoint_vector().size()) {
        cutpoint_samplers_[i].set_upper_limit(model_->cutpoint_vector()[i + 1]);
      } else {
        cutpoint_samplers_[i].set_upper_limit(infinity());
      }

      double cutpoint = cutpoint_samplers_[i].draw(
          model_->cutpoint_vector()[i]);
      model_->set_cutpoint(i, cutpoint);
    }
    
  }
  
} // namespace BOOM

