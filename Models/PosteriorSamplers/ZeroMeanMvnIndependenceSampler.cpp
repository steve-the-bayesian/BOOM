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

#include "Models/PosteriorSamplers/ZeroMeanMvnIndependenceSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "distributions/trun_gamma.hpp"

namespace BOOM {
  typedef ZeroMeanMvnIndependenceSampler ZMMI;

  ZMMI::ZeroMeanMvnIndependenceSampler(ZeroMeanMvnModel *model,
                                       const Ptr<GammaModelBase> &prior,
                                       int which_variable, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        m_(model),
        prior_(prior),
        which_variable_(which_variable),
        sampler_(prior_) {}

  ZMMI::ZeroMeanMvnIndependenceSampler(ZeroMeanMvnModel *model, double prior_df,
                                       double prior_sigma_guess,
                                       int which_variable, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        m_(model),
        prior_(new GammaModel(prior_df / 2,
                              pow(prior_sigma_guess, 2) * prior_df / 2)),
        which_variable_(which_variable),
        sampler_(prior_) {}

  ZMMI *ZMMI::clone_to_new_host(Model *new_host) const {
    ZMMI *ans = new ZMMI(
        dynamic_cast<ZeroMeanMvnModel *>(new_host),
        prior_->clone(),
        which_variable_,
        rng());
    ans->set_sigma_upper_limit(sampler_.sigma_max());
    return ans;
  }

  void ZMMI::set_sigma_upper_limit(double max_sigma) {
    sampler_.set_sigma_max(max_sigma);
  }

  void ZMMI::draw() {
    SpdMatrix siginv = m_->siginv();
    int i = which_variable_;
    double df = m_->suf()->n();
    SpdMatrix sumsq = m_->suf()->center_sumsq(m_->mu());
    siginv(i, i) = 1.0 / sampler_.draw(rng(), df, sumsq(i, i));
    m_->set_siginv(siginv);
  }

  double ZMMI::logpri() const {
    int i = which_variable_;
    double siginv = m_->siginv()(i, i);
    return sampler_.log_prior(1.0 / siginv);
  }

  //======================================================================
  typedef ZeroMeanMvnCompositeIndependenceSampler ZMMCIS;
  ZMMCIS::ZeroMeanMvnCompositeIndependenceSampler(
      ZeroMeanMvnModel *model,
      const std::vector<Ptr<GammaModelBase> > &siginv_priors,
      const Vector &sigma_upper_truncation_points, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), model_(model), priors_(siginv_priors) {
    if (model_->dim() != priors_.size()) {
      report_error(
          "'model' and 'siginv_priors' arguments are not compatible "
          "in "
          "ZeroMeanMvnCompositeIndependenceSampler constructor.");
    }

    if (model_->dim() != sigma_upper_truncation_points.size()) {
      report_error(
          "'model' and 'sigma_upper_truncation_points' arguments "
          "are not compatible in "
          "ZeroMeanMvnCompositeIndependenceSampler constructor.");
    }

    for (int i = 0; i < sigma_upper_truncation_points.size(); ++i) {
      if (sigma_upper_truncation_points[i] < 0) {
        ostringstream err;
        err << "Element " << i << " (counting from 0) of "
            << "sigma_upper_truncation_points is negative in "
            << "ZeroMeanMvnCompositeIndependenceSampler constructor." << endl
            << sigma_upper_truncation_points << endl;
        report_error(err.str());
      }
    }

    for (int i = 0; i < priors_.size(); ++i) {
      GenericGaussianVarianceSampler sampler(priors_[i],
                                             sigma_upper_truncation_points[i]);
      samplers_.push_back(sampler);
    }
  }

  void ZMMCIS::draw() {
    SpdMatrix Sigma = model_->Sigma();
    SpdMatrix sumsq = model_->suf()->center_sumsq(model_->mu());
    for (int i = 0; i < model_->dim(); ++i) {
      Sigma(i, i) = samplers_[i].draw(rng(), model_->suf()->n(), sumsq(i, i));
    }
    model_->set_Sigma(Sigma);
  }

  double ZMMCIS::logpri() const {
    const SpdMatrix &Sigma(model_->Sigma());
    double ans = 0;
    for (int i = 0; i < Sigma.nrow(); ++i) {
      if (samplers_[i].sigma_max() > 0) {
        ans += samplers_[i].log_prior(Sigma(i, i));
      }
    }
    return ans;
  }

}  // namespace BOOM
