// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include "Models/Glm/PosteriorSamplers/PoissonRegressionRwmSampler.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  PoissonRegressionRwmSampler::PoissonRegressionRwmSampler(
      PoissonRegressionModel *model, const Ptr<MvnBase> &prior,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), model_(model), prior_(prior) {
    if (model_->xdim() != prior_->dim()) {
      report_error(
          "Prior and model are incompatible in "
          "PoissonRegressionRwmSampler constructor.");
    }
  }

  void PoissonRegressionRwmSampler::draw() {
    const std::vector<Ptr<PoissonRegressionData> > &data(model_->dat());
    int nobs = data.size();
    SpdMatrix proposal_information = prior_->siginv();

    for (int i = 0; i < nobs; ++i) {
      const PoissonRegressionData &d(*data[i]);
      double eta = model_->predict(d.x());
      proposal_information.add_outer(d.x(), d.exposure() * exp(eta), false);
    }

    proposal_information.reflect();
    const Vector &beta = model_->Beta();

    Vector candidate = rmvt_ivar_mt(rng(), beta, proposal_information, 2);
    double logp_cand =
        prior_->logp(candidate) + model_->log_likelihood(candidate);
    double logp_original = prior_->logp(beta) + model_->log_likelihood(beta);

    if (log(runif_mt(rng())) < logp_cand - logp_original) {
      model_->set_Beta(candidate);
    }
  }

  double PoissonRegressionRwmSampler::logpri() const {
    return prior_->logp(model_->Beta());
  }

}  // namespace BOOM
