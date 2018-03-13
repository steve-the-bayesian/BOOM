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

#include "Models/Glm/PosteriorSamplers/BinomialLogitSamplerRwm.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    class BinomialLogitLogPosterior {
     public:
      BinomialLogitLogPosterior(BinomialLogitModel *model,
                                const Ptr<MvnBase> &prior)
          : m_(model), prior_(prior) {}
      double operator()(const Vector &beta) const {
        return prior_->logp(beta) + m_->log_likelihood(beta, 0, 0);
      }

     private:
      BinomialLogitModel *m_;
      Ptr<MvnBase> prior_;
    };
  }  // namespace

  BinomialLogitSamplerRwm::BinomialLogitSamplerRwm(BinomialLogitModel *model,
                                                   const Ptr<MvnBase> &prior,
                                                   double nu, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        m_(model),
        pri_(prior),
        proposal_(new MvtRwmProposal(SpdMatrix(model->xdim(), 1.0), nu)),
        sam_(BinomialLogitLogPosterior(m_, pri_), proposal_) {}

  void BinomialLogitSamplerRwm::draw() {
    const std::vector<Ptr<BinomialRegressionData> > &data(m_->dat());
    SpdMatrix ivar(pri_->siginv());
    Vector beta(m_->Beta());
    for (int i = 0; i < data.size(); ++i) {
      Ptr<BinomialRegressionData> dp = data[i];
      double eta = beta.dot(dp->x());
      double prob = plogis(eta);
      ivar.add_outer(dp->x(), dp->n() * prob * (1 - prob));
    }

    proposal_->set_ivar(ivar);
    beta = sam_.draw(beta);
    m_->set_Beta(beta);
  }

  double BinomialLogitSamplerRwm::logpri() const {
    return pri_->logp(m_->Beta());
  }
  //======================================================================
}  // namespace BOOM
