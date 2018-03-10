// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#ifndef BOOM_MVN_META_ANALYSIS_MVN_POSTERIOR_SAMPLER_HPP_
#define BOOM_MVN_META_ANALYSIS_MVN_POSTERIOR_SAMPLER_HPP_

#include "Models/DoubleModel.hpp"
#include "Models/Mixtures/MvnMetaAnalysisDPMPriorModel.hpp"
#include "Models/MvnGivenSigma.hpp"
#include "Models/WishartModel.hpp"

namespace BOOM {

  class MvnMetaAnalysisDPMPriorSampler : public PosteriorSampler {
   public:
    // The top level of the MvnMetaAnalysisDPMPriorModel is a Multivariate
    // Normal distribution, which is used to model measurement error for each
    // group in the meta-analysis. The prior distribution for true effects in
    // each group is a Dirichlet Process Mixtures of Multivariate Normals.
    //
    // Args:
    //   model: The model whose parameters are to be to be sampled
    //     from their posterior distribution.
    //   mean_base_measure: A conditional MVN model describing the
    //     prior distribution of the normal mean conditional on the
    //     normal variance.  Note that when it is used in other
    //     contexts, an MvnGivenSigma object needs to have Sigma set
    //     somehow.  In this case only the parameters of the
    //     MvnGivenSigma object are used, so there is no need to set
    //     Sigma.
    //   precision_base_measure: A WishartModel describing the
    //     marginal distribution of Siginv, the precision parameter of
    //     each mixture component.
    //   seeding_rng: The RNG to use to set the seed for this
    //     posterior sampler.
    MvnMetaAnalysisDPMPriorSampler(
        MvnMetaAnalysisDPMPriorModel *model,
        const Ptr<MvnGivenSigma> &mean_base_measure,
        const Ptr<WishartModel> &precision_base_measure,
        RNG &seeding_rng = GlobalRng::rng);
    double logpri() const override;
    void draw() override;

   private:
    MvnMetaAnalysisDPMPriorModel *model_;
    Ptr<MvnGivenSigma> mean_base_measure_;
    Ptr<WishartModel> precision_base_measure_;
  };

}  // namespace BOOM

#endif  //  BOOM_MVN_META_ANALYSIS_MVN_POSTERIOR_SAMPLER_HPP_
