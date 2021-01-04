// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#ifndef BOOM_DIRICHLET_PROCESS_COLLAPSED_GIBBS_SAMPLER_HPP_
#define BOOM_DIRICHLET_PROCESS_COLLAPSED_GIBBS_SAMPLER_HPP_

#include "Models/Mixtures/DirichletProcessMixture.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  //===========================================================================
  // If the model family and base measure are conjugate then the model
  // parameters and DP mixing weights (sticks) can be integrated out, which
  // means that the collapsed Gibbs sampler can be used.
  //
  // NOTE: This algorithm is fast in terms of mixing time, but computationally
  // slow because individual data points are constantly being added and removed
  // from the component models.
  class DirichletProcessCollapsedGibbsSampler : public PosteriorSampler {
   public:
    explicit DirichletProcessCollapsedGibbsSampler(
        ConjugateDirichletProcessMixtureModel *model,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    void collapsed_gibbs_update();
    void conjugate_split_merge_update();

    // Simulate cluster membership indicators using the marginal cluster
    // membership probability
    void draw_marginal_cluster_membership_indicators();

    void draw_parameters_given_cluster_membership();

    // Compute the probability that a data point belongs to each mixture
    // component, given all other data, but integrating out model parameters.
    //
    // Args:
    //   dp: An element of data, which is not currently assigned to any cluster.
    //
    // Returns:
    //   A vector containing a discrete probability distribution describing the
    //   cluster to which d belongs.  The vector is one size larger than the
    //   current number of clusters, with the final element corresponding to a
    //   new cluster.
    //
    //   The probability is 'marginal' because both parameters of the mixture
    //   components and the mixing weights in the Dirichlet process are
    //   integrated out.
    const Vector &marginal_cluster_membership_probabilities(
        const Ptr<Data> &dp);

   private:
    ConjugateDirichletProcessMixtureModel *model_;

    // Workspace for computing marginal_cluster_membership_probabilities.
    Vector cluster_membership_probabilities_;
  };

}  // namespace BOOM

#endif  // BOOM_DIRICHLET_PROCESS_COLLAPSED_GIBBS_SAMPLER_HPP_
