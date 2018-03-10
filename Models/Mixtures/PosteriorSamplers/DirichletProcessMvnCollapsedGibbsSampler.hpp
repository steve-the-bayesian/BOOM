// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#ifndef BOOM_DIRICHLET_PROCESS_MVN_COLLAPSED_GIBBS_SAMPLER_HPP_
#define BOOM_DIRICHLET_PROCESS_MVN_COLLAPSED_GIBBS_SAMPLER_HPP_

#include "Models/Mixtures/DirichletProcessMvnModel.hpp"

#include "Models/MvnGivenSigma.hpp"
#include "Models/PosteriorSamplers/MvnConjSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/WishartModel.hpp"

namespace BOOM {

  // A Posterior sampler for a Dirichlet process model that describes
  // observation vector y[i] as a mixture of normals, with a
  // normal-inverse-Wishart prior distribution.
  class DirichletProcessMvnCollapsedGibbsSampler : public PosteriorSampler {
   public:
    // Args:
    //   model: The model for which posterior samples are desired.
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
    DirichletProcessMvnCollapsedGibbsSampler(
        DirichletProcessMvnModel *model,
        const Ptr<MvnGivenSigma> &mean_base_measure,
        const Ptr<WishartModel> &precision_base_measure,
        RNG &seeding_rng = GlobalRng::rng);

    // Note that logpri is a required overload, but it doesn't really
    // make sense here.  Calling it results in an exception.
    double logpri() const override;

    void draw() override;

    // Sample the cluster membership indicators, and allocate the data
    // in the model object to different clusters.
    void draw_cluster_membership_indicators();

    // Draws the model parameters given the cluster indicators.
    void draw_parameters();

    // Compute the discrete probability distribution of cluster
    // membership for observation y, which is currently unassigned,
    // conditional on the cluster membership of the other data points.
    Vector cluster_membership_probability(const Vector &y);

    // Returns the log marginal density of y given a cluster of other
    // observations summarized by suf.  The marginal density of y is
    // the integral of p(y | theta) * p(theta | suf) with respect to
    // theta.  For the math, see Murphy (Machine Learning: A
    // probabilistic perspective) page 161 (eq: 5.29).
    double log_marginal_density(const Vector &y, const MvnSuf &suf) const;

    // Assign the (currently unassigned) observation to the given
    // cluster, recording the assignment in both the held model as
    // well as the set of cluster indicators.
    // Args:
    //   y:  The data to be assigned.
    //   cluster:  The cluster indicator.
    void assign_data_to_cluster(const Vector &y, int cluster);

    // Remove the observation y from the specified cluster.  It is the
    // caller's responsibility to ensure that y was previously
    // assigned to the cluster in the first place.
    void remove_data_from_cluster(const Vector &y, int cluster);

   private:
    DirichletProcessMvnModel *model_;
    Ptr<MvnGivenSigma> mean_base_measure_;
    Ptr<WishartModel> precision_base_measure_;

    MvnSuf empty_suf_;

    mutable NormalInverseWishart::NormalInverseWishartParameters prior_;
    mutable NormalInverseWishart::NormalInverseWishartParameters posterior_;
  };

}  // namespace BOOM

#endif  //  BOOM_DIRICHLET_PROCESS_MVN_COLLAPSED_GIBBS_SAMPLER_HPP_
