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

#ifndef BOOM_DIRICHLET_PROCESS_MVN_MODEL_HPP_
#define BOOM_DIRICHLET_PROCESS_MVN_MODEL_HPP_

#include "Models/MvnModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/VectorModel.hpp"

namespace BOOM {

  // A nonparametric model for multivariate continuous data.  The data
  // are described by a Dirichlet process mixture of normals.  The
  // prior (base measure) is a normal inverse Wishart model.
  class DirichletProcessMvnModel : public VectorModel,
                                   public CompositeParamPolicy,
                                   public IID_DataPolicy<VectorData>,
                                   public PriorPolicy {
   public:
    // Creates a DirichletProcessMvnModel with a single mixture
    // component of dimension dim.
    // Args:
    //   dim:  The dimension of the data to be modeled.
    //   alpha: The content parameter (a real number > 0) of the
    //     Dirichlet process.  Smaller values lead to fewer (larger)
    //     clusters.
    explicit DirichletProcessMvnModel(int dim, double alpha = 1.0);

    DirichletProcessMvnModel(const DirichletProcessMvnModel &rhs);
    DirichletProcessMvnModel *clone() const override;

    // Dimension of the data being modeled.
    int dim() const;

    // Current number of clusters (with at least one data point
    // assigned to them).
    int number_of_clusters() const;

    // Get or set the concentration parameter alpha.
    double alpha() const;
    void set_alpha(double alpha);

    // Assigns the data y to the sufficient statistics of the
    // specified cluster.  If cluster is one past the end of the
    // mixture components a new cluster will be created to store the
    // data.  If cluster is two or more past the end then an exception
    // is thrown.
    void assign_data_to_cluster(const Vector &y, int cluster);

    // If removing the observation from the specified cluster results
    // in an empty cluster then the cluster is removed.
    void remove_data_from_cluster(const Vector &y, int cluster);

    // Change value of data currently used in model.
    void update_cluster(const Vector &old_y, const Vector &new_y, int cluster);

    // The mixture component defining cluster i.  It is an error to call this
    // function with i >= number_of_clusters().
    const MvnModel &cluster(int i) const;

    // Sets the parameters of the specified mixture component to the
    // given values.
    //
    // Args:
    //   cluster: the cluster whose paramters are to be set.  It is an
    //     error to pass a value outside of 0 <= cluster <
    //     number_of_clusters().
    //   mu: A vector of size dim() for the cluster mean.
    //   Siginv: A matrix of dimension dim() x dim() for the cluster
    //     precision (inverse variance).
    void set_component_params(int cluster, const Vector &mu,
                              const SpdMatrix &Siginv);

    // Returns the log likelihood of the data under the current set of
    // mixture components.
    double log_likelihood() const;

    // Calculates probability of single data point.
    double logp(const Vector &x) const override;

    // Simulates data from model.
    // TODO: add ownership of mean and precision base measures to model
    //                so that simulation possible.
    Vector sim(RNG &rng = GlobalRng::rng) const override;

    // Returns:
    //   A vector of length equal to the number of mixture components.
    //   Elements give the number of observations in each component.
    Vector allocation_counts() const;

    // Vector of all cluster indicators.
    const std::vector<int> &cluster_indicators() const {
      return cluster_indicators_;
    }

    // Cluster indicator for i-th observation.
    int cluster_indicators(int i) const {
      return cluster_indicators_[i];
    }

    // Set cluster indicator for i-th observation.
    void set_cluster_indicator(int i, int k) { cluster_indicators_[i] = k; }

    const Matrix cluster_membership_probabilities() const {
      return cluster_membership_probabilities_;
    }

    void set_cluster_membership_probabilities(
        int observation_index, const Vector &probs);

    // Initialize vector of cluster indicators.
    void initialize_cluster_indicators(int s) {
      cluster_indicators_.clear();
      cluster_indicators_.resize(s, -1);
    }

    // Ensure that appropriate space is declared to hold cluster membership
    // probabilities.  Cluster membership probabilities are optional
    void initialize_cluster_membership_probabilities();

   private:
    // Clears the ParamPolicy and re-registers all models with it.
    // This should be called by all constructors.
    void register_models();

    // This function should be called when clusters are being deleted
    // from the model, to ensure that there is always at least one
    // cluster available.
    void ensure_at_least_one_cluster();

    // alpha_ holds the concentration parameter for the Dirichlet process.
    Ptr<UnivParams> alpha_;

    std::vector<Ptr<MvnModel> > mixture_components_;

    // Cluster indicators augmented by sampler (not used in likelihood
    // calculations).
    // cluster_indicators_[i] == -1 means observation i is unassigned.
    std::vector<int> cluster_indicators_;

    // Row i stores the cluster membership probabilities for observation i.  The
    // columns go from 0 to number_of_clusters().  The final column indicates an
    // as-yet unassigned cluster.
    //
    // This data element is optional and the initial value for the matrix may be
    // an empty matrix.
    Matrix cluster_membership_probabilities_;

    // Dimension of the data being modeled.
    int dim_;
  };

}  // namespace BOOM

#endif  //  BOOM_DIRICHLET_PROCESS_MVN_MODEL_HPP_
