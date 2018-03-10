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

#ifndef BOOM_DIRICHLET_PROCESS_MIXTURE_HPP_
#define BOOM_DIRICHLET_PROCESS_MIXTURE_HPP_

#include "Models/CategoricalData.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/PosteriorSamplers/HierarchicalPosteriorSampler.hpp"

namespace BOOM {

  namespace SplitMerge {
    class Proposal;
  }  // namespace SplitMerge

  // A Dirichlet process mixture model is an exchangeable model for a sequence
  // of data defined by a "concentration parameter" alpha, a base distribution
  // F, and a parametric family g(y | theta).  The model is a stochastic process
  // indexed by observation number (so it is a time series, in a highly
  // non-traditional sort of way).
  //
  // At time 0 a parameter theta[0] is drawn from F, and an observation y is
  // drawn from g(y | theta[0]).  At time 1 a parameter theta[1] is drawn.
  // Theta[1] is either a copy of theta[0], or it is a new, independent draw
  // from F.  The probability that theta[1] == theta[0] is 1 / (1 + alpha).  At
  // time t there will be K unique values of theta.  Denote these by phi[0],
  // ..., phi[K-1].  The change in notation is because theta's are indexed by
  // observation number, and phi's are indexed by cluster number.  This notation
  // is from Neal (2000, JCGS).
  //
  // At time t+1, the probability that theta[t+1] == phi[k] is n[k] / (t +
  // alpha), where n[k] is the number of observations from cluster k, and the
  // probability that theta[t+1] is a new, fresh independent draw from F is
  // alpha / (t + alpha).
  //
  // The sequence of theta's produced by the process above is a Dirichlet
  // process (DP), and the sequence of y's is a Dirichlet process mixture model
  // (DPMM).
  //
  // When compared to a standard finite mixture model, the mixing weights in the
  // DPMM are a deterministic function of the (unobserved) cluster indicators.
  // However, as with any exchangeable model, you can also write the DPMM in
  // terms of a model where the observations are independent, conditional on a
  // latent variable with an approprate distribution.  In the case of the
  // Dirichlet process (and thus the DPMM), the latent parameter is a set of
  // mixing weights w (an infinite sequence):
  //
  //       p(theta) = \sum_k w[k] I(theta = phi[k]).
  //
  // Here the phi[k] are a set of IID draws from F, and the w's follow a "stick
  // breaking process" formed by a set of stick fractions v[t] ~ Beta(1, alpha).
  //        w[0] = v[0]
  //        w[1] = v[1] * (1 - w[0]);
  //   ...  w[t] = v[t] * (1 - w[0] - w[1]) ...
  //
  // Each stick fraction v[t] is the fraction of the interval (0, 1) not
  // consumed by previous mixing weights.
  //
  // This class represents the DPMM using a set of mixture components and stick
  // fractions.  Some posterior samplers don't need the stick fractions.
  class DirichletProcessMixtureModel : public CompositeParamPolicy,
                                       public PriorPolicy {
   public:
    typedef DirichletProcessMixtureComponent DpMixtureComponent;

    // Args:
    //   mixture_component_prototype: A model that can be cloned to produce a
    //     new mixture component.  Each data point from this model is from an
    //     instance of a model from this family, with potentially different
    //     parameters.
    //   prior: The "base measure".  When the process needs a new mixture
    //     component, it creates a new element from mixture_comopnent_prototype,
    //     and sets its parameters using a draw from 'prior'.
    //   concentration_parameter: The scalar that determines when a new mixture
    //     component is needed.  Larger values tend to yield more components.
    //     If a is the concentration parameter and n data points have been
    //     produced so far then the probability that a new component is
    //     generated is a / (a + n - 1).
    DirichletProcessMixtureModel(
        const Ptr<DpMixtureComponent> &mixture_component_prototype,
        const Ptr<HierarchicalPosteriorSampler> &base_distribution,
        const Ptr<UnivParams> &concentration_parameter);

    DirichletProcessMixtureModel *clone() const override {
      return new DirichletProcessMixtureModel(*this);
    }

    virtual bool conjugate() const { return false; }
    int number_of_components() const { return mixture_components_.size(); }
    int number_of_observations() const { return dat().size(); }

    virtual DirichletProcessMixtureComponent *component(int i) {
      return mixture_components_[i].get();
    }
    virtual const DirichletProcessMixtureComponent *component(int i) const {
      return mixture_components_[i].get();
    }

    virtual HierarchicalPosteriorSampler *base_distribution() {
      return base_distribution_.get();
    }

    double concentration_parameter() const {
      return concentration_parameter_->value();
    }

    double log_concentration_parameter() const {
      return log_concentration_parameter_;
    }

    // There is one mixing weight for each mixture component, plus one at the
    // end for "all future components".
    const Vector &mixing_weights() const { return mixing_weights_; }
    const Vector &stick_fractions() const { return stick_fractions_; }

    // Set the vector of stick fractions used to compute the mixing weights.
    // Requires stick_fractions.size() == number_of_components().
    void set_stick_fractions(const Vector &stick_fractions);

    // The number of observations that have been assigned to the specified
    // cluster.
    int cluster_count(int cluster) const {
      return mixture_components_[cluster]->number_of_observations();
    }

    // Return the cluster number of the specified observation.  If the specified
    // observation is unassigned return -1.
    int cluster_indicator(int observation) const;

    // Fill the argument 'indicators' with the cluster indicators for the
    // corresponding data points.  The argument will be resized to match the
    // number of observations.
    void cluster_indicators(std::vector<int> &indicators) const;

    // Data policy related functions are overridden so that map of mixture
    // indicators can be maintained.
    void add_data(const Ptr<Data> &dp) override;
    void clear_data() override;
    void combine_data(const Model &other_model, bool just_suf = true) override;
    const std::vector<Ptr<Data>> &dat() const { return data_; }

    // Args:
    //   proposal:  The proposed split or merge.
    void accept_split_merge_proposal(const SplitMerge::Proposal &proposal);

    // Args:
    //   dp:  The data point to be assigned to a cluster.
    //   cluster:  The index of the cluster that will take ownership of dp.
    //   rng: If 'cluster' is one past the end of the vector of mixture
    //     components, a new component is added.  The random number generator is
    //     needed to simulate a mixing weight for the new component.
    void assign_data_to_cluster(const Ptr<Data> &dp, int cluster, RNG &rng);

    // If 'dp' is currently in a cluster, remove it.  Set its mixture indicator
    // accordingly.  If the remove_empty_cluster flag is set then remove the
    // cluster to which dp belonged if dp was the only data point.
    void remove_data_from_cluster(const Ptr<Data> &dp,
                                  bool remove_empty_cluster = true);

    // Adds an empty cluster to the back of the vector of mixture components.
    // The cluster's parameters are simulated from the prior.  A stick fraction
    // for the new component is simulated from the prior, and stick_fractions_
    // and mixing_weights_ are adjusted accordingly.
    virtual void add_empty_cluster(RNG &rng);

    // Removes the empty cluster from the vector of mixture components,
    // adjusting mixing weights and stick fractions.
    virtual void remove_empty_cluster(
        const Ptr<DirichletProcessMixtureComponent> &component,
        bool adjust_mixing_weights);

    // Removes all the empty clusters from the model.
    virtual void remove_all_empty_clusters();

    // Moves the cluster at 'from' to position 'to', shifting intervening
    // clusters as needed.
    virtual void shift_cluster(int from, int to);

    // Recompute the vector of mixing weights from the vector of stick
    // fractions, and vice versa.
    void compute_mixing_weights();
    void compute_stick_fractions_from_mixing_weights();

    // Returns the "stick breaking density" of a vector of weights built from a
    // stick breaking process.  A stick breaking process produces an infinite
    // sequence of w's by successively breaking off a fraction of the remaining
    // portion of the unit interval.
    //
    // w[1] = v[1].
    // w[2] = v[2] * (1 - v[1]).
    // w[3] = v[3] * (1 - v[1]) * (1 - v[2]) = v[3] * (1 - w[1] - w[2]).
    // ...
    //
    // For the Dirichlet process, the v[i]'s are independent Beta(1, alpha)
    // random variables.
    static double dstick(const Vector &weights, double alpha, bool logscale);

   protected:
    // Methods in this section are needed to implement the parallel data
    // structures between this class and ConjugateDirichletProcessMixtureModel.

    // Used to implment add_empty_cluster();
    virtual void repopulate_spare_mixture_components();

    // Remove the final element from the vector of spare mixture components.
    virtual void pop_spare_component_stack();

    // Sets the mixture component index of the argument to -1, and adds it to
    // the back of the spare_mixture_components_ stack.
    void unassign_component_and_add_to_spares(
        const Ptr<DirichletProcessMixtureComponent> &component);

    // Add component to the set of mixture components.
    // Args:
    //   component:  The component to add.
    //   rng: The random number generator to use when assigning a mixing weight
    //     to the new component.
    void assign_and_add_mixture_component(
        const Ptr<DpMixtureComponent> &component, RNG &rng);

    // The desired size of the spare_mixture_components_ buffer.  If the buffer
    // is empty it will be repopulated with this many elements.  If it gets to
    // more than twice this size, the surplus elements will be removed.
    int spare_mixture_component_target_buffer_size() const {
      return spare_mixture_component_target_buffer_size_;
    }

    // Replace one cluster with another, preserving the parallel data structures
    // in the conjugate model, cluster position markers, and cluster indicators.
    virtual void replace_cluster(
        const Ptr<DpMixtureComponent> &component_to_replace,
        const Ptr<DpMixtureComponent> &new_component);

    // Insert a new mixture component at the given position.  Indices for all
    // mixture components to the right of position are set appropriately.
    virtual void insert_cluster(const Ptr<DpMixtureComponent> &component,
                                int index);

   private:
    std::vector<Ptr<DpMixtureComponent>> mixture_components_;

    // Hold the mixture_component_prototype_ separately from the remaining
    // mixture components so that it does not get accidentally deleted.
    Ptr<DpMixtureComponent> mixture_component_prototype_;

    // This is the 'prior distribution' for each of the models.  We'd like to
    // only have one of these.  It may contain more than one Model object to
    // represent the prior (e.g. Normal inverse Wishart).
    Ptr<HierarchicalPosteriorSampler> base_distribution_;

    Ptr<UnivParams> concentration_parameter_;
    double log_concentration_parameter_;
    // Set an observer on the concentration_parameter_ so that
    // log_concentration_parameter_ is updated whenever concentration_parameter_
    // changes.
    void observe_concentration_parameter();

    // Keeps track of the cluster to which each data point has been assigned.
    // An unassigned data point has cluster indicator -1.  This is the inverse
    // mapping of each mixture component's dat() method.
    std::map<Ptr<Data>, Ptr<DpMixtureComponent>> cluster_indicators_;

    // The data for this class which would normally be managed by a policy.  If
    // similar model classes are built later, consider creating a shared policy
    // class.
    std::vector<Ptr<Data>> data_;

    // Used to determine the mixing weights for the infinte mixture.  Element i
    // of stick_fractions_ gives the fraction of the remaining probability (not
    // used by components prior to i) allocated to mixture component i.
    Vector stick_fractions_;
    Vector mixing_weights_;

    // Keep a few spare mixture components on hand to prevent unnecessary
    // allocations.
    std::vector<Ptr<DpMixtureComponent>> spare_mixture_components_;
    const int spare_mixture_component_target_buffer_size_;
  };

  // A Dirichlet process mixture is "conjugate" if the base measure F is a
  // conjugate prior to the data distributions g.  Conjugacy carries two
  // benefits:
  //
  // 1) Posterior sampling is easier, because mixture component parameters can
  //    be updated conditional on cluster assignments.
  //
  // 2) The parameters can be analytically integrated from the posterior
  //    distribution, which means the state of the model is simply the cluster
  //    assignments, so e.g. a "collapsed Gibbs sampler" can be used.
  //
  // Implementing a conjugate model means the private data managed by this class
  // must be kept in parallel with the underlying DirichletProcessMixtureModel,
  // which means lots of virtual function overrides.
  class ConjugateDirichletProcessMixtureModel
      : public DirichletProcessMixtureModel {
   public:
    typedef ConjugateDirichletProcessMixtureComponent
        ConjugateDpMixtureComponent;
    typedef DirichletProcessMixtureComponent DpMixtureComponent;

    ConjugateDirichletProcessMixtureModel(
        const Ptr<ConjugateDpMixtureComponent> &mixture_component_prototype,
        const Ptr<ConjugateHierarchicalPosteriorSampler> &base_distribution,
        const Ptr<UnivParams> &concentration_parameter);

    ConjugateDirichletProcessMixtureModel *clone() const override {
      return new ConjugateDirichletProcessMixtureModel(*this);
    }

    bool conjugate() const override { return true; }
    ConjugateDpMixtureComponent *component(int i) override {
      return conjugate_mixture_components_[i].get();
    }

    const ConjugateDpMixtureComponent *component(int i) const override {
      return conjugate_mixture_components_[i].get();
    }

    ConjugateHierarchicalPosteriorSampler *base_distribution() override {
      return conjugate_base_distribution_.get();
    }

    double log_marginal_density(const Ptr<Data> &data_point,
                                int which_component) const;

    void add_empty_cluster(RNG &rng) override;

    void remove_empty_cluster(const Ptr<DpMixtureComponent> &component,
                              bool adjust_mixing_weights) override;

    void replace_cluster(const Ptr<DpMixtureComponent> &component_to_replace,
                         const Ptr<DpMixtureComponent> &new_component) override;

    void insert_cluster(const Ptr<DpMixtureComponent> &component,
                        int index) override;
    void shift_cluster(int from, int to) override;

   private:
    // Used to implment add_empty_cluster();
    void repopulate_spare_mixture_components() override;
    void pop_spare_component_stack() override;

    // Each pointer in this class is mirrored by a parent-pointer to the same
    // object in the parent class.
    std::vector<Ptr<ConjugateDpMixtureComponent>> conjugate_mixture_components_;
    Ptr<ConjugateDpMixtureComponent> conjugate_mixture_component_prototype_;
    std::vector<Ptr<ConjugateDpMixtureComponent>> spare_conjugate_components_;
    Ptr<ConjugateHierarchicalPosteriorSampler> conjugate_base_distribution_;
  };

}  // namespace BOOM

#endif  //  BOOM_DIRICHLET_PROCESS_MIXTURE_HPP_
