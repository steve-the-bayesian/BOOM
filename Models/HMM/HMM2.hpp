// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#ifndef BOOM_HMM_HPP
#define BOOM_HMM_HPP

#include <vector>
#include "uint.hpp"
#include "Models/DataTypes.hpp"
#include "Models/EmMixtureComponent.hpp"
#include "Models/MarkovModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/TimeSeries/TimeSeriesDataPolicy.hpp"
#include "cpputil/ThreadTools.hpp"

namespace BOOM {

  class HmmFilter;
  class HmmEmFilter;
  class HmmDataImputer;

  // A HiddenMarkovModel models one or more time series using a hidden Markov
  // mixture of an arbitrary set of mixture components.  If multiple time series
  // are being modeled then the class is capable of imputing latent data for
  // each series in a separate thread.
  class HiddenMarkovModel : public TimeSeriesDataPolicy<Data>,
                            public CompositeParamPolicy,
                            public PriorPolicy {
   public:
    // Args:
    //   Mix: The mixture components for the hidden Markov mixture.  These
    //     define the "emission distribution" or the "observation equation" for
    //     the HMM.
    //   Mark: The Markov chain describing the transition distribution for the
    //     hidden states.  The dimension of this model must match the number of
    //     mixture components passed in the first argument.
    //
    // A posterior sampling method, but no data, should be assigned to each
    // argument prior to creating the HMM.
    HiddenMarkovModel(const std::vector<Ptr<MixtureComponent>> &Mix,
                      const Ptr<MarkovModel> &Mark);
    HiddenMarkovModel(const HiddenMarkovModel &rhs);
    HiddenMarkovModel *clone() const override;

    // The number of mixture components
    uint state_space_size() const;

    // If 'save' then save the values of the hidden Markov chain that are
    // simulated during a call to impute_latent_data.  If 'save' is false then
    // clear any storage for the saved draws, and stop saving them in the future.
    void save_state_draws(bool save = true);

    // Randomly assign values to the latent Markov chain, and set all Markov
    // transition probabilities to uniform.
    virtual void initialize_params(RNG &rng);

    // When modeling multiple users, the model can use multiple threads for data
    // augmentation.
    void set_nthreads(uint);
    uint nthreads() const;

    // Args:
    //   dp:  A TimeSeries<Data> object describing a sequence of data values.
    //   logscale: see below.
    //
    // Returns: If logscale is true then return the probability (density) of dp
    //   on the log scale.  Otherwise report on the probability scale.
    double pdf(const Ptr<Data> &dp, bool logscale) const;

    // Clear any latent data that has been stored by the data augmentation
    // algorithm in the mixture components or the hidden Markov chain.
    void clear_client_data();

    // Access to the underlying mixture components or Markov model object.
    const std::vector<Ptr<MixtureComponent>> &mixture_components() const;

    Ptr<MixtureComponent> mixture_component(uint s);
    const Ptr<MixtureComponent> &mixture_component(uint s) const;

    Ptr<MarkovModel> mark();
    const Ptr<MarkovModel> &mark() const;

    // Impute a value for each (subject, timestamp) location of the hidden
    // Markov chain(s), store the imputed values in the appropriate mixture
    // components or Markov model object.
    //
    // Args:
    //   rng:  A random number generator used to drive the imputation.
    //
    // Returns:
    //   The log likelihood of the assigned data.
    double impute_latent_data(RNG &rng);

    // Args:
    //   series_index:  The
    //
    // Returns:
    //   The Markov chain values that were simulted for that subject by
    //   the most recent call to 'impute_latent_data'.
    const std::vector<int> &imputed_state(int series_index = 0) const;

    // Compute (or recompute) the log likelihood of the assigned data.
    double loglike() const;

    // The most recent log likelihood value computed by the forward recursions.
    double saved_loglike() const;

    // Randomly assign values to the hidden Markov chain.  Useful as a random
    // starting point to an MCMC algorithm.
    void randomly_assign_data(RNG &rng);

    // For managing the distribution of hidden states.
    void clear_prob_hist();

    // The transition probability matrix of the hidden Markov chain.
    const Matrix &Q() const;
    void set_Q(const Matrix &Q);

    // The initial distribution of the hidden Markov chain.
    const Vector &pi0() const;
    void set_pi0(const Vector &Pi0);

    // Options for managing the distribution of the initial state.
    void fix_pi0(const Vector &Pi0);
    void fix_pi0_stationary();
    bool pi0_fixed() const;

   protected:
    template <class Fwd>  // needed for copy constructor
    void set_mixture_components(Fwd b, Fwd e) {
      mix_.assign(b, e);
      ParamPolicy::set_models(b, e);
      ParamPolicy::add_model(mark_);
    }

    void set_loglike(double);
    void set_logpost(double);
    void set_filter(const Ptr<HmmFilter> &f);

   private:
    Ptr<MarkovModel> mark_;
    std::vector<Ptr<MixtureComponent>> mix_;
    Ptr<HmmFilter> filter_;

    std::map<Ptr<Data>, Vector> prob_hist_;

    Ptr<UnivParams> loglike_;
    Ptr<UnivParams> logpost_;

    // imputed_states_[series][time]
    bool save_state_draws_;
    std::vector<std::vector<int>> imputed_state_;

    std::vector<Ptr<HmmDataImputer>> workers_;

    ThreadWorkerPool thread_pool_;

    double impute_latent_data_with_threads();
  };

  //===========================================================================
  // A hidden Markov model that can be estimated by the EM algorithm instead of
  // the data augmentation algorithm.
  class HMM_EM : public HiddenMarkovModel {
   public:
    typedef EmMixtureComponent EMC;
    HMM_EM(const std::vector<Ptr<EMC>> &Mix, const Ptr<MarkovModel> &Mark);
    HMM_EM(const HMM_EM &rhs);
    HMM_EM *clone() const override;

    void initialize_params(RNG &rng) override;
    virtual void mle();
    double Estep(bool bayes = false);
    void Mstep(bool bayes = false);
    void find_posterior_mode();
    void map();  // throw an exception if any of the mixture
    // components do not have a conjugate prior set
    void mle_trace();
    void set_epsilon(double);

   private:
    void find_mode(bool bayes = false);
    std::vector<Ptr<MixtureComponent>> tomod(
        const std::vector<Ptr<EMC>> &v) const;

    std::vector<Ptr<EMC>> mix_;
    Ptr<HmmEmFilter> filter_;
    double eps;
  };
}  // namespace BOOM
#endif  // BOOM_HMM_HPP
