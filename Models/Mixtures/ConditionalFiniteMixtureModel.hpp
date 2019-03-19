// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2014 Steven L. Scott

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

#ifndef BOOM_MIXTURES_CONDITIONAL_FINITE_MIXTURE_MODEL_HPP_
#define BOOM_MIXTURES_CONDITIONAL_FINITE_MIXTURE_MODEL_HPP_

#include "Models/DataTypes.hpp"
#include "Models/Glm/MultinomialLogitModel.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  class ConditionalMixtureData : public Data {
   public:
    // Args:
    //   data: The data to be modeled by the mixture components in the
    //     ConditionalFiniteMixtureModel.
    //   mixture_category_predictors: The vector of predictors used to
    //     help determine the prior probability of mixture category
    //     membership.
    //   number_of_mixture_components: The number of mixture
    //     components being modeled.
    //   known_mixture_component: If the mixture component that
    //     produced this observation is known, then supply its
    //     component number here.  In most cases this will not be
    //     known, in which case any negative number can be supplied
    //     (thus the default of -1).
    ConditionalMixtureData(const Ptr<Data> &data,
                           const Ptr<VectorData> &mixture_category_predictors,
                           int number_of_mixture_components,
                           int known_mixture_component = -1);

    ConditionalMixtureData(const ConditionalMixtureData &rhs);
    ConditionalMixtureData *clone() const override;
    std::ostream &display(std::ostream &out) const override;

    // The individual data point being modeled by the mixture.
    const Data *data() const;
    Ptr<Data> shared_data();

    // The predictor data for determining the mixture category, along
    // with the mixture category indicator.
    Ptr<ChoiceData> shared_mixture_category_data();
    const ChoiceData *mixture_category_data() const;

    // If non-negative, the index of the mixture component that
    // produced this observation.  If negative (the usual case), then
    // the responsible mixture component is unknown.
    int known_mixture_component() const;

    // Sets the mixture component for this observation, which is
    // stored in mixture_category_data()->value().  The mixture
    // component is usually determined by imputation as part of an
    // MCMC algorithm.  If the mixture component responsible for this
    // observation is known in advance (e.g. known_mixture_component()
    // >= 0) then you can only call set_mixture_component with that
    // known value.  Calling with any other value will throw an
    // exception.
    void set_mixture_component(int component);

   private:
    Ptr<Data> data_;
    Ptr<ChoiceData> mixture_category_data_;
    int known_mixture_component_;
  };

  //======================================================================
  // Models a respsonse y[i] given a set of predictors x[i] as a mixture
  //   f(y | x) = \sum_k w[k](x) where w[k](x) \propto exp(beta[k] * x)
  //
  // Note that by having this class inherit from MixtureComponent we
  // can nest several layers of ConditionalFiniteMixtureModels
  // together to get a hierarchical mixture of experts.
  class ConditionalFiniteMixtureModel : virtual public MixtureComponent,
                                        public LatentVariableModel,
                                        public CompositeParamPolicy,
                                        public PriorPolicy {
   public:
    // Args:
    //   mixture_components: The vector of mixture components used in
    //     the model.
    //   mixing_distribution: The mixing distribution "prior" for each
    //     observation.  The dimension of the mixing distribution must
    //     match the number of mixture components in the first
    //     argument.
    //
    // Ideally the mixture components and mixing_distribution will
    // have their posterior samplers set before being passed to the
    // constructor.  In practice one _could_ set them later, but all
    // PosteriorSamplers must be set before logpri() or
    // sample_posterior() is called.  None of the models should have
    // data assigned to them.
    ConditionalFiniteMixtureModel(
        const std::vector<Ptr<MixtureComponent> > &mixture_components,
        const Ptr<MultinomialLogitModel> &mixing_distribution);

    ConditionalFiniteMixtureModel *clone() const override;

    // Clears data from the mixture components.  No data is cleared
    // from the mixing distribution or the vector of data for this
    // model.
    void clear_component_data();

    // Clear the data from the model and all its components.
    void clear_data() override;

    void add_data(const Ptr<Data> &dp) override;
    void add_conditional_mixture_data(const Ptr<ConditionalMixtureData> &dp);
    virtual std::vector<Ptr<ConditionalMixtureData> > &dat();
    virtual const std::vector<Ptr<ConditionalMixtureData> > &dat() const;
    void combine_data(const Model &other_model, bool just_suf = true) override;

    // Assigns each (non-missing) observation to a mixture component.
    // Each call to impute_latent_data also calculates the log
    // likelihood as a byproduct.  You can access the log likelihood
    // by calling last_loglike().
    void impute_latent_data(RNG &rng) override;

    // The number of mixture components in the model.
    int number_of_mixture_components() const;

    MultinomialLogitModel *mixing_distribution();
    const MultinomialLogitModel *mixing_distribution() const;

    MixtureComponent *mixture_component(int s);
    const MixtureComponent *mixture_component(int s) const;

    double last_loglike() const;

    double pdf(const Data *dp, bool logscale) const override;
    double logp(const ConditionalMixtureData &data) const;
    int number_of_observations() const override { return dat().size(); }

    // Sets the mixture component for a specific observation number.
    // Args:
    //   observation_number: The observation for which a mixture
    //     component is to be assigned.
    //   which_component: The number of the mixture component for that
    //     observation.
    void set_mixture_component_for_observation(int observation_number,
                                               int which_component);

    const Matrix &class_membership_probabilities() const {
      return class_membership_probabilities_;
    }

   private:
    std::vector<Ptr<MixtureComponent> > mixture_components_;
    Ptr<MultinomialLogitModel> mixing_distribution_;
    std::vector<Ptr<ConditionalMixtureData> > data_;

    // The likelihood function is evaluated each time we call
    // impute_latent_data.  This stores the result.
    double last_loglike_;

    // workspace for evaluating class member probabilities.
    mutable Vector wsp_;

    // A number_of_observations X number_of_mixture_components matrix,
    // giving the class membership probabilities for each observation
    // as of the last call to impute_latent_data.
    Matrix class_membership_probabilities_;
  };

}  // namespace BOOM

#endif  // BOOM_MIXTURES_CONDITIONAL_FINITE_MIXTURE_MODEL_HPP_
