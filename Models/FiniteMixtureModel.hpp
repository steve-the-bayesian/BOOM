// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007-2012 Steven L. Scott

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

#ifndef BOOM_FINITE_MIXTURE_MODEL_HPP
#define BOOM_FINITE_MIXTURE_MODEL_HPP

#include "Models/EmMixtureComponent.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/MultinomialModel.hpp"
#include "Models/ParamTypes.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/MixtureDataPolicy.hpp"

namespace BOOM {

  class FiniteMixtureModel : public LatentVariableModel,
                             public CompositeParamPolicy,
                             public MixtureDataPolicy,
                             public PriorPolicy {
   public:
    FiniteMixtureModel(const Ptr<MixtureComponent> &, uint S);
    FiniteMixtureModel(const Ptr<MixtureComponent> &mixture_component,
                       const Ptr<MultinomialModel> &mixing_weights);

    template <class M>
    FiniteMixtureModel(const std::vector<Ptr<M>> &mixture_components,
                       const Ptr<MultinomialModel> &mixing_weights);

    template <class FwdIt>
    FiniteMixtureModel(FwdIt Beg, FwdIt End,
                       const Ptr<MultinomialModel> &mixing_weights);

    FiniteMixtureModel(const FiniteMixtureModel &rhs);
    FiniteMixtureModel *clone() const override;

    // Clear data from mixture components and the mixing distribution.
    void clear_component_data();

    void impute_latent_data(RNG &rng) override;
    void class_membership_probability(const Ptr<Data> &, Vector &ans) const;
    int impute_observation(const Ptr<Data> &data, RNG &rng) const;
    int impute_observation(const Ptr<Data> &data, RNG &rng,
                            bool update_complete_data_suf);
    double last_loglike() const;

    double pdf(const Ptr<Data> &dp, bool logscale) const;
    uint number_of_mixture_components() const;

    const Vector &pi() const;
    const Vector &logpi() const;

    Ptr<MultinomialModel> mixing_distribution();
    const MultinomialModel *mixing_distribution() const;

    Ptr<MixtureComponent> mixture_component(int s);
    const MixtureComponent *mixture_component(int s) const;

    // Returns a matrix of class membership probabilities for each
    // observation.  The table of membership probabilities is
    // re-written with each call to impute_latent_data().
    const Matrix &class_membership_probability() const;

    // Returns a vector giving the latent class to which each
    // observation was assigned during the most recent call to
    // impute_latent_data().
    Vector class_assignment() const;

   protected:
    void set_logpi() const;
    mutable Vector wsp_;

    // Save the class membership probabilities for user i.
    void update_class_membership_probabilities(int i, const Vector &probs);

   private:
    std::vector<Ptr<MixtureComponent>> mixture_components_;
    Ptr<MultinomialModel> mixing_dist_;
    mutable Vector logpi_;
    mutable bool logpi_current_;
    void observe_pi() const;
    void set_observers();
    virtual std::vector<Ptr<MixtureComponent>> models();
    virtual const std::vector<Ptr<MixtureComponent>> models() const;
    double last_loglike_;
    Matrix class_membership_probabilities_;
    std::vector<int> which_mixture_component_;
  };
  //----------------------------------------------------------------------
  template <class FwdIt>
  FiniteMixtureModel::FiniteMixtureModel(
      FwdIt Beg, FwdIt End, const Ptr<MultinomialModel> &mixing_weights)
      : DataPolicy(mixing_weights->dim()),
        mixture_components_(Beg, End),
        mixing_dist_(mixing_weights) {
    set_observers();
  }

  template <class M>
  FiniteMixtureModel::FiniteMixtureModel(
      const std::vector<Ptr<M>> &mixture_components,
      const Ptr<MultinomialModel> &mixing_weights)
      : DataPolicy(mixing_weights->dim()),
        mixture_components_(mixture_components.begin(),
                            mixture_components.end()),
        mixing_dist_(mixing_weights) {
    set_observers();
  }

  //======================================================================
  class EmFiniteMixtureModel : public FiniteMixtureModel {
   public:
    EmFiniteMixtureModel(
        const Ptr<EmMixtureComponent> &prototype_mixture_component,
        uint state_space_size);

    EmFiniteMixtureModel(
        const Ptr<EmMixtureComponent> &prototype_mixture_component,
        const Ptr<MultinomialModel> &mixing_distribution);

    template <class M>
    EmFiniteMixtureModel(const std::vector<const Ptr<M> &> &mixture_components,
                         const Ptr<MultinomialModel> &mixing_distribution)
        : FiniteMixtureModel(mixture_components, mixing_distribution),
          em_mixture_components_(mixture_components.begin(),
                                 mixture_components.end()) {}

    template <class FwdIt>
    EmFiniteMixtureModel(FwdIt Beg, FwdIt End,
                         Ptr<MultinomialModel> mixing_distribution)
        : FiniteMixtureModel(Beg, End, mixing_distribution),
          em_mixture_components_(Beg, End) {}

    EmFiniteMixtureModel(const EmFiniteMixtureModel &rhs);
    EmFiniteMixtureModel *clone() const override;

    double loglike() const;
    void mle();

    // The EStep returns the observed data likelihood
    double EStep();
    void MStep(bool posterior_mode);

    Ptr<EmMixtureComponent> em_mixture_component(int s);
    const EmMixtureComponent *em_mixture_component(int s) const;

   private:
    std::vector<Ptr<EmMixtureComponent>> em_mixture_components_;
    void populate_em_mixture_components();
  };

}  // namespace BOOM
#endif  // BOOM_FINITE_MIXTURE_MODEL_HPP
