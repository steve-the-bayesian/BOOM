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

#ifndef BOOM_VARIABLE_SELECTION_PRIOR_HPP
#define BOOM_VARIABLE_SELECTION_PRIOR_HPP

#include "LinAlg/Selector.hpp"
#include "Models/Glm/GlmCoefs.hpp"
#include "Models/Glm/ModelSelectionConcepts.hpp"
#include "Models/ParamTypes.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

/*************************************************************************
 * A VariableSelectionPrior associates 'variable' with a prior
 * probability of inclusion in a model.  Thus it is a prior for the
 * 'gamma' portion of GlmCoefs.  It is, in effect a sequence of
 * binomial models adjusted to know about interactions and variable
 * observation indicators.  The prior for the conditional distribution
 * of beta given gamma is GlmMvnPrior
 *************************************************************************/

namespace BOOM {
  class GlmCoefs;
  class VariableSelectionPrior;

  class VsSuf : public SufstatDetails<GlmCoefs> {
   public:
    typedef ModelSelection::Variable Variable;
    VsSuf();
    VsSuf(const VsSuf &rhs);
    VsSuf *clone() const override;
    void clear() override;
    void Update(const GlmCoefs &) override;
    void add_var(const Ptr<Variable> &v);
    void combine(const Ptr<VsSuf> &);
    void combine(const VsSuf &);
    VsSuf *abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    ostream &print(ostream &out) const override;

   private:
    std::vector<Ptr<Variable>> vars_;
  };

  //______________________________________________________________________

  class VariableSelectionPrior : public SufstatDataPolicy<GlmCoefs, VsSuf>,
                                 public PriorPolicy {
    typedef ModelSelection::Variable Variable;
    typedef ModelSelection::MainEffect MainEffect;
    typedef ModelSelection::MissingMainEffect MissingMainEffect;
    //    typedef ModelSelection::ObsIndicator ObsIndicator;
    typedef ModelSelection::Interaction Interaction;

   public:
    VariableSelectionPrior();

    // A prior for coefficients of dimension n, with marginal inclusion
    // probability.
    explicit VariableSelectionPrior(uint n, double inclusion_probability = 1.0);

    // Args:
    //   marginal_inclusion_probabilities: Each entry gives the marginal
    //     inclusion probability for the corresponding regression coefficient.
    explicit VariableSelectionPrior(
        const Vector &marginal_inclusion_probabilities);

    VariableSelectionPrior(const VariableSelectionPrior &rhs);
    VariableSelectionPrior *clone() const override;

    void mle();
    double pdf(const Ptr<Data> &dp, bool logscale) const;
    double logp(const Selector &inc) const;
    void make_valid(Selector &inc) const;
    const Ptr<Variable> &variable(uint i) const;
    Ptr<Variable> variable(uint i);

    Selector simulate(RNG &rng) const;
    uint potential_nvars() const;

    // A fully observed main effect has probability "prob" to be
    // present.
    void add_main_effect(uint position, double prob,
                         const std::string &name = "");

    // A missing main effect has probability prob of being present if
    // its observation indicator is present.  If the observation
    // indicator is absent then the inclusion probability is 0.
    void add_missing_main_effect(uint position, double prob, uint oi_pos,
                                 const std::string &name = "");

    // an interaction has probability "prob" to be present if all of
    // its parents are also present.  If any of its parents are absent
    // then the interaction has inclusion probability 0.
    void add_interaction(uint position, double prob,
                         const std::vector<uint> &parents,
                         const std::string &name = "");

    // TODO: This class needs to be split apart.  The bit
    // about interactions and main effects (which are dependent on one
    // another) is at odds with the notion that there is a vector of
    // prior inclusion probabilities (which implies independence).
    // Most instances of this class assume the independence case.
    Vector prior_inclusion_probabilities() const;
    double prob(uint i) const;
    void set_probs(const Vector &pi);
    void set_prob(double prob, uint i);
    ParamVector parameter_vector() override;
    const ParamVector parameter_vector() const override;
    void unvectorize_params(const Vector &v, bool minimal = true) override;

    ostream &print(ostream &out) const;

   private:
    std::vector<Ptr<Variable>> vars_;
    std::vector<Ptr<MainEffect>> observed_main_effects_;
    std::vector<Ptr<MissingMainEffect>> missing_main_effects_;
    std::vector<Ptr<Interaction>> interactions_;

    mutable Ptr<VectorParams> pi_;  // for managing io
    void fill_pi() const;
    void check_size_eq(uint n, const std::string &fun) const;
    void check_size_gt(uint n, const std::string &fun) const;
  };

  ostream &operator<<(ostream &out, const VariableSelectionPrior &);

}  // namespace BOOM
#endif  // BOOM_VARIABLE_SELECTION_PRIOR_HPP
