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

  //===========================================================================
  // Sufficient statistics for a variable selection prior model.
  class VariableSelectionSuf : public SufstatDetails<GlmCoefs> {
   public:
    typedef ModelSelection::Variable Variable;
    VariableSelectionSuf();
    VariableSelectionSuf(const VariableSelectionSuf &rhs);
    VariableSelectionSuf *clone() const override;
    void clear() override;
    void Update(const GlmCoefs &) override;
    void add_var(const Ptr<Variable> &v);
    void combine(const Ptr<VariableSelectionSuf> &);
    void combine(const VariableSelectionSuf &);
    VariableSelectionSuf *abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

   private:
    std::vector<Ptr<Variable>> vars_;
  };

  //===========================================================================
  class VariableSelectionPriorBase : virtual public Model {
   public:
    // Evaluate the log prior probability of the set of included coefficients.
    virtual double logp(const Selector &included_coefficients) const = 0;

    // Modify the selector so that it starts in a region of positive
    // probability.  Set any entries in inc to 0 if the corresponding prior
    // inclusion probability is zero, and set them to 1 if the corresponding
    // prior inclusion probability is 1.  Leave inc unchanged otherwise.
    virtual void make_valid(Selector &inc) const = 0;

    // The total number of potential predictor variables available.
    virtual uint potential_nvars() const = 0;

    virtual std::ostream &print(std::ostream &out) const = 0;
  };

  //===========================================================================
  class VariableSelectionPrior
      : public VariableSelectionPriorBase,
        public ParamPolicy_1<VectorParams>,
        public IID_DataPolicy<GlmCoefs>,
        public PriorPolicy
  {
   public:
    VariableSelectionPrior();
    explicit VariableSelectionPrior(uint n, double inclusion_probability = 1.0);
    explicit VariableSelectionPrior(
        const Vector &marginal_inclusion_probabilities);
    VariableSelectionPrior *clone() const override;
    double logp(const Selector &included_coefficients) const override;

    void set_prior_inclusion_probabilities(const Vector &probs) {
      ParamPolicy::prm_ref().set(probs);
    }
    void set_prior_inclusion_probability(int i, double value) {
      ParamPolicy::prm_ref().set_element(value, i);
    }

    const Vector &prior_inclusion_probabilities() const {
      return ParamPolicy::prm_ref().value();
    }

    void make_valid(Selector &inc) const override;
    uint potential_nvars() const override;

    virtual std::ostream &print(std::ostream &out) const override;

   private:
    // Set an observer on the vector of prior inclusion probabilities so that
    // the vectors of log probabilities will be marked not current if the raw
    // probabilities change.
    void observe_prior_inclusion_probabilities();

    // If the vectors of log probabilities and their complements are not
    // current, recompute them and mark them current.
    //
    // This function is logically const.
    void ensure_log_probabilities() const;

    mutable bool current_;
    mutable Vector log_inclusion_probabilities_;
    mutable Vector log_complementary_inclusion_probabilities_;
  };

  //===========================================================================
  // A variable selection prior that is aware of the different types of effects
  // (e.g. main effects, interactions, indicator variables, etc).  It can be
  // used to facilitate things like only including interaction terms if main
  // effects are included.
  class StructuredVariableSelectionPrior :
      public VariableSelectionPriorBase,
      public SufstatDataPolicy<GlmCoefs, VariableSelectionSuf>,
      public PriorPolicy {
    typedef ModelSelection::Variable Variable;
    typedef ModelSelection::MainEffect MainEffect;
    typedef ModelSelection::MissingMainEffect MissingMainEffect;
    typedef ModelSelection::Interaction Interaction;

   public:
    StructuredVariableSelectionPrior();

    // A prior for coefficients of dimension n, with marginal inclusion
    // probability.
    explicit StructuredVariableSelectionPrior(
        uint n, double inclusion_probability = 1.0);

    // Args:
    //   marginal_inclusion_probabilities: Each entry gives the marginal
    //     inclusion probability for the corresponding regression coefficient.
    explicit StructuredVariableSelectionPrior(
        const Vector &marginal_inclusion_probabilities);

    StructuredVariableSelectionPrior(
        const StructuredVariableSelectionPrior &rhs);
    StructuredVariableSelectionPrior *clone() const override;

    void mle();
    double pdf(const Ptr<Data> &dp, bool logscale) const;
    double logp(const Selector &included_coefficients) const override;
    void make_valid(Selector &inc) const override;
    const Ptr<Variable> &variable(uint i) const;
    Ptr<Variable> variable(uint i);

    Selector simulate(RNG &rng) const;
    uint potential_nvars() const override;

    // A fully observed main effect has probability "prob" to be
    // present.
    void add_main_effect(uint position, double prob,
                         const std::string &name = "");

    // A missing main effect has probability prob of being present if its
    // observation indicator is present.  If the observation indicator is absent
    // then the inclusion probability is 0.
    void add_missing_main_effect(uint position, double prob, uint oi_pos,
                                 const std::string &name = "");

    // an interaction has probability "prob" to be present if all of its parents
    // are also present.  If any of its parents are absent then the interaction
    // has inclusion probability 0.
    void add_interaction(uint position, double prob,
                         const std::vector<uint> &parents,
                         const std::string &name = "");

    // TODO: This class needs to be split apart.  The bit about interactions and
    // main effects (which are dependent on one another) is at odds with the
    // notion that there is a vector of prior inclusion probabilities (which
    // implies independence).  Most instances of this class assume the
    // independence case.
    Vector prior_inclusion_probabilities() const;
    double prob(uint i) const;
    void set_probs(const Vector &pi);
    void set_prob(double prob, uint i);
    std::vector<Ptr<Params>> parameter_vector() override;
    const std::vector<Ptr<Params>> parameter_vector() const override;
    void unvectorize_params(const Vector &v, bool minimal = true) override;

    std::ostream &print(std::ostream &out) const override;

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

  std::ostream &operator<<(std::ostream &out, const VariableSelectionPriorBase &);

  //===========================================================================
  // Model the include / exclude behavior for a set of coefficients that has
  // been organized in a matrix.
  class MatrixVariableSelectionPrior
      : public ParamPolicy_1<MatrixParams>,
        public IID_DataPolicy<MatrixGlmCoefs>,
        public PriorPolicy {
    public:
    explicit MatrixVariableSelectionPrior(
        const Matrix &prior_inclusion_probabilities);

    MatrixVariableSelectionPrior *clone() const override {
      return new MatrixVariableSelectionPrior(*this);
    }

     const Matrix &prior_inclusion_probabilities() const {
       return ParamPolicy::prm_ref().value();
     }

    double logp(const SelectorMatrix &included) const;

    int nrow() const {return prior_inclusion_probabilities().nrow();}
    int ncol() const {return prior_inclusion_probabilities().ncol();}

   private:
    // Throw an error if any element of probs is outside the range [0, 1].
    void check_probabilities(const Matrix &probs) const;

    // Set an observer on the vector of prior inclusion probabilities so that
    // the vectors of log probabilities will be marked not current if the raw
    // probabilities change.
    void observe_prior_inclusion_probabilities();

    // If the vectors of log probabilities and their complements are not
    // current, recompute them and mark them current.
    //
    // This function is logically const.
    void ensure_log_probabilities() const;

    mutable bool current_;
    mutable Matrix log_inclusion_probabilities_;
    mutable Matrix log_complementary_inclusion_probabilities_;
   };

}  // namespace BOOM
#endif  // BOOM_VARIABLE_SELECTION_PRIOR_HPP
