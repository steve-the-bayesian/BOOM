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

#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/BinomialModel.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    typedef VariableSelectionPrior VSP;
    typedef StructuredVariableSelectionPrior SVSP;
    typedef ModelSelection::Variable Variable;
  }  // namespace

  namespace ModelSelection {
    Variable::Variable(uint pos, double prob, const std::string &name)
        : pos_(pos), mod_(new BinomialModel(prob)), name_(name) {}

    Variable::Variable(const Variable &rhs)
        : RefCounted(rhs),
          pos_(rhs.pos_),
          mod_(rhs.mod_->clone()),
          name_(rhs.name_) {}

    Variable::~Variable() {}

    std::ostream &Variable::print(std::ostream &out) const {
      out << "Variable " << name_ << " position " << pos_ << " probability "
          << mod_->prob() << " ";
      return out;
    }

    std::ostream &operator<<(std::ostream &out, const Variable &v) {
      return v.print(out);
    }

    uint Variable::pos() const { return pos_; }
    double Variable::prob() const { return mod_->prob(); }
    void Variable::set_prob(double prob) { mod_->set_prob(prob); }
    double Variable::logp(const Selector &inc) const {
      return mod_->pdf(1, inc[pos_], true);
    }
    Ptr<BinomialModel> Variable::model() { return mod_; }
    const Ptr<BinomialModel> Variable::model() const { return mod_; }

    const std::string &Variable::name() const { return name_; }

    //=====================================================================
    MainEffect::MainEffect(uint position, double prob, const std::string &name)
        : Variable(position, prob, name) {}

    MainEffect *MainEffect::clone() const { return new MainEffect(*this); }

    bool MainEffect::observed() const { return true; }
    bool MainEffect::parents_are_present(const Selector &) const {
      return true;
    }

    void MainEffect::make_valid(Selector &inc) const {
      double p = prob();
      bool in = inc[pos()];
      if ((p >= 1.0 && !in) || (p <= 0.0 && in)) {
        inc.flip(pos());
      }
    }

    void MainEffect::add_to(SVSP &vsp) const {
      vsp.add_main_effect(pos(), prob(), name());
    }

    //=====================================================================
    MissingMainEffect::MissingMainEffect(uint position, double prob,
                                         uint obs_ind_pos,
                                         const std::string &name)
        : MainEffect(position, prob, name), obs_ind_pos_(obs_ind_pos) {}

    MissingMainEffect::MissingMainEffect(const MissingMainEffect &rhs)
        : MainEffect(rhs), obs_ind_pos_(rhs.obs_ind_pos_) {}

    MissingMainEffect *MissingMainEffect::clone() const {
      return new MissingMainEffect(*this);
    }

    double MissingMainEffect::logp(const Selector &inc) const {
      bool in = inc[pos()];
      bool oi_in = inc[obs_ind_pos_];
      if (oi_in) return Variable::logp(inc);
      return in ? BOOM::negative_infinity() : 0;
    }

    void MissingMainEffect::make_valid(Selector &inc) const {
      bool in = inc[pos()];
      double p = prob();
      if (p <= 0.0 && in) {
        inc.drop(pos());
      } else if (p >= 1.0 && !in) {
        inc.add(pos());
        inc.add(obs_ind_pos_);
      }
    }

    bool MissingMainEffect::observed() const { return false; }

    bool MissingMainEffect::parents_are_present(const Selector &g) const {
      return g[obs_ind_pos_];
    }

    void MissingMainEffect::add_to(SVSP &vsp) const {
      vsp.add_missing_main_effect(pos(), prob(), obs_ind_pos_, name());
    }

    //=====================================================================
    Interaction::Interaction(uint position, double prob,
                             const std::vector<uint> &parents,
                             const std::string &name)
        : Variable(position, prob, name), parent_pos_(parents) {}

    Interaction::Interaction(const Interaction &rhs)
        : Variable(rhs), parent_pos_(rhs.parent_pos_) {}

    Interaction *Interaction::clone() const { return new Interaction(*this); }

    uint Interaction::nparents() const { return parent_pos_.size(); }

    double Interaction::logp(const Selector &inc) const {
      uint n = nparents();
      for (uint i = 0; i < n; ++i) {
        uint indx = parent_pos_[i];
        if (!inc[indx]) return BOOM::negative_infinity();
      }
      return Variable::logp(inc);
    }

    void Interaction::make_valid(Selector &inc) const {
      double p = prob();
      bool in = inc[pos()];
      if (p <= 0.0 && in) {
        inc.drop(pos());
      } else if (p >= 1.0 && !in) {
        inc.add(pos());
        for (int i = 0; i < parent_pos_.size(); ++i) {
          inc.add(parent_pos_[i]);
        }
      }
    }

    bool Interaction::parents_are_present(const Selector &g) const {
      uint n = parent_pos_.size();
      for (uint i = 0; i < n; ++i) {
        if (!g[parent_pos_[i]]) return false;
      }
      return true;
    }

    void Interaction::add_to(SVSP &vsp) const {
      vsp.add_interaction(pos(), prob(), parent_pos_, name());
    }
  }  // namespace ModelSelection

  //===========================================================================
  VariableSelectionSuf::VariableSelectionSuf() {}

  VariableSelectionSuf::VariableSelectionSuf(const VariableSelectionSuf &rhs)
      : Sufstat(rhs), SufTraits(rhs), vars_(rhs.vars_) {}

  VariableSelectionSuf *VariableSelectionSuf::clone() const {
    return new VariableSelectionSuf(*this);
  }

  void VariableSelectionSuf::add_var(const Ptr<Variable> &v) {
    vars_.push_back(v);
  }

  void VariableSelectionSuf::clear() {
    uint n = vars_.size();
    for (uint i = 0; i < n; ++i) vars_[i]->model()->clear_suf();
  }

  void VariableSelectionSuf::Update(const GlmCoefs &beta) {
    uint n = vars_.size();
    for (uint i = 0; i < n; ++i) {
      const Selector &g(beta.inc());
      if (vars_[i]->parents_are_present(g)) {
        bool y = g[i];
        vars_[i]->model()->suf()->update_raw(y);
      }
    }
  }

  void VariableSelectionSuf::combine(const Ptr<VariableSelectionSuf> &) {
    report_error("cannot combine VariableSelectionSuf");
  }

  void VariableSelectionSuf::combine(const VariableSelectionSuf &) {
    report_error("cannot combine VariableSelectionSuf");
  }

  VariableSelectionSuf *VariableSelectionSuf::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  Vector VariableSelectionSuf::vectorize(bool) const {
    report_error("cannot vectorize VariableSelectionSuf");
    return Vector(1, 0.0);
  }

  Vector::const_iterator VariableSelectionSuf::unvectorize(
      Vector::const_iterator &v, bool) {
    report_error("cannot unvectorize VariableSelectionSuf");
    return v;
  }

  Vector::const_iterator VariableSelectionSuf::unvectorize(
      const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  std::ostream &VariableSelectionSuf::print(std::ostream &out) const {
    return out << "VariableSelectionSuf is hard to print!";
  }

  //===========================================================================
  VSP::VariableSelectionPrior()
      : ParamPolicy(new VectorParams(0)),
        current_(false)
  {
    observe_prior_inclusion_probabilities();
  }

  VSP::VariableSelectionPrior(uint n, double inclusion_probability)
      : ParamPolicy(new VectorParams(n, inclusion_probability)),
        current_(false)
  {
    if (inclusion_probability < 0 || inclusion_probability > 1) {
      report_error("Prior inclusion probability must be between 0 and 1.");
    }
    observe_prior_inclusion_probabilities();
  }

  VSP::VariableSelectionPrior(const Vector &marginal_inclusion_probabilities)
      : ParamPolicy(new VectorParams(marginal_inclusion_probabilities)),
        current_(false)
  {
    observe_prior_inclusion_probabilities();
  }

  VSP *VSP::clone() const {return new VSP(*this);}

  double VSP::logp(const Selector &inc) const {
    ensure_log_probabilities();
    double ans = 0;
    for (int i = 0; i < inc.nvars_possible(); ++i) {
      ans += inc[i] ? log_inclusion_probabilities_[i] :
          log_complementary_inclusion_probabilities_[i];
      if (!std::isfinite(ans)) {
        return negative_infinity();
      }
    }
    return ans;
  }

  void VSP::make_valid(Selector &inc) const {
    const Vector &probs(prior_inclusion_probabilities());
    if (inc.nvars_possible() != probs.size()) {
      report_error("Wrong size Selector passed to make_valid.");
    }
    for (int i = 0; i < probs.size(); ++i) {
      if (probs[i] <= 0.0 && inc[i]) {
        inc.flip(i);
      }
      if (probs[i] >= 1.0 && !inc[i]) {
        inc.flip(i);
      }
    }
  }

  uint VSP::potential_nvars() const {
    return prior_inclusion_probabilities().size();
  }

  void VSP::observe_prior_inclusion_probabilities() {
    prm()->add_observer([this]() { this->current_ = false;});
  }

  void VSP::ensure_log_probabilities() const {
    if (!current_) {
      log_inclusion_probabilities_ = log(prior_inclusion_probabilities());
      log_complementary_inclusion_probabilities_ =
          log(1 - prior_inclusion_probabilities());
      current_ = true;
    }
  }

  std::ostream &VSP::print(std::ostream &out) const {
    return out << prior_inclusion_probabilities() << std::endl;
  }

  //===========================================================================
  SVSP::StructuredVariableSelectionPrior()
      : DataPolicy(new VariableSelectionSuf), pi_(new VectorParams(0)) {}

  SVSP::StructuredVariableSelectionPrior(uint n, double inclusion_probability)
      : DataPolicy(new VariableSelectionSuf), pi_(new VectorParams(0)) {
    for (uint i = 0; i < n; ++i) {
      add_main_effect(i, inclusion_probability);
    }
  }

  SVSP::StructuredVariableSelectionPrior(const Vector &marginal_inclusion_probabilities)
      : DataPolicy(new VariableSelectionSuf), pi_(new VectorParams(0)) {
    uint n = marginal_inclusion_probabilities.size();
    for (uint i = 0; i < n; ++i) {
      add_main_effect(i, marginal_inclusion_probabilities[i]);
    }
  }

  SVSP::StructuredVariableSelectionPrior(const SVSP &rhs)
      : Model(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        pi_(new VectorParams(rhs.pi_->size())) {
    uint n = rhs.vars_.size();
    for (uint i = 0; i < n; ++i) {
      rhs.vars_[i]->add_to(*this);
    }
  }

  SVSP *SVSP::clone() const { return new SVSP(*this); }

  void SVSP::check_size_eq(uint n, const std::string &fun) const {
    if (vars_.size() == n) return;
    ostringstream err;
    err << "error in SVSP::" << fun << endl
        << "you passed a vector of size " << n << " but there are "
        << vars_.size() << " variables." << endl;
    report_error(err.str());
  }

  void SVSP::check_size_gt(uint n, const std::string &fun) const {
    if (vars_.size() > n) return;
    ostringstream err;
    err << "error in SVSP::" << fun << endl
        << "you tried to access a variable at position " << n
        << ", but there are only " << vars_.size() << " variables." << endl;
    report_error(err.str());
  }

  void SVSP::set_prob(double prob, uint i) {
    check_size_gt(i, "set_prob");
    vars_[i]->set_prob(prob);
  }

  void SVSP::set_probs(const Vector &pi) {
    uint n = pi.size();
    check_size_eq(n, "set_probs");
    for (uint i = 0; i < n; ++i) vars_[i]->set_prob(pi[i]);
  }

  Vector SVSP::prior_inclusion_probabilities() const {
    Vector ans(potential_nvars());
    for (int i = 0; i < ans.size(); ++i) {
      ans[i] = prob(i);
    }
    return ans;
  }

  double SVSP::prob(uint i) const {
    check_size_gt(i, "prob");
    return vars_[i]->prob();
  }

  void SVSP::fill_pi() const {
    uint n = vars_.size();
    Vector tmp(n);
    for (uint i = 0; i < n; ++i) tmp[i] = vars_[i]->prob();
    pi_->set(tmp);
  }

  std::vector<Ptr<Params>> SVSP::parameter_vector() {
    fill_pi();
    return std::vector<Ptr<Params>>(1, pi_);
  }

  const std::vector<Ptr<Params>> SVSP::parameter_vector() const {
    fill_pi();
    return std::vector<Ptr<Params>>(1, pi_);
  }

  void SVSP::unvectorize_params(const Vector &v, bool) {
    uint n = v.size();
    check_size_eq(n, "unvectorize_params");
    for (uint i = 0; i < n; ++i) {
      double p = v[i];
      vars_[i]->model()->set_prob(p);
    }
  }

  void SVSP::mle() {
    uint n = vars_.size();
    for (uint i = 0; i < n; ++i) vars_[i]->model()->mle();
  }

  double SVSP::pdf(const Ptr<Data> &dp, bool logscale) const {
    Ptr<GlmCoefs> d(DAT(dp));
    double ans = logp(d->inc());
    return logscale ? ans : exp(ans);
  }

  Ptr<ModelSelection::Variable> SVSP::variable(uint i) { return vars_[i]; }
  const Ptr<Variable> &SVSP::variable(uint i) const { return vars_[i]; }

  namespace {
    inline void draw(const Ptr<ModelSelection::Variable> &v, Selector &g,
                     RNG &rng) {
      double u = runif_mt(rng, 0, 1);
      uint pos = v->pos();
      if (u < v->prob()) {
        g.add(pos);
      }
    }
  }  // namespace

  Selector SVSP::simulate(RNG &rng) const {
    uint n = potential_nvars();
    Selector ans(n, false);
    // Simulate main_effects.
    uint nme = observed_main_effects_.size();
    for (uint i = 0; i < nme; ++i) draw(observed_main_effects_[i], ans, rng);

    // Simulate missing main effects.
    uint nmis = missing_main_effects_.size();
    for (uint i = 0; i < nmis; ++i) {
      Ptr<MissingMainEffect> v = missing_main_effects_[i];
      if (v->parents_are_present(ans)) draw(v, ans, rng);
    }

    uint nint = interactions_.size();
    for (uint i = 0; i < nint; ++i) {
      Ptr<Interaction> v = interactions_[i];
      if (v->parents_are_present(ans)) draw(v, ans, rng);
    }
    return ans;
  }

  uint SVSP::potential_nvars() const { return vars_.size(); }

  double SVSP::logp(const Selector &included_coefficients) const {
    const double neg_inf = BOOM::negative_infinity();
    uint n = vars_.size();
    double ans = 0;
    for (uint i = 0; i < n; ++i) {
      ans += vars_[i]->logp(included_coefficients);
      if (ans <= neg_inf) {
        return ans;
      }
    }
    return ans;
  }

  void SVSP::make_valid(Selector &inc) const {
    int n = vars_.size();
    for (int i = 0; i < n; ++i) {
      vars_[i]->make_valid(inc);
    }
  }

  void SVSP::add_main_effect(uint position, double prob,
                            const std::string &name) {
    NEW(MainEffect, me)(position, prob, name);
    observed_main_effects_.push_back(me);
    Ptr<Variable> v(me);
    vars_.push_back(v);
    suf()->add_var(v);
  }

  void SVSP::add_missing_main_effect(uint position, double prob, uint oi_pos,
                                    const std::string &name) {
    NEW(MissingMainEffect, mme)(position, prob, oi_pos, name);
    suf()->add_var(mme);
    vars_.push_back(Ptr<Variable>(mme));
    missing_main_effects_.push_back(mme);
  }

  void SVSP::add_interaction(uint position, double prob,
                            const std::vector<uint> &parents,
                            const std::string &name) {
    NEW(Interaction, inter)(position, prob, parents, name);
    Ptr<Variable> v(inter);
    vars_.push_back(v);
    suf()->add_var(v);
    interactions_.push_back(inter);
  }

  std::ostream &SVSP::print(std::ostream &out) const {
    uint nv = vars_.size();
    for (uint i = 0; i < nv; ++i) {
      out << *(vars_[i]) << endl;
    }
    return out;
  }

  std::ostream &operator<<(std::ostream &out, const VariableSelectionPriorBase &vsp) {
    return vsp.print(out);
  }

  //===========================================================================
  namespace {
    using MVSP = MatrixVariableSelectionPrior;
  }

  MVSP::MatrixVariableSelectionPrior(
      const Matrix &prior_inclusion_probabilities)
      : ParamPolicy(new MatrixParams(prior_inclusion_probabilities)),
        current_(false)
  {
    check_probabilities(prior_inclusion_probabilities);
    observe_prior_inclusion_probabilities();
  }

  double MVSP::logp(const SelectorMatrix &included) const {
    if (included.nrow() != nrow() || included.ncol() != ncol()) {
      report_error("Wrong size selector matrix passed to logp.");
    }
    ensure_log_probabilities();
    double ans = 0;
    for (int i = 0; i < nrow(); ++i) {
      for (int j = 0; j < ncol(); ++j) {
        ans += included(i, j) ? log_inclusion_probabilities_(i, j)
            : log_complementary_inclusion_probabilities_(i, j);
        if (!std::isfinite(ans)) return negative_infinity();
      }
    }
    return ans;
  }

  void MVSP::check_probabilities(const Matrix &probs) const {
    for (int i = 0; i < probs.nrow(); ++i) {
      for (int j = 0; j < probs.ncol(); ++j) {
        if (probs(i, j) < 0.0 || probs(i, j) > 1.0) {
          report_error("All probabilities must be in the range [0, 1].");
        }
      }
    }
  }

  void MVSP::ensure_log_probabilities() const {
    if (!current_) {
      log_inclusion_probabilities_ = log(prior_inclusion_probabilities());
      log_complementary_inclusion_probabilities_ =
          log(1 - prior_inclusion_probabilities());
      current_ = true;
    }
  }

  void MVSP::observe_prior_inclusion_probabilities() {
    prm()->add_observer(
        [this]() {
          this->current_ = false;
        });
  }


}  // namespace BOOM
