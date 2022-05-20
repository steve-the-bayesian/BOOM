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

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <set>
#include <tuple>

#include "uint.hpp"
#include "cpputil/make_unique_preserve_order.hpp"
#include "cpputil/report_error.hpp"
#include "stats/Design.hpp"

namespace BOOM {

  LabeledMatrix generate_design_matrix(const ExperimentStructure &xp,
                                       const RowBuilder &builder) {
    std::vector<std::vector<int>> configurations;
    Configuration configuration(xp.nlevels());
    while (!configuration.done()) {
      configurations.push_back(configuration.levels());
      configuration.next();
    }
    Matrix design(configurations.size(), builder.dimension());
    for (int i = 0; i < configurations.size(); ++i) {
      design.row(i) = builder.build_row(configurations[i]);
    }
    LabeledMatrix ans(design, std::vector<std::string>(),
                      builder.variable_names());
    return ans;
  }

  typedef std::map<std::string, std::vector<std::string>> LevelNamesMap;
  LabeledMatrix generate_design_matrix(const LevelNamesMap &level_names_map,
                                       int order) {
    std::vector<std::string> factor_names;
    std::vector<std::vector<std::string>> level_names;
    for (LevelNamesMap::const_iterator it = level_names_map.begin();
         it != level_names_map.end(); ++it) {
      factor_names.push_back(it->first);
      level_names.push_back(it->second);
    }
    ExperimentStructure xp(factor_names, level_names);
    RowBuilder builder(xp, order);
    Configuration config(xp.nlevels());
    int nr = xp.nconfigurations();
    int nc = builder.dimension();
    Matrix X(nr, nc);
    int i = 0;
    while (!config.done()) {
      X.row(i++) = builder.build_row(config.levels());
      config.next();
    }
    std::vector<std::string> vnames = builder.variable_names();
    std::vector<std::string> row_names;  // empty
    LabeledMatrix ans(X, row_names, vnames);
    return ans;
  }

  LabeledMatrix generate_contextual_design_matrix(
      const ExperimentStructure &context_structure,
      const ContextualRowBuilder &builder) {
    Configuration configuration(context_structure.nlevels());
    std::vector<std::vector<int>> context_configurations;
    while (!configuration.done()) {
      context_configurations.push_back(configuration.levels());
      configuration.next();
    }
    std::vector<int> dummy_experimental_configuration(
        builder.number_of_experimental_factors(), 0);
    Matrix design(context_configurations.size(), builder.dimension());
    for (int i = 0; i < context_configurations.size(); ++i) {
      design.row(i) = builder.build_row(dummy_experimental_configuration,
                                        context_configurations[i]);
    }
    Selector pure_context = builder.pure_context();
    LabeledMatrix ans(pure_context.select_cols(design),
                      std::vector<std::string>(),
                      pure_context.select(builder.variable_names()));
    return ans;
  }

  LabeledMatrix generate_experimental_design_matrix(
      const ExperimentStructure &xp, const ContextualRowBuilder &builder) {
    Configuration configuration(xp.nlevels());
    std::vector<std::vector<int>> configurations;
    while (!configuration.done()) {
      configurations.push_back(configuration.levels());
      configuration.next();
    }
    std::vector<int> dummy_context_configuration(
        builder.number_of_contextual_factors(), 0);
    Matrix design(configurations.size(), builder.dimension());
    for (int i = 0; i < configurations.size(); ++i) {
      design.row(i) =
          builder.build_row(configurations[i], dummy_context_configuration);
    }
    Selector pure_experiment = builder.pure_experiment();
    LabeledMatrix ans(pure_experiment.select_cols(design),
                      std::vector<std::string>(),
                      pure_experiment.select(builder.variable_names()));
    return ans;
  }

  ExperimentStructure::ExperimentStructure(const std::vector<int> &nlevels,
                                           bool context)
      : nlevels_(nlevels) {
    int nfactors = nlevels.size();
    for (int i = 0; i < nfactors; ++i) {
      std::ostringstream factor_name_builder;
      if (context) {
        factor_name_builder << "Context" << i;
      } else {
        factor_name_builder << "X" << i;
      }
      factor_names_.push_back(factor_name_builder.str());

      std::vector<std::string> current_level_names;
      for (int level = 0; level < nlevels_[i]; ++level) {
        std::ostringstream level_name_builder;
        level_name_builder << level;
        current_level_names.push_back(level_name_builder.str());
      }
      level_names_.push_back(current_level_names);
    }
  }

  ExperimentStructure::ExperimentStructure(
      const std::vector<std::string> &factor_names,
      const std::vector<std::vector<std::string>> &level_names)
      : factor_names_(factor_names),
        level_names_(level_names),
        nlevels_(level_names.size()) {
    if (factor_names_.size() != level_names_.size()) {
      report_error("factor_names and level_names must have the same size");
    }
    for (int i = 0; i < level_names_.size(); ++i) {
      nlevels_[i] = level_names_[i].size();
    }
  }

  int ExperimentStructure::nfactors() const { return factor_names_.size(); }

  int ExperimentStructure::nlevels(int factor) const {
    return level_names_[factor].size();
  }

  const std::vector<int> &ExperimentStructure::nlevels() const {
    return nlevels_;
  }

  int ExperimentStructure::nconfigurations() const {
    int ans = 1;
    for (int i = 0; i < nfactors(); ++i) ans *= nlevels(i);
    return ans;
  }

  const std::string &ExperimentStructure::level_name(int factor,
                                                     int level) const {
    return level_names_[factor][level];
  }

  std::string ExperimentStructure::full_level_name(
      int factor, int level, const std::string &separator) const {
    std::ostringstream name;
    name << factor_names_[factor] << separator << level_name(factor, level);
    return name.str();
  }

  //======================================================================
  Configuration::Configuration(const std::vector<int> &nlevels)
      : nlevels_(nlevels), levels_(nlevels.size(), 0) {}

  Configuration::Configuration(const std::vector<int> &nlevels,
                               const std::vector<int> &levels)
      : nlevels_(nlevels), levels_(levels) {}

  void Configuration::next() {
    if (done()) return;
    int pos = levels_.size() - 1;
    while (pos >= 0) {
      ++levels_[pos];
      if (levels_[pos] < nlevels_[pos]) return;
      levels_[pos] = 0;
      --pos;
    }
    // if you've gotten here then all levels have seen their maximum value
    levels_.assign(levels_.size(), -1);
  }

  bool Configuration::done() const {
    return levels_.empty() || levels_[0] == -1;
  }

  int Configuration::level(int factor) const { return levels_[factor]; }

  const std::vector<int> &Configuration::levels() const { return levels_; }

  bool Configuration::operator==(const Configuration &rhs) const {
    return (levels_ == rhs.levels_) && (nlevels_ == rhs.nlevels_);
  }

  bool Configuration::operator!=(const Configuration &rhs) const {
    return !(*this == rhs);
  }

  ostream &Configuration::print(ostream &out) const {
    int n = levels_.size();
    if (n == 0) return out;
    out << levels_[0];
    for (int i = 1; i < n; ++i) {
      out << " " << levels_[i];
    }
    return out;
  }

  //======================================================================
  FactorDummy::FactorDummy(int factor, int level, const std::string &name)
      : factor_(factor), level_(level), name_(name) {
    if (level_ < 0) {
      factor_ = -1;
    }
  }

  bool FactorDummy::eval(const std::vector<int> &levels) const {
    return factor_ < 0 || level_ < 0 ? false : levels[factor_] == level_;
  }

  const std::string &FactorDummy::name() const { return name_; }

  bool FactorDummy::operator==(const FactorDummy &rhs) const {
    return (factor_ == rhs.factor_) && (level_ == rhs.level_);
  }

  bool FactorDummy::operator<(const FactorDummy &rhs) const {
    if (factor_ < rhs.factor_) return true;
    if (factor_ > rhs.factor_) return false;
    return level_ < rhs.level_;
  }

  int FactorDummy::factor() const { return factor_; }

  int FactorDummy::level() const { return level_; }

  void FactorDummy::set_level(std::vector<int> &configuration) const {
    if (configuration.size() <= factor_) {
      configuration.resize(factor_ + 1);
    }
    configuration[factor_] = level_;
  }

  //======================================================================
  Effect::Effect() {}
  Effect::Effect(const FactorDummy &factor) { add_factor(factor); }
  Effect::Effect(const Effect &first, const Effect &second) {
    // Add the factors from first.
    std::copy(first.factors_.begin(), first.factors_.end(),
              back_inserter(factors_));
    // Add the factors from second.
    std::copy(second.factors_.begin(), second.factors_.end(),
              back_inserter(factors_));
    // Remove any duplicates.
    std::sort(factors_.begin(), factors_.end());
    std::vector<FactorDummy>::iterator it =
        std::unique(factors_.begin(), factors_.end());
    if (it != factors_.end()) {
      factors_.erase(it);
    }

    // If there are two factor dummies for the same factor, checking
    // for different levels, then this Effect will always return 0.
    // This is represented with a single null FactorDummy.
    for (int i = 1; i < factors_.size(); ++i) {
      if (factors_[i - 1].factor() == factors_[i].factor()) {
        factors_.clear();
        factors_.push_back(FactorDummy(-1, -1, ""));
        break;
      }
    }
  }

  int Effect::order() const { return factors_.size(); }

  void Effect::add_factor(const FactorDummy &factor) {
    if (!has_factor(factor)) factors_.push_back(factor);
    std::sort(factors_.begin(), factors_.end());
  }

  void Effect::add_effect(const Effect &effect) {
    int nef = effect.factors_.size();
    for (int i = 0; i < nef; ++i) {
      add_factor(effect.factors_[i]);
    }
  }

  bool Effect::eval(const std::vector<int> &levels) const {
    for (int i = 0; i < factors_.size(); ++i) {
      if (!factors_[i].eval(levels)) return false;
    }
    return true;
  }

  std::string Effect::name() const {
    int nterms = factors_.size();
    if (nterms == 0) return "Intercept";
    std::string ans = factors_[0].name();
    for (int i = 1; i < nterms; ++i) {
      ans += ":";
      ans += factors_[i].name();
    }
    return ans;
  }

  bool Effect::operator==(const Effect &rhs) const {
    return factors_ == rhs.factors_;
  }

  bool Effect::operator<(const Effect &rhs) const {
    if (order() == rhs.order()) {
      return factors_ < rhs.factors_;
    }
    // Low order interactions come before high order interactions.
    return order() < rhs.order();
  }

  // Returns true if the effect already has a factor from the same
  // factor family as the factor dummy.
  bool Effect::has_factor(const FactorDummy &factor) const {
    int factor_value = factor.factor();
    for (int i = 0; i < factors_.size(); ++i) {
      if (factors_[i].factor() == factor_value) return true;
    }
    return false;
  }

  bool Effect::models_factor(int factor_position_in_input_data) const {
    for (int i = 0; i < factors_.size(); ++i) {
      if (factors_[i].factor() == factor_position_in_input_data) {
        return true;
      }
    }
    return false;
  }

  // Effects made of multiple factor dummies are sorted by factor.
  // Invalid FactorDummies have factor() < 0, so we can just check
  // whether the first FactorDummy is valid.
  bool Effect::is_valid() const {
    return factors_.empty() || factors_[0].factor() >= 0;
  }

  const FactorDummy &Effect::factor(int internal_factor_number) const {
    return factors_[internal_factor_number];
  }

  const FactorDummy &Effect::factor_dummy_for_factor(
      int factor_position_in_input_data) const {
    for (int i = 0; i < factors_.size(); ++i) {
      if (factors_[i].factor() == factor_position_in_input_data) {
        return factors_[i];
      }
    }
    ostringstream err;
    err << "Factor position: " << factor_position_in_input_data
        << " not found.";
    report_error(err.str());
    return factors_[0];  // never get here, but silence the compiler.
  }

  void Effect::set_levels(std::vector<int> &configuration) const {
    for (int i = 0; i < factors_.size(); ++i) {
      factors_[i].set_level(configuration);
    }
  }

  void print(const Effect &e) { std::cout << e << std::endl; }

  //======================================================================
  ContextualEffect::ContextualEffect() {}

  ContextualEffect::ContextualEffect(const FactorDummy &factor,
                                     bool is_context) {
    if (is_context) {
      add_context_factor(factor);
    } else {
      add_experiment_factor(factor);
    }
  }

  ContextualEffect::ContextualEffect(const Effect &effect, bool is_context) {
    for (int i = 0; i < effect.order(); ++i) {
      if (is_context) {
        add_context_factor(effect.factor(i));
      } else {
        add_experiment_factor(effect.factor(i));
      }
    }
  }

  ContextualEffect::ContextualEffect(const ContextualEffect &first,
                                     const ContextualEffect &second)
      : experiment_effect_(first.experiment_effect_),
        context_effect_(first.context_effect_) {
    experiment_effect_.add_effect(second.experiment_effect_);
    context_effect_.add_effect(second.context_effect_);
  }

  int ContextualEffect::order() const {
    return experiment_effect_.order() + context_effect_.order();
  }

  int ContextualEffect::experiment_order() const {
    return experiment_effect_.order();
  }

  int ContextualEffect::context_order() const {
    return context_effect_.order();
  }

  void ContextualEffect::add_experiment_factor(const FactorDummy &factor) {
    experiment_effect_.add_factor(factor);
  }

  void ContextualEffect::add_context_factor(const FactorDummy &factor) {
    context_effect_.add_factor(factor);
  }

  bool ContextualEffect::shares_factors_with(
      const ContextualEffect &first_order_effect) const {
    int experiment_order = first_order_effect.experiment_order();
    int context_order = first_order_effect.context_order();
    if (experiment_order == 1 && context_order == 0) {
      return experiment_effect_.has_factor(
          first_order_effect.experiment_effect_.factor(0));
    } else if (experiment_order == 0 && context_order == 1) {
      return context_effect_.has_factor(
          first_order_effect.context_effect_.factor(0));
    } else {
      report_error(
          "Argument to ContextualEffect::shares_factors_with "
          "must be a first order ContextualEffect");
      return false;
    }
  }

  bool ContextualEffect::eval(const std::vector<int> &experiment_levels,
                              const std::vector<int> &context_levels) const {
    return experiment_effect_.eval(experiment_levels) &&
           context_effect_.eval(context_levels);
  }

  std::string ContextualEffect::name() const {
    if (order() == 0) {
      return experiment_effect_.name();
    } else if (experiment_effect_.order() == 0) {
      return context_effect_.name();
    } else if (context_effect_.order() == 0) {
      return experiment_effect_.name();
    } else {
      return experiment_effect_.name() + ":" + context_effect_.name();
    }
  }

  bool ContextualEffect::operator==(const ContextualEffect &rhs) const {
    return experiment_effect_ == rhs.experiment_effect_ &&
           context_effect_ == rhs.context_effect_;
  }

  bool ContextualEffect::operator<(const ContextualEffect &rhs) const {
    int cx = context_order();
    int rcx = rhs.context_order();
    // Effects with less context come before effects with more context.
    if (cx < rcx) {
      return true;
    } else if (cx > rcx) {
      return false;
    }

    // Now handle effects with the same amount of context...

    int ex = experiment_order();
    int rex = rhs.experiment_order();
    if (ex < rex) {
      return true;
    } else if (ex > rex) {
      return false;
    }

    // From here on out *this and rhs have the same amount of context
    // and the same amount of experimental stuff.
    if (experiment_effect_ < rhs.experiment_effect_) {
      return true;
    } else if (rhs.experiment_effect_ < experiment_effect_) {
      return false;
    }

    // Experiment effects are tied, so rank on the equal context effects.
    if (context_effect_ < rhs.context_effect_) {
      return true;
    } else if (rhs.context_effect_ < context_effect_) {
      return false;
    }

    // Everything is tied, so effects are the same.
    return false;
  }

  bool ContextualEffect::models_experiment_factor(
      int factor_position_in_input_data) const {
    return experiment_effect_.models_factor(factor_position_in_input_data);
  }

  bool ContextualEffect::models_context_factor(
      int factor_position_in_input_data) const {
    return context_effect_.models_factor(factor_position_in_input_data);
  }

  bool ContextualEffect::is_valid() const {
    return experiment_effect_.is_valid() && context_effect_.is_valid();
  }

  const FactorDummy &ContextualEffect::experiment_factor(
      int internal_factor_number) const {
    return experiment_effect_.factor(internal_factor_number);
  }

  const FactorDummy &ContextualEffect::context_factor(
      int internal_factor_number) const {
    return context_effect_.factor(internal_factor_number);
  }

  const FactorDummy &ContextualEffect::factor_dummy_for_experiment_factor(
      int factor_position_in_input_data) const {
    return experiment_effect_.factor_dummy_for_factor(
        factor_position_in_input_data);
  }

  const FactorDummy &ContextualEffect::factor_dummy_for_context_factor(
      int factor_position_in_input_data) const {
    return context_effect_.factor_dummy_for_factor(
        factor_position_in_input_data);
  }

  void ContextualEffect::set_levels(std::vector<int> &experiment_levels,
                                    std::vector<int> &context_levels) const {
    experiment_effect_.set_levels(experiment_levels);
    context_effect_.set_levels(context_levels);
  }

  void print(const ContextualEffect &e) { std::cout << e << std::endl; }

  //======================================================================
  EffectGroup::EffectGroup(int factor_position_in_input_data,
                           int number_of_levels,
                           const std::string &factor_name) {
    for (int i = 1; i < number_of_levels; ++i) {
      std::ostringstream name;
      name << factor_name << "." << i;
      effects_.push_back(
          Effect(FactorDummy(factor_position_in_input_data, i, name.str())));
    }
    std::sort(effects_.begin(), effects_.end());
  }

  EffectGroup::EffectGroup(int factor_position_in_input_data,
                           const std::vector<std::string> &level_names,
                           const std::string &factor_name) {
    int number_of_levels = level_names.size();
    for (int i = 1; i < number_of_levels; ++i) {
      std::ostringstream name;
      name << factor_name << "." << level_names[i];
      effects_.push_back(
          Effect(FactorDummy(factor_position_in_input_data, i, name.str())));
    }
    std::sort(effects_.begin(), effects_.end());
  }

  EffectGroup::EffectGroup(const EffectGroup &first,
                           const EffectGroup &second) {
    if (first == second) {
      // An interaction of a discrete factor with itself is
      // idempotent.
      effects_ = first.effects_;
    } else {
      for (int i = 0; i < first.effects_.size(); ++i) {
        for (int j = 0; j < second.effects_.size(); ++j) {
          Effect e(first.effects_[i], second.effects_[j]);
          if (e.is_valid()) {
            effects_.push_back(e);
          }
        }
      }
    }
    std::sort(effects_.begin(), effects_.end());
    std::vector<Effect>::iterator it =
        std::unique(effects_.begin(), effects_.end());
    if (it != effects_.end()) {
      effects_.erase(it);
    }
  }

  int EffectGroup::dimension() const { return effects_.size(); }

  void EffectGroup::fill_row(const std::vector<int> &input_data,
                             VectorView &output_row) const {
    if (output_row.size() != dimension()) {
      ostringstream err;
      err << "Size of output_row: " << output_row.size()
          << " does not match EffectGroup::dimension(): " << dimension() << "."
          << std::endl;
      report_error(err.str());
    }
    for (int i = 0; i < output_row.size(); ++i) {
      output_row[i] = effects_[i].eval(input_data);
    }
  }

  const std::vector<Effect> &EffectGroup::effects() const { return effects_; }

  //======================================================================
  ContextualEffectGroup::ContextualEffectGroup(
      int factor_position_in_input_data, int number_of_levels,
      const std::string &factor_name, bool is_context) {
    for (int i = 1; i < number_of_levels; ++i) {
      std::ostringstream name;
      name << factor_name << "." << i;
      effects_.push_back(ContextualEffect(
          FactorDummy(factor_position_in_input_data, i, name.str()),
          is_context));
    }
    std::sort(effects_.begin(), effects_.end());
  }

  ContextualEffectGroup::ContextualEffectGroup(
      int factor_position_in_input_data,
      const std::vector<std::string> &level_names,
      const std::string &factor_name, bool is_context) {
    int number_of_levels = level_names.size();
    for (int i = 1; i < number_of_levels; ++i) {
      std::ostringstream name;
      name << factor_name << "." << level_names[i];
      effects_.push_back(ContextualEffect(
          FactorDummy(factor_position_in_input_data, i, name.str()),
          is_context));
    }
    std::sort(effects_.begin(), effects_.end());
  }

  ContextualEffectGroup::ContextualEffectGroup(
      const ContextualEffectGroup &first, const ContextualEffectGroup &second) {
    if (first == second) {
      // An interaction of a discrete factor with itself is
      // idempotent.
      effects_ = first.effects_;
    } else {
      for (int i = 0; i < first.effects_.size(); ++i) {
        for (int j = 0; j < second.effects_.size(); ++j) {
          ContextualEffect e(first.effects_[i], second.effects_[j]);
          if (e.is_valid()) {
            effects_.push_back(e);
          }
        }
      }
    }
    std::sort(effects_.begin(), effects_.end());
    std::vector<ContextualEffect>::iterator it =
        std::unique(effects_.begin(), effects_.end());
    if (it != effects_.end()) {
      effects_.erase(it);
    }
  }

  int ContextualEffectGroup::dimension() const { return effects_.size(); }

  void ContextualEffectGroup::fill_row(
      const std::vector<int> &experiment_factors,
      const std::vector<int> &context_factors, VectorView &output_row) const {
    if (output_row.size() != dimension()) {
      ostringstream err;
      err << "Size of output_row: " << output_row.size()
          << " does not match ContextualEffectGroup::dimension(): "
          << dimension() << "." << std::endl;
      report_error(err.str());
    }
    for (int i = 0; i < output_row.size(); ++i) {
      output_row[i] = effects_[i].eval(experiment_factors, context_factors);
    }
  }

  const std::vector<ContextualEffect> &ContextualEffectGroup::effects() const {
    return effects_;
  }

  bool ContextualEffectGroup::operator==(
      const ContextualEffectGroup &rhs) const {
    return effects_ == rhs.effects_;
  }

  bool ContextualEffectGroup::operator<(
      const ContextualEffectGroup &rhs) const {
    if (effects_.size() < rhs.effects_.size()) {
      // Main effects and low order interactions come before higher
      // order interaction.
      return true;
    }
    if (effects_.size() > rhs.effects_.size()) {
      return false;
    } else {
      return effects_ < rhs.effects_;
    }
  }

  //======================================================================
  std::vector<EffectGroup> ExpandInteraction(
      const std::vector<EffectGroup> &first_set_of_effects,
      const std::vector<EffectGroup> &second_set_of_effects) {
    std::vector<EffectGroup> ans(first_set_of_effects);
    ans.insert(ans.end(), second_set_of_effects.begin(),
               second_set_of_effects.end());
    for (int i = 0; i < first_set_of_effects.size(); ++i) {
      for (int j = 0; j < second_set_of_effects.size(); ++j) {
        EffectGroup interaction(first_set_of_effects[i],
                                second_set_of_effects[j]);
        ans.push_back(interaction);
      }
    }
    return make_unique_preserve_order(ans);
  }

  std::vector<EffectGroup> ExpandInteraction(
      const std::vector<EffectGroup> &group, const EffectGroup &single_factor) {
    std::vector<EffectGroup> single_factor_group(1, single_factor);
    return ExpandInteraction(group, single_factor_group);
  }

  std::vector<EffectGroup> ExpandInteraction(
      const EffectGroup &single_factor, const std::vector<EffectGroup> &group) {
    std::vector<EffectGroup> single_factor_group(1, single_factor);
    return ExpandInteraction(single_factor_group, group);
  }

  //======================================================================
  std::vector<ContextualEffectGroup> ExpandInteraction(
      const std::vector<ContextualEffectGroup> &first_set_of_effects,
      const std::vector<ContextualEffectGroup> &second_set_of_effects) {
    std::vector<ContextualEffectGroup> ans(first_set_of_effects);
    ans.insert(ans.end(), second_set_of_effects.begin(),
               second_set_of_effects.end());
    for (int i = 0; i < first_set_of_effects.size(); ++i) {
      for (int j = 0; j < second_set_of_effects.size(); ++j) {
        ContextualEffectGroup interaction(first_set_of_effects[i],
                                          second_set_of_effects[j]);
        ans.push_back(interaction);
      }
    }
    return make_unique_preserve_order(ans);
  }

  std::vector<ContextualEffectGroup> ExpandInteraction(
      const std::vector<ContextualEffectGroup> &group,
      const ContextualEffectGroup &single_factor) {
    std::vector<ContextualEffectGroup> single_factor_group(1, single_factor);
    return ExpandInteraction(group, single_factor_group);
  }

  std::vector<ContextualEffectGroup> ExpandInteraction(
      const ContextualEffectGroup &single_factor,
      const std::vector<ContextualEffectGroup> &group) {
    std::vector<ContextualEffectGroup> single_factor_group(1, single_factor);
    return ExpandInteraction(single_factor_group, group);
  }

  //======================================================================
  RowBuilder::RowBuilder() {}

  RowBuilder::RowBuilder(const std::vector<EffectGroup> &effect_groups,
                         bool add_intercept) {
    std::set<Effect> already_seen;
    if (add_intercept) {
      Effect intercept;
      already_seen.insert(intercept);
      add_effect(intercept);
    }
    for (int group = 0; group < effect_groups.size(); ++group) {
      const std::vector<Effect> &effects(effect_groups[group].effects());
      for (int e = 0; e < effects.size(); ++e) {
        if (already_seen.find(effects[e]) == already_seen.end()) {
          already_seen.insert(effects[e]);
          add_effect(effects[e]);
        }
      }
    }
  }

  RowBuilder::RowBuilder(const ExperimentStructure &xp,
                         unsigned int interaction_order) {
    Effect intercept;
    add_effect(intercept);
    // If interaction_order == 0 then only 0'th order interactions are
    // possible.  That means just the intercept term.
    if (interaction_order == 0) return;
    if (interaction_order > xp.nfactors()) {
      interaction_order = xp.nfactors();
    }

    std::vector<Effect> main_effects;
    for (int factor = 0; factor < xp.nfactors(); ++factor) {
      for (int level = 1; level < xp.nlevels(factor); ++level) {
        FactorDummy dummy(factor, level,
                          xp.full_level_name(factor, level, "."));
        Effect main_effect(dummy);
        effects_.push_back(main_effect);
        main_effects.push_back(main_effect);
      }
    }

    int nmain = main_effects.size();
    std::vector<Effect> last_effects = main_effects;

    for (int order = 2; order <= interaction_order; ++order) {
      // last_effects contains all the (order - 1) way interactions.
      // Each step in the loop takes all the existing effects and
      // creates interactions with the main effects.  The results are
      // stored in current_effects, which then contain all the
      // "order"-way interactions.
      std::vector<Effect> current_effects;
      for (int i = 0; i < nmain; ++i) {
        Effect main_effect = main_effects[i];
        for (int j = 0; j < last_effects.size(); ++j) {
          Effect lower_order = last_effects[j];
          if (lower_order.has_factor(main_effect.factor(0))) {
            // Skip any interactions of a factor with itself.
            continue;
          }
          Effect interaction(main_effect, lower_order);
          if (has_effect(interaction)) {
            // Skip any interactions that are already part of this
            // RowBuilder object.
            continue;
          }
          current_effects.push_back(interaction);
          effects_.push_back(interaction);
        }
      }
      last_effects = current_effects;
    }
  }

  bool RowBuilder::has_effect(const Effect &effect) const {
    return std::find(effects_.begin(), effects_.end(), effect) !=
           effects_.end();
  }

  void RowBuilder::remove_effect(const Effect &effect) {
    std::vector<Effect>::iterator it =
        std::find(effects_.begin(), effects_.end(), effect);
    if (it != effects_.end()) {
      effects_.erase(it);
    }
  }

  void RowBuilder::add_effect(const Effect &e) { effects_.push_back(e); }

  void RowBuilder::add_effect_group(const EffectGroup &group) {
    const std::vector<Effect> &effects(group.effects());
    for (int i = 0; i < effects.size(); ++i) {
      add_effect(effects[i]);
    }
  }

  int RowBuilder::number_of_main_effects() const {
    int ans = 0;
    for (int i = 0; i < effects_.size(); ++i) {
      ans += (effects_[i].order() == 1);  // excludes intercept, with order()==0
    }
    return ans;
  }

  int RowBuilder::number_of_factors() const {
    std::set<int> factors;
    for (int i = 0; i < effects_.size(); ++i) {
      int order = effects_[i].order();
      for (int j = 0; j < order; ++j) {
        factors.insert(effects_[i].factor(j).factor());
      }
    }
    return factors.size();
  }

  std::vector<int> RowBuilder::main_effect_positions(int which_factor) const {
    std::vector<int> ans;
    for (int i = 0; i < effects_.size(); ++i) {
      if (effects_[i].order() == 1 &&
          effects_[i].factor(0).factor() == which_factor) {
        ans.push_back(i);
      }
    }
    return ans;
  }

  std::vector<std::vector<int>> RowBuilder::second_order_interaction_positions(
      int first_factor, int second_factor) const {
    if (first_factor == second_factor) {
      return std::vector<std::vector<int>>();
    }
    int max_first_factor_level = 0;
    int max_second_factor_level = 0;
    for (int i = 0; i < effects_.size(); ++i) {
      if (effects_[i].models_factor(first_factor)) {
        max_first_factor_level = std::max<int>(
            max_first_factor_level,
            effects_[i].factor_dummy_for_factor(first_factor).level());
      }
      if (effects_[i].models_factor(second_factor)) {
        max_second_factor_level = std::max<int>(
            max_second_factor_level,
            effects_[i].factor_dummy_for_factor(second_factor).level());
      }
    }

    // There may be no interaction between first_factor and
    // second_factor in the model, or there may only be interactions
    // between certain levels.  The default value in case of no
    // interaction is -1.
    std::vector<std::vector<int>> ans;
    ans.reserve(max_first_factor_level);
    for (int i = 0; i < max_first_factor_level; ++i) {
      ans.push_back(std::vector<int>(max_second_factor_level, -1));
    }

    bool found_an_interaction = false;
    for (int i = 0; i < effects_.size(); ++i) {
      if (effects_[i].order() == 2 && effects_[i].models_factor(first_factor) &&
          effects_[i].models_factor(second_factor)) {
        int first_level =
            effects_[i].factor_dummy_for_factor(first_factor).level();
        int second_level =
            effects_[i].factor_dummy_for_factor(second_factor).level();
        ans[first_level - 1][second_level - 1] = i;
        found_an_interaction = true;
      }
    }
    if (!found_an_interaction) {
      // Return an empty vector if there are no interactions between
      // these factors at all.
      std::vector<std::vector<int>> empty;
      return empty;
    }
    return ans;
  }

  const Effect &RowBuilder::effect(int i) const { return effects_[i]; }

  Vector RowBuilder::build_row(const std::vector<int> &levels) const {
    int neffects = effects_.size();
    Vector ans(neffects);
    for (int i = 0; i < neffects; ++i) ans[i] = effects_[i].eval(levels);
    return ans;
  }

  void RowBuilder::recover_configuration(
      const ConstVectorView &design_matrix_row,
      std::vector<int> &configuration) const {
    for (int i = 0; i < effects_.size(); ++i) {
      if (design_matrix_row[i] >= .9999) {
        effects_[i].set_levels(configuration);
      }
    }
  }

  void RowBuilder::recover_configuration(
      const Vector &design_matrix_row, std::vector<int> &configuration) const {
    recover_configuration(ConstVectorView(design_matrix_row), configuration);
  }

  int RowBuilder::dimension() const { return effects_.size(); }

  std::vector<std::string> RowBuilder::variable_names() const {
    std::vector<std::string> ans;
    int neffects = effects_.size();
    ans.reserve(neffects);
    for (int i = 0; i < neffects; ++i) {
      ans.push_back(effects_[i].name());
    }
    return ans;
  }

  //======================================================================
  ContextualRowBuilder::ContextualRowBuilder(const ExperimentStructure &xp,
                                             const ExperimentStructure &context,
                                             int interaction_order) {
    ContextualEffect intercept;
    contextual_effects_.push_back(intercept);
    if (interaction_order == 0) return;
    if (interaction_order > xp.nfactors() + context.nfactors()) {
      interaction_order = xp.nfactors() + context.nfactors();
    }

    std::vector<ContextualEffect> main_effects;
    for (int factor = 0; factor < xp.nfactors(); ++factor) {
      for (int level = 1; level < xp.nlevels(factor); ++level) {
        FactorDummy dummy(factor, level,
                          xp.full_level_name(factor, level, "."));
        ContextualEffect main_effect(dummy, false);
        main_effects.push_back(main_effect);
        contextual_effects_.push_back(main_effect);
      }
    }
    for (int factor = 0; factor < context.nfactors(); ++factor) {
      for (int level = 1; level < context.nlevels(factor); ++level) {
        FactorDummy dummy(factor, level,
                          context.full_level_name(factor, level, "."));
        ContextualEffect main_effect(dummy, true);
        main_effects.push_back(main_effect);
        contextual_effects_.push_back(main_effect);
      }
    }

    int number_of_main_effects = main_effects.size();

    std::vector<ContextualEffect> last_effects = main_effects;
    for (int order = 2; order <= interaction_order; ++order) {
      std::vector<ContextualEffect> current_effects;
      for (int i = 0; i < number_of_main_effects; ++i) {
        ContextualEffect main_effect = main_effects[i];
        for (int j = 0; j < last_effects.size(); ++j) {
          ContextualEffect lower_order = last_effects[j];
          if (lower_order.shares_factors_with(main_effect)) {
            continue;
          }
          ContextualEffect interaction(main_effect, lower_order);
          if (has_effect(interaction)) {
            continue;
          }
          current_effects.push_back(interaction);
          contextual_effects_.push_back(interaction);
        }
      }
      last_effects = current_effects;
    }
  }

  void ContextualRowBuilder::add_effect(const ContextualEffect &e) {
    contextual_effects_.push_back(e);
  }

  void ContextualRowBuilder::remove_effect(const ContextualEffect &effect) {
    std::vector<ContextualEffect>::iterator it = std::find(
        contextual_effects_.begin(), contextual_effects_.end(), effect);
    if (it != contextual_effects_.end()) {
      contextual_effects_.erase(it);
    }
  }

  void ContextualRowBuilder::add_effect_group(
      const ContextualEffectGroup &group) {
    const std::vector<ContextualEffect> &effects(group.effects());
    for (int i = 0; i < effects.size(); ++i) {
      add_effect(effects[i]);
    }
  }

  int ContextualRowBuilder::dimension() const {
    return contextual_effects_.size();
  }

  std::vector<std::string> ContextualRowBuilder::variable_names() const {
    std::vector<std::string> ans;
    ans.reserve(contextual_effects_.size());
    for (int i = 0; i < contextual_effects_.size(); ++i) {
      ans.push_back(contextual_effects_[i].name());
    }
    return ans;
  }

  std::vector<int> ContextualRowBuilder::main_effect_positions(
      int which_factor, bool contextual) const {
    std::vector<int> ans;
    for (int i = 0; i < contextual_effects_.size(); ++i) {
      const ContextualEffect &effect(contextual_effects_[i]);
      if (contextual) {
        // Check effect i to see if it is a contextual main effect
        // corresponding to contextual factor 'which_factor.'  If so,
        // record its position.
        if (effect.context_order() == 1 && effect.experiment_order() == 0 &&
            effect.models_context_factor(which_factor)) {
          ans.push_back(i);
        }
      } else {
        // Check effect i to see if it is an experimental main effect
        // modeling 'which_factor'.  If so, record its position.
        if (effect.context_order() == 0 && effect.experiment_order() == 1 &&
            effect.models_experiment_factor(which_factor)) {
          ans.push_back(i);
        }
      }
    }
    return ans;
  }

  // Find the largest observed factor level for the first and second
  // factor s.
  int ContextualRowBuilder::find_max_observed_level(int factor,
                                                    bool contextual) const {
    int max_level = 0;
    for (int i = 0; i < contextual_effects_.size(); ++i) {
      const ContextualEffect &effect(contextual_effects_[i]);
      if (contextual) {
        if (effect.models_context_factor(factor)) {
          max_level = std::max<int>(
              max_level,
              effect.factor_dummy_for_context_factor(factor).level());
        }
      } else {
        if (effect.models_experiment_factor(factor)) {
          max_level = std::max<int>(
              max_level,
              effect.factor_dummy_for_experiment_factor(factor).level());
        }
      }
    }
    return max_level;
  }

  namespace {

    // Returns true if the effect models a second order interaction
    // between the two factors.
    bool models_second_order_interaction(const ContextualEffect &effect,
                                         int first_factor,
                                         bool first_factor_is_contextual,
                                         int second_factor,
                                         bool second_factor_is_contextual) {
      if (effect.order() != 2) return false;

      bool models_first_factor =
          first_factor_is_contextual
              ? effect.models_context_factor(first_factor)
              : effect.models_experiment_factor(first_factor);

      bool models_second_factor =
          second_factor_is_contextual
              ? effect.models_context_factor(second_factor)
              : effect.models_experiment_factor(second_factor);

      return models_first_factor && models_second_factor;
    }

    std::pair<int, int> extract_levels(const ContextualEffect &effect,
                                       int first_factor,
                                       bool first_factor_is_contextual,
                                       int second_factor,
                                       bool second_factor_is_contextual) {
      int first_level;
      if (first_factor_is_contextual) {
        first_level =
            effect.factor_dummy_for_context_factor(first_factor).level();
      } else {
        first_level =
            effect.factor_dummy_for_experiment_factor(first_factor).level();
      }

      int second_level;
      if (second_factor_is_contextual) {
        second_level =
            effect.factor_dummy_for_context_factor(second_factor).level();
      } else {
        second_level =
            effect.factor_dummy_for_experiment_factor(second_factor).level();
      }
      return std::pair<int, int>(first_level, second_level);
    }

  }  // unnamed namespace

  std::vector<std::vector<int>>
  ContextualRowBuilder::second_order_interaction_positions(
      int first_factor, bool first_factor_is_contextual, int second_factor,
      bool second_factor_is_contextual) const {
    if (first_factor == second_factor &&
        first_factor_is_contextual == second_factor_is_contextual) {
      return std::vector<std::vector<int>>();
    }

    int max_first_factor_level =
        find_max_observed_level(first_factor, first_factor_is_contextual);
    int max_second_factor_level =
        find_max_observed_level(second_factor, second_factor_is_contextual);

    // Create a 'matrix' of int's filled with -1's.
    std::vector<std::vector<int>> ans;
    ans.reserve(max_first_factor_level);
    for (int i = 0; i < max_first_factor_level; ++i) {
      ans.push_back(std::vector<int>(max_second_factor_level, -1));
    }

    bool found_an_interaction = false;
    for (int i = 0; i < contextual_effects_.size(); ++i) {
      if (models_second_order_interaction(
              contextual_effects_[i], first_factor, first_factor_is_contextual,
              second_factor, second_factor_is_contextual)) {
        int first_level, second_level;
        std::tie(first_level, second_level) = extract_levels(
            contextual_effects_[i], first_factor, first_factor_is_contextual,
            second_factor, second_factor_is_contextual);
        found_an_interaction = true;
        ans[first_level - 1][second_level - 1] = i;
      }
    }
    if (!found_an_interaction) {
      // Return an empty vector if there are no interactions between
      // these factors at all.
      std::vector<std::vector<int>> empty;
      return empty;
    }
    return ans;
  }

  const ContextualEffect &ContextualRowBuilder::effect(int i) const {
    return contextual_effects_[i];
  }

  bool ContextualRowBuilder::has_effect(const ContextualEffect &effect) const {
    return std::find(contextual_effects_.begin(), contextual_effects_.end(),
                     effect) != contextual_effects_.end();
  }

  Vector ContextualRowBuilder::build_row(
      const std::vector<int> &experiment_levels,
      const std::vector<int> &context_levels) const {
    Vector ans(contextual_effects_.size());
    for (int i = 0; i < contextual_effects_.size(); ++i) {
      ans[i] = contextual_effects_[i].eval(experiment_levels, context_levels);
    }
    return ans;
  }

  bool ContextualRowBuilder::interaction_with_context() const {
    for (int i = 0; i < contextual_effects_.size(); ++i) {
      const ContextualEffect &e(contextual_effects_[i]);
      if (e.experiment_order() > 0 && e.context_order() > 0) {
        return true;
      }
    }
    return false;
  }

  Selector ContextualRowBuilder::pure_experiment() const {
    Selector ans(dimension(), false);
    for (int i = 0; i < contextual_effects_.size(); ++i) {
      if (contextual_effects_[i].context_order() == 0) {
        ans.add(i);
      } else {
        ans.drop(i);
      }
    }
    return ans;
  }

  Selector ContextualRowBuilder::pure_context() const {
    Selector ans(dimension(), false);
    for (int i = 0; i < contextual_effects_.size(); ++i) {
      if (contextual_effects_[i].context_order() > 0 &&
          contextual_effects_[i].experiment_order() == 0) {
        ans.add(i);
      } else {
        ans.drop(i);
      }
    }
    return ans;
  }

  Selector ContextualRowBuilder::experiment_context_interactions() const {
    Selector ans(dimension(), false);
    for (int i = 0; i < contextual_effects_.size(); ++i) {
      if (contextual_effects_[i].context_order() > 0 &&
          contextual_effects_[i].experiment_order() > 0) {
        ans.add(i);
      } else {
        ans.drop(i);
      }
    }
    return ans;
  }

  int ContextualRowBuilder::number_of_experimental_factors() const {
    int max_experimental_factor = -1;
    for (int i = 0; i < contextual_effects_.size(); ++i) {
      const ContextualEffect &effect(contextual_effects_[i]);
      for (int f = 0; f < effect.experiment_order(); ++f) {
        const FactorDummy &dummy(effect.experiment_factor(f));
        max_experimental_factor =
            std::max<int>(max_experimental_factor, dummy.factor());
      }
    }
    return max_experimental_factor + 1;
  }

  int ContextualRowBuilder::number_of_contextual_factors() const {
    int max_contextual_factor = -1;
    for (int i = 0; i < contextual_effects_.size(); ++i) {
      const ContextualEffect &effect(contextual_effects_[i]);
      for (int f = 0; f < effect.context_order(); ++f) {
        const FactorDummy &dummy(effect.context_factor(f));
        max_contextual_factor =
            std::max<int>(max_contextual_factor, dummy.factor());
      }
    }
    return max_contextual_factor + 1;
  }

  //======================================================================
}  // namespace BOOM
