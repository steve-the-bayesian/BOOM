// Copyright 2018 Google LLC. All Rights Reserved.
#ifndef BOOM_DESIGN_HPP
#define BOOM_DESIGN_HPP
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

#include <limits>
#include <map>
#include <string>
#include <vector>

#include "uint.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Selector.hpp"

namespace BOOM {
  //======================================================================
  // A Configuration is a sequence of factor levels, represented by a
  // vector of integers, indicating the value of each experimental
  // factor in an experiment.  For example, if a button can be Red,
  // Green, or Blue, and its position can be Low or High, then the
  // Configuration has two factors with 3 and 2 potential levels.  The
  // possible values for the Configuration are {0, 0}, {0, 1}, {1, 0},
  // {1, 1}, {2, 0}, {2, 1}.
  //
  // In addition to its representation as a vector<int>, a
  // Configuration knows how to advance itself to the "next"
  // Configuration.  The sequence starts at {0, 0, 0,... 0} and ends
  // with each component at its maximum value.  Values on the right
  // (with high position numbers) are incremented most rapidly.
  class Configuration {
   public:
    // A new Configuration with all levels set to zero.
    // Args:
    //   factor_limits: The number of possible levels for a set of experimental
    //     factors.
    explicit Configuration(const std::vector<int> &factor_limits);

    // A new Configuration with levels set to specific values.
    // Args:
    //   factor_limits: The number of possible levels for a set of experimental
    //     factors.
    //   levels: A specific set of values for a specific experimental
    //     configuration.  In each ordinate i, we must have levels[i] >= 0 and
    //     levels[i] < factor_limits[i].
    Configuration(const std::vector<int> &factor_limits,
                  const std::vector<int> &levels);

    // Advance this configuration to the next one.  Factors with high
    // indices are incremented more rapidly than factors with low
    // indices.  If all levels are at their maximum values, then the
    // value at level 0 is set to -1, signaling one-past-the-end.
    void next();
    bool done() const;  // one past the end

    // Returns the value of a specific factor.
    int level(int factor) const;

    // Returns the values of all factors.
    const std::vector<int> &levels() const;

    // Two Configurations are equal if all their component data
    // (factors_ and levels_) are equal.
    bool operator==(const Configuration &rhs) const;
    bool operator!=(const Configuration &rhs) const;
    ostream &print(ostream &out) const;

   private:
    // The number of potential values available to each factor.
    std::vector<int> nlevels_;

    // The specific values for each factor in the current observation.
    std::vector<int> levels_;
  };

  inline ostream &operator<<(ostream &out, const Configuration &config) {
    return config.print(out);
  }

  //======================================================================
  // An ExperimentStructure records the names of the factors in an
  // experiment, as well as the names of all the levels in each
  // factor.  It is the meta-data for the 'raw data' in a data frame
  // that is to be turned into a design matrix by a RowBuilder.
  class ExperimentStructure {
   public:
    // Use this constructor if you know the number of levels for each
    // factor and don't care about names.  (The names will be
    // automatically generated).
    // Args:
    //   nlevels: A vector giving the number of distinct values
    //     possible for each experimental factor.
    //   context: Indicates whether this structure describes the contextual part
    //     of an experiment.  This only affects the variable names.
    explicit ExperimentStructure(const std::vector<int> &nlevels,
                                 bool context = false);

    // Use this constructor if you want to specify the experiment
    // structure using the names of the factors and levels, where the
    // names can occur in an arbitrary order.
    // Args:
    //   factor_names: gives the name of each factor in the experiment.
    //   level_names: gives the names of each of the levels for each
    //     factor.  Must satisfy level_names.size() ==
    //     factor_names.size().
    ExperimentStructure(
        const std::vector<std::string> &factor_names,
        const std::vector<std::vector<std::string> > &level_names);

    // The number of factors in the experiment.
    int nfactors() const;

    // The number of levels available for the given factor.
    int nlevels(int factor) const;

    // The number of levels available to all factors.
    const std::vector<int> &nlevels() const;

    // Returns the number of possible configurations in the
    // experiment, given by the product of the number of levels for
    // each factor.
    int nconfigurations() const;

    // The name of the specified level for the specified factor.
    const std::string &level_name(int factor, int level) const;

    // The name of the specified level for the specified factor.  The
    // factor name is prepended to the level name, separated by
    // 'separator'.
    std::string full_level_name(int factor, int level,
                                const std::string &separator = ".") const;

    const std::vector<std::string> &factor_names() const {
      return factor_names_;
    }

   private:
    std::vector<std::string> factor_names_;
    std::vector<std::vector<std::string> > level_names_;
    std::vector<int> nlevels_;
  };

  //======================================================================
  // A FactorDummy represents a dummy variable for a particular level
  // of a certain factor.  If a Configuration can be thought of as a
  // vector of integers giving the levels of one or more experimental
  // factors.
  //
  // An example of a FactorDummy is X(red), where X is a multi-level
  // factor and 'red' is one of its potential levels.
  class FactorDummy {
   public:
    // Args:
    //   factor_position_in_input_data: The position in the input data
    //     (a sequence of integers) representing the experimental
    //     factor that this object models.
    //   level: The specific value of the experimental factor to be
    //     matched by this object.
    //   name: The name of this FactorDummy (e.g. for use as a column
    //     heading in a design matrix).  For example:
    //     ThisFactor:ThisLevel.
    //
    // If factor or level is negative then this FactorDummy will
    // always evaluate to 0.
    FactorDummy(int factor_position_in_input_data, int level,
                const std::string &name);

    // Evaluate the dummy variable for a specific configuration of levels.
    // Args:
    //   levels: A vector of data, interpreted as the specific levels
    //     for a set of experimental factors.
    // Returns:
    //   This object monitors a specific position in 'levels' for a
    //   specific value.  Returns 'true' if the value is matched and
    //   'false' otherwise.
    bool eval(const std::vector<int> &levels) const;

    // The name of the factor and level this dummy variable
    // represents.  For example ButtonColor:blue.
    const std::string &name() const;

    // Two FactorDummy objects are equal if factor_ and level_ are the
    // same.  The name is not checked.
    bool operator==(const FactorDummy &rhs) const;
    bool operator!=(const FactorDummy &rhs) const { return !(*this == rhs); }

    // Objects are sorted first by factor, then by level.
    bool operator<(const FactorDummy &rhs) const;
    bool operator>(const FactorDummy &rhs) const {
      bool less_or_equal = *this < rhs || *this == rhs;
      return !less_or_equal;
    }

    // The factor (the position in the input data) that this
    // FactorDummy measures.
    int factor() const;

    // The specific level of the input factor that this dummy variable
    // screens for.
    int level() const;

    // Set configuration[factor_] = level_.  If configuration is not
    // long enough, it is resized.
    void set_level(std::vector<int> &configuration) const;

   private:
    int factor_;
    int level_;
    std::string name_;
  };

  //======================================================================
  // An effect models a single variable in a design matrix.  An Effect
  // can be an intercept, part of a main effect, or part of an
  // interation.  It a "functor" class that is the product of zero or
  // more FactorDummy's.
  //
  // An example of an Effect is the X.red*Y.left interaction (i.e. a
  // specific term in the interaction between X and Y), where X and Y
  // are multi-level factors.
  class Effect {
   public:
    // The empty Effect is the intercept.
    Effect();

    // Create an effect representing a particular level of a single
    // factor.
    explicit Effect(const FactorDummy &factor);

    // This effect is the interaction between first and second.
    Effect(const Effect &first, const Effect &second);

    // The number of factor dummies represented by this effect.
    int order() const;

    // If factor is not already part of this effect, add it to create
    // an interaction.
    void add_factor(const FactorDummy &factor);

    // Adds all the factors in the specified Effect to this Effect,
    // creating an interaction.
    void add_effect(const Effect &effect);

    // Returns true if all the factors that are part of this effect
    // are set to the specified levels.
    bool eval(const std::vector<int> &levels) const;

    // The name of this effect, determined by the component factors.
    std::string name() const;

    // Two effects are equal if they contain the same factors.  Order
    // is not relevant.
    bool operator==(const Effect &rhs) const;
    bool operator!=(const Effect &rhs) const { return !(*this == rhs); }

    // Effects are sorted lexicographically, first by lowest factor,
    // then by level of lowest factor.  If both tie then move to the
    // second factor, etc.
    bool operator<(const Effect &rhs) const;

    bool has_factor(const FactorDummy &f) const;

    // Returns true if any of the factor dummies have factor() ==
    // factor_position_in_input_data.
    bool models_factor(int factor_position_in_input_data) const;

    // An Effect is invalid if it has any FactorDummies that always
    // return 0.
    bool is_valid() const;

    // Args:
    //   internal_factor_number: The index of the factor stored by this object.
    //     This is the internal storage position 0, 1, 2, ... order()
    //     - 1.  It is NOT the position of the factor in the input
    //     data.  It is an error if internal_factor_number >= order().
    const FactorDummy &factor(int internal_factor_number) const;

    // Returns the factor dummy associated with the given factor
    // position.  It is an error to request a factor position not
    // managed by this object.
    const FactorDummy &factor_dummy_for_factor(
        int factor_position_in_input_data) const;

    // Set the appropriate entries in levels to the necessary values
    // to make this->eval(levels) == true.
    void set_levels(std::vector<int> &levels) const;

   private:
    std::vector<FactorDummy> factors_;
  };

  inline ostream &operator<<(ostream &out, const Effect &effect) {
    out << effect.name();
    return out;
  }

  void print(const Effect &e);
  //======================================================================
  // A ContextualEffect is an Effect that is (potentially) determined
  // by a combination of two data sources: contextual data as well as
  // experimental data.
  class ContextualEffect {
   public:
    // An empty ContextualEffect is the intercept.
    ContextualEffect();

    // Create a ContextualEffect from a single FactorDummy.
    // Args:
    //   factor: A FactorDummy describing a factor to be modeled.
    //   is_context: If 'true' then 'factor' refers to contextual
    //     data.  Otherwise it refers to experimental data.
    ContextualEffect(const FactorDummy &factor, bool is_context);

    // Create a ContextualEffect from a standard Effect.
    ContextualEffect(const Effect &effect, bool is_context);

    // Create an interaction between two contextual effects.
    ContextualEffect(const ContextualEffect &first,
                     const ContextualEffect &second);

    // The number of factor dummies represented by this effect, across
    // both experimental and contextual factors.
    int order() const;
    int experiment_order() const;
    int context_order() const;

    void add_experiment_factor(const FactorDummy &experiment_factor);
    void add_context_factor(const FactorDummy &context_factor);

    // If two ContextualEffect objects share a factor then they should
    // not be in an interaction together.  If the shared factors are
    // for the same level then they will collapse, which is okay for
    // eval(), but order() will be wrong.  If the shared factors have
    // different levels then the resulting interaction would always
    // evaluate to 0.
    bool shares_factors_with(const ContextualEffect &first_order_effect) const;

    // Returns true if all the factors that are part of this effect
    // are set to the specified levels.
    bool eval(const std::vector<int> &experiment_configuration,
              const std::vector<int> &context) const;

    // The name of this effect, determined by its component factors.
    std::string name() const;

    // Two effects are equal if they contain the same factor dummies
    // (i.e. if they match the same set of factors and levels).  Order
    // is not relevant.
    bool operator==(const ContextualEffect &rhs) const;
    bool operator!=(const ContextualEffect &rhs) const {
      return !(*this == rhs);
    }

    // Effects are sorted lexicographically, first by lowest factor,
    // then by level of lowest factor.  If both tie then move to the
    // second factor, etc.  This is done first for the experiment
    // factors, then for the context factors.
    bool operator<(const ContextualEffect &rhs) const;

    // Returns true if any of the factor dummies have factor() ==
    // factor_position_in_input_data.
    bool models_experiment_factor(int factor_position_in_input_data) const;
    bool models_context_factor(int factor_position_in_input_data) const;

    // An Effect is invalid if it has any FactorDummies that always
    // return 0.
    bool is_valid() const;

    // Args:
    //   internal_factor_number: The index of the factor stored by this object.
    //     This is the internal storage position 0, 1, 2, ... order()
    //     - 1.  It is NOT the position of the factor in the input
    //     data.  It is an error if internal_factor_number >= order().
    const FactorDummy &experiment_factor(int internal_factor_number) const;
    const FactorDummy &context_factor(int internal_factor_number) const;

    // Returns the factor dummy associated with the given factor
    // position.  It is an error to request a factor position not
    // managed by this object.
    const FactorDummy &factor_dummy_for_experiment_factor(
        int factor_position_in_input_data) const;
    const FactorDummy &factor_dummy_for_context_factor(
        int factor_position_in_input_data) const;

    // Set the appropriate entries in levels to the necessary values
    // to make this->eval(experiment_levels, context_levels) == true.
    void set_levels(std::vector<int> &experiment_levels,
                    std::vector<int> &context_levels) const;

   private:
    Effect experiment_effect_;
    Effect context_effect_;
  };

  inline ostream &operator<<(ostream &out, const ContextualEffect &e) {
    return out << e.name();
  }

  void print(const ContextualEffect &e);

  //======================================================================
  // An EffectGroup exists to model factor-level effects.  Its job is
  // to translate a factor or multi-factor interaction into a
  // collection of dummy variables.
  //
  // An example of an EffectGroup is the X*Y*Z interaction where X, Y,
  // and Z are multi-level factors.
  class EffectGroup {
   public:
    // Build a set of dummy variables for a multi-level factor.
    // Args:
    //   factor_position_in_input_data: The position in the input data
    //     holding the factor this object is modeling.
    //   number_of_levels: The number of levels the factor can assume.
    //     It is assumed the factor ranges from 0 to number_of_levels
    //     - 1.
    //   factor_name: The name of the variable that the factor is
    //     modeling.  Used to label output.
    EffectGroup(int factor_position_in_input_data, int number_of_levels,
                const std::string &factor_name);

    // Build a set of dummy variables for a multi-level factor by
    // providing the names of all the factor levels.
    // Args:
    //   factor_position_in_input_data: The position in the input data
    //     holding the factor this object is modeling.
    //   level_names: The names of the level values the factor can
    //     assume.  The data supplied to this object must range
    //     between 0 and level_names.size() - 1.
    //   factor_name: The name of the variable that the factor is
    //     modeling.  Used to label output.
    EffectGroup(int factor_position_in_input_data,
                const std::vector<std::string> &level_names,
                const std::string &factor_name);

    // TODO: EffectGroup may one day be changed to model an additive collection
    // of effects.  When that happens the following constructor could be changed
    // and renamed: static EffectGroup Interaction(EffectGroup &first,
    // EffectGroup &second);
    //
    // Create an interaction between two effect groups.  Every term in
    // first is multiplied by every term in second.
    // The order of the effects is first.0:second.0, first.0:second.1, ...
    EffectGroup(const EffectGroup &first, const EffectGroup &second);

    // The number of columns required to represent this EffectGroup in
    // a design matrix.
    int dimension() const;

    // Fills in this EffectGroup's subset of the design matrix row.
    // Args:
    //   input_data: The full row of data to be converted into a
    //     design matrix row.
    //   output_row: The subset of a design matrix row to be filled by
    //     *this.
    void fill_row(const std::vector<int> &input_data,
                  VectorView &output_row) const;

    const std::vector<Effect> &effects() const;

    // The set of effects_ managed by an EffectGroup is kept sorted by
    // the constructor, so the check for equality can be passed
    // through to the data.
    bool operator==(const EffectGroup &rhs) const {
      return effects_ == rhs.effects_;
    }
    bool operator!=(const EffectGroup &rhs) const { return !(*this == rhs); }

    // Sort first by size, then lexicographically within groups of the
    // same size.
    bool operator<(const EffectGroup &rhs) const {
      if (effects_.size() < rhs.effects_.size()) {
        // Main effects and low order interaactions come before higher
        // order interaction.
        return true;
      }
      if (effects_.size() > rhs.effects_.size()) {
        return false;
      } else {
        return effects_ < rhs.effects_;
      }
    }

   private:
    std::vector<Effect> effects_;
  };

  //======================================================================
  // A ContextualEffectGroup exists to model factor-level effects.
  // Its job is to translate a factor or multi-factor interaction into
  // a collection of dummy variables.
  //
  // An example of a ContextualEffectGroup is the X*Y*Z interaction
  // where X, Y, and Z are multi-level factors.
  class ContextualEffectGroup {
   public:
    // A set of dummies modeling a single factor.
    // Args:
    //   factor_position_in_input_data: The position of the factor to
    //     be modeled in the input data.
    //   number_of_levels: The number of levels the factor can assume.
    //   factor_name: The name of the factor being modeled.
    //   is_context: If true then treat the factor as contextual.  If
    //     false, treat it as experimental.
    ContextualEffectGroup(int factor_position_in_input_data,
                          int number_of_levels, const std::string &factor_name,
                          bool is_context);

    // A set of dummies for a single factor.
    // Args:
    //   factor_position_in_input_data: The position of the factor to
    //     be modeled in the input data.
    //   level_names:  The names of the levels this factor can assume.
    //   factor_name: The name of the factor being modeled.
    //   is_context: If true then treat the factor as contextual.  If
    //     false, treat it as experimental.
    ContextualEffectGroup(int factor_position_in_input_data,
                          const std::vector<std::string> &level_names,
                          const std::string &factor_name, bool is_context);

    // An interaction between two existing ContextualEffectGroups.
    // The first might represent (X+Y).  The second might be (Z + W).
    // Then this constructor would create (X*Z + Y*Z + X*W + Y*W).
    ContextualEffectGroup(const ContextualEffectGroup &first,
                          const ContextualEffectGroup &second);

    // The total number of effects being modeled.
    int dimension() const;

    // Fills in this ContextualEffectGroup's subset of the design matrix row.
    // Args:
    //   experimental_factors: The experimental configuration being
    //     observed.
    //   context_factor: The state of the contextual factors during
    //     the observation.
    //   output_row: The subset of a design matrix row to be filled
    //     in.
    void fill_row(const std::vector<int> &experiment_factors,
                  const std::vector<int> &context_factors,
                  VectorView &output_row) const;

    // The vector of ContextualEffect's managed by this object.
    const std::vector<ContextualEffect> &effects() const;

    // Two ContextualEffectGroup objects are equal if their vector of
    // effects() is equal.
    bool operator==(const ContextualEffectGroup &rhs) const;
    bool operator!=(const ContextualEffectGroup &rhs) const {
      return !(*this == rhs);
    }

    // *this < rhs if its dimension is smaller.  If the dimension is
    // *the same size then effects are compared lexicographically.
    bool operator<(const ContextualEffectGroup &rhs) const;

   private:
    std::vector<ContextualEffect> effects_;
  };

  //======================================================================
  // Produce an interaction of the form (x + y) * (z + w), where x, y,
  // z, and w are potentially multi-level factors.  The set of effects
  // in the output would be
  // [x, y, z, w, x*z, x*w, y*z, y*w]
  //
  // Args:
  //   first_set_of_effects: A vector of EffectGroups whose elements
  //     define distinct groups of columns in a design matrix.  In the
  //     R formula language these elements define effects connected by '+'.
  //   second_set_of_effects: A vector of EffectGroups analogous to
  //     first_set_of_effects.
  // Returns:
  //   The vector of EffectGroups corresponding to
  //   first_set_of_effects * second_set_of_effects.
  std::vector<EffectGroup> ExpandInteraction(
      const std::vector<EffectGroup> &first_set_of_effects,
      const std::vector<EffectGroup> &second_set_of_effects);

  // As above, but with with contextual effects.
  std::vector<ContextualEffectGroup> ExpandInteraction(
      const std::vector<ContextualEffectGroup> &first_set_of_effects,
      const std::vector<ContextualEffectGroup> &second_set_of_effects);

  // TODO: The notion of an EffectGroup should probably be expanded to mean what
  // is currently std::vector<EffectGroup>, in which case ExpandInteraction
  // should be made part of the EffectGroup class.

  // Produce an interaction of the form (x + y) * z, where x, y, and z
  // are potentially multi-level factors.  The set of effects produced
  // in this example is
  // x, y, z, x*z, y*z
  std::vector<EffectGroup> ExpandInteraction(
      const std::vector<EffectGroup> &group, const EffectGroup &single_factor);

  // As above, but with contextual effects.
  std::vector<ContextualEffectGroup> ExpandInteraction(
      const std::vector<ContextualEffectGroup> &group,
      const ContextualEffectGroup &single_factor);

  // Produce an interaction of the form x * (y + z), where x, y, and z
  // are potentially multi-level factors.  The set of effects produced
  // in this example is x, y, z, x*y, x*z.
  std::vector<EffectGroup> ExpandInteraction(
      const EffectGroup &single_factor, const std::vector<EffectGroup> &group);

  // As above, but with contextual effects.
  std::vector<ContextualEffectGroup> ExpandInteraction(
      const ContextualEffectGroup &single_factor,
      const std::vector<ContextualEffectGroup> &group);

  //======================================================================
  // A RowBuilder converts a vector of integers, representing levels
  // of experimental factors, into the corresponding row in a design
  // matrix.
  class RowBuilder {
   public:
    // Default constructor creates an empty RowBuilder.  Any effects
    // must be added by add_effect().
    RowBuilder();

    // This constructor provides low-level control over the structure
    // of the experiment.
    explicit RowBuilder(const std::vector<EffectGroup> &effects,
               bool add_intercept = true);

    // Take all the factors in an experiment and combine them up to
    // 'interaction_order'.
    // Args:
    //   xp: Describes the set of factors defining the experiment.
    //   interaction_order: the number of factors involved in the
    //     highest order interaction.  I.e. interaction_order = 2
    //     means all two-factor interactions will be included.
    //     interaction_order = 0 just provides an intercept term.  If
    //     interaction_order equals or exceeds the number of factors
    //     in xp a saturated design is produced.
    RowBuilder(const ExperimentStructure &xp, unsigned int interaction_order);

    // Adds an individual effect to the set of effects defining the
    // design matrix.  This adds a single column to the design matrix
    // modeled by the RowBuilder.
    void add_effect(const Effect &e);

    // Adds the effects in 'group' to the set of effects defining the
    // design matrix.  This adds one or more columns corresponding to
    // the EffectGroup to the design matrix modeled by the RowBuilder.
    void add_effect_group(const EffectGroup &group);

    // Checks whether 'e' is in the set of effect defining the design
    // matrix.
    bool has_effect(const Effect &e) const;

    // If the given effect is present then remove it.  If it is not
    // present do nothing.
    void remove_effect(const Effect &e);
    void remove_intercept() { remove_effect(Effect()); }

    // The number of columns in the design matrix corresponding to
    // main effects.  The intercept is not a main effect.
    int number_of_main_effects() const;

    // Returns the number of factors considered by this object.  This
    // is less than or equal to the dimension to the argument to
    // build_row(), because some factors might be ignored.
    int number_of_factors() const;

    // Returns the positions in the design matrix of the main effects
    // corresponding to the given factor.
    // Args:
    //   which_factor:  The position in the input data of the desired factor.
    // Returns:
    //   A vector of positions (in the output of build_row)
    //   corresponding to the dummy variables modeling the main
    //   effects for 'which_factor'.  If 'which_factor' is not part of
    //   the model then an empty vector is returned.
    std::vector<int> main_effect_positions(int which_factor) const;

    // Returns a 'matrix' of positions in the design matrix
    // corresponding to the interaction effects between first_factor
    // and second_factor.
    // Args:
    //   first_factor:  The position in the input data of the first factor.
    //   second_factor:  The position in the input data of the second factor.
    // Returns:
    //   A vector of vectors where element [i][j] gives the position
    //   in the design matrix of the interaction between level i-1 of
    //   first_factor and level j-1 of second_factor.  All the 'inner'
    //   vectors are the same size.  If the two factors do not
    //   interact then an empty vector is returned.  If two levels do
    //   not interact then that position in the matrix is recorded as
    //   -1.
    std::vector<std::vector<int> > second_order_interaction_positions(
        int first_factor, int second_factor) const;

    // An accessor returning the i'th effect.
    const Effect &effect(int i) const;

    // Produce a row of a design matrix (or "model matrix")
    // corresponding to a given configuration of input data.
    Vector build_row(const std::vector<int> &levels) const;

    // Given a row of a design matrix build using build_row, compute
    // the reverse map.
    // Args:
    //   design_matrix_row:  A vector of 0's and 1's.
    //   configuration: On output, this->build_row(configuration) ==
    //     design_matrix_row.
    void recover_configuration(const ConstVectorView &design_matrix_row,
                               std::vector<int> &configuration) const;
    void recover_configuration(const Vector &design_matrix_row,
                               std::vector<int> &configuration) const;

    // The number of variables in the row to be built.
    int dimension() const;

    // The name of each variable in output produced by build_row().
    std::vector<std::string> variable_names() const;

   private:
    std::vector<Effect> effects_;
  };

  //======================================================================
  // A ContextualRowBuilder does the same thing as a RowBuilder, but
  // it considers both experimental and contextual factors.
  // Experimental factors are factors whose levels are under the
  // control of the experimenter, like whether the button on your web
  // page is red or blue.  Contextual factors are factors that are not
  // under the experimenter's control, like whether the visit to your
  // web site occurs on a weekend or a week day.
  class ContextualRowBuilder {
   public:
    // An empty ContextualRowBuilder does nothing useful.  It has
    // dimension zero and produces empty design matrix rows.  It does
    // not include an intercept term.  To add an intercept, add a
    // null-constructed ContextualEffect.
    ContextualRowBuilder() {}

    // A ContextualRowBuilder formed by taking all
    // interaction_order-way interactions between all input variables.
    // Args:
    //   experiment:  The experimental factors being modeled.
    //   context:  The contextual factors being modeled.
    //   interaction_order: The maximum number of factors involved in
    //     an interaction term.  E.g. 2 means "all 2-way interactions".
    ContextualRowBuilder(const ExperimentStructure &experiment,
                         const ExperimentStructure &context,
                         int interaction_order);

    // Adds an effect to this RowBuilder.  This is effectively the
    // same thing as adding another column to the design matrix that
    // this object will produce.
    void add_effect(const ContextualEffect &effect);

    // If the given effect is present then remove it.  If it is not
    // present do nothing.
    void remove_effect(const ContextualEffect &effect);
    void remove_intercept() { remove_effect(ContextualEffect()); }

    // Adds the effects in 'group' to the set of effects defining the
    // design matrix.  This adds one or more columns corresponding to
    // the EffectGroup to the design matrix modeled by the RowBuilder.
    void add_effect_group(const ContextualEffectGroup &group);

    // The size of the row to be built;
    int dimension() const;

    // The name of each variable in output produced by build_row().
    std::vector<std::string> variable_names() const;

    // Returns the positions in the design matrix of the main effects
    // corresponding to the given factor.
    // Args:
    //   which_factor:  The position in the input data of the desired factor.
    //   contextual: If true then 'which_factor' refers to a
    //     contextual factor.  Otherwise 'which_factor' refers to an
    //     experimental factor.
    // Returns:
    //   A vector of positions (in the output of 'build_row()')
    //   corresponding to the dummy variables modeling the main
    //   effects for 'which_factor'.  If 'which_factor' is not part of
    //   the model then an empty vector is returned.
    std::vector<int> main_effect_positions(int which_factor,
                                           bool contextual) const;

    // Returns a 'matrix' of positions in the design matrix
    // corresponding to the interaction effects between first_factor
    // and second_factor.
    // Args:
    //   first_factor: The position in the input data of the first
    //     factor in the interaction.
    //   first_factor_is_contextual: If true then look for
    //     'first_factor' in the set of contextual factors.  Otherwise
    //     look in the set of experimental factors.
    //   second_factor: The position in the input data of the second
    //     factor in the interaction.
    //   second_factor_is_contextual: If true then look for
    //     'second_factor' in the set of contextual factors.
    //     Otherwise look in the set of experimental factors.
    //
    // Returns:
    //   A vector of vectors where element [i][j] gives the position
    //   in the design matrix of the interaction between level i-1 of
    //   first_factor and level j-1 of second_factor.  All the 'inner'
    //   vectors are the same size (i.e. the matrix has a fixed number
    //   of columns).  If the two factors do not interact then an
    //   empty vector is returned.  If two levels do not interact then
    //   that position in the matrix is recorded as -1.
    std::vector<std::vector<int> > second_order_interaction_positions(
        int first_factor, bool first_factor_is_contextual, int second_factor,
        bool second_factor_is_contextual) const;

    // Returns i'th effect managed by this object.
    const ContextualEffect &effect(int i) const;

    // Checks whether this object manages the specified effect.
    bool has_effect(const ContextualEffect &effect) const;

    // Produces a design matrix row corresponding to the given set of
    // experimental and factor levels.
    Vector build_row(const std::vector<int> &experiment_levels,
                     const std::vector<int> &context_levels) const;

    // Returns true if there are interaction terms involving both
    // experimental and contextual effects.  Returns false otherwise.
    bool interaction_with_context() const;

    // Returns a selector that identifies which elements of a Vector
    // built by build_row() correspond purly to experimental factors.
    Selector pure_experiment() const;

    // Returns a selector that identifies which elements of a Vector
    // built by build_row() correspond purly to contextual factors.
    Selector pure_context() const;

    // Returns a selector that identifies which elements of a Vector
    // built by build_row() correspond interactions between
    // experimental and contextual factors.
    Selector experiment_context_interactions() const;

    // Returns the minimal number of experimental factors necessary to
    // use this row builder.
    int number_of_experimental_factors() const;

    // Returns the minimal number of contextual factors necessary to
    // use this row builder.
    int number_of_contextual_factors() const;

   private:
    std::vector<ContextualEffect> contextual_effects_;

    // Find largest observed level for the given factor.
    // Args:
    //   factor:  The factor to search for.
    //   contextual: If true then look for 'factor' in the set of
    //     contextual factors.  Otherwise look in the experimental
    //     factors.
    // Returns:
    //   The largest observed level for the specified factor.
    int find_max_observed_level(int factor, bool contextual) const;
  };

  //======================================================================
  // Generates a design matrix consisting of an intercept and dummy
  // variables for all effects up to the specified order.  If order =
  // 0 just the intercept is used.  If order = 1 then main effects are
  // added.  If order = 2 then second order interactions are added.
  // Etc.
  LabeledMatrix generate_design_matrix(
      const std::map<std::string, std::vector<std::string> > &level_names,
      int interaction_order);

  LabeledMatrix generate_design_matrix(const ExperimentStructure &xp,
                                       const RowBuilder &builder);

  // Generates a design matrix corresponding to all possible values of
  // a given set of contexts.
  LabeledMatrix generate_contextual_design_matrix(
      const ExperimentStructure &context_structure,
      const ContextualRowBuilder &row_builder);

  LabeledMatrix generate_experimental_design_matrix(
      const ExperimentStructure &xp, const ContextualRowBuilder &row_builder);

}  // namespace BOOM

#endif  // BOOM_DESIGN_HPP
