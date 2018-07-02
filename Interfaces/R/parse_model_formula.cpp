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
#include <set>
#include <tuple>

#include "cpputil/make_unique_preserve_order.hpp"
#include "cpputil/report_error.hpp"

#include "r_interface/parse_model_formula.hpp"
#include <Rinternals.h>

namespace BOOM {
  namespace RInterface {
    namespace {
      template<class EFFECT_GROUP>
      struct NameStorage;

      // To parse the right hand side of an R model formula, create an
      // ExpressionFactory by passing in the data frame containing the
      // data in the formula.  Then call ExpandFormulaRHS() on the R
      // formula object comprising the right hand side of the formula to
      // be evaluated.
      template <class EFFECT_GROUP>
      class ExpressionFactory {
       public:
        // Constructor for ExpressionFactory<EffectGroup>
        // Args:
        //   r_frame: An R data frame containing the variables to be
        //     used as part of a model formula.
        explicit ExpressionFactory(SEXP r_frame);

        // Constructor for ExpressionFactory<BOOM::ContextualEffectGroup>
        ExpressionFactory(SEXP r_experiment_data,
                          SEXP r_context_data);

        std::vector<EFFECT_GROUP> ExpandFormulaRHS(SEXP r_expression);

       private:
        // Parse a single symbol r_expression, assumed to be a
        // variable name.  The name of the symbol must appear in names_.
        std::vector<EFFECT_GROUP>
        ParseSymbol(SEXP r_expression);

        // Parse an R formula expression where the first term is a
        // unary operator (assumed to be a paranthesis).
        std::vector<EFFECT_GROUP>
        ParseUnaryOperator(SEXP r_expression);

        //  Parse an R formula expression where the first term is a
        //  binary operator.
        std::vector<EFFECT_GROUP>
        ParseBinaryOperator(SEXP r_expression);

        // Append rhs to the end of lhs.
        void Concatenate(std::vector<EFFECT_GROUP> &lhs,
                         const std::vector<EFFECT_GROUP> &rhs) const {
          lhs.insert(lhs.end(), rhs.begin(), rhs.end());
        }

        // Args:
        //   level_names: on exit this will contain the level names
        //     for the factors in r_frame.
        //   r_frame: An R data frame containing only factors whose
        //     level names are to be extracted.
        void FillLevelNames(std::vector<std::vector<std::string>> &level_names,
                            SEXP r_frame) {
          int number_of_columns = Rf_length(r_frame);
          level_names.reserve(number_of_columns);
          for (int i = 0; i < number_of_columns; ++i) {
            level_names.push_back(BOOM::GetFactorLevels(VECTOR_ELT(
                r_frame, i)));
          }
        }
        NameStorage<EFFECT_GROUP> names_;
      };

      //======================================================================
      // Specializations for EffectGroup
      template<>
      struct NameStorage<EffectGroup> {
        std::vector<std::string> variable_names_;
        std::vector<std::vector<std::string> > level_names_;
      };

      // Args:
      //   r_frame:  An R data frame containing only factors.
      template<>
      ExpressionFactory<EffectGroup>::ExpressionFactory(SEXP r_frame) {
        names_.variable_names_ = getListNames(r_frame);
        FillLevelNames(names_.level_names_, r_frame);
      }

      // Args:
      //   r_expression is a single element, which is assumed to be a
      //     variable name.
      template<>
      std::vector<EffectGroup> ExpressionFactory<EffectGroup>::ParseSymbol(
          SEXP r_expression) {
        std::string variable_name = ToString(PRINTNAME(r_expression));
        if (variable_name == ".") {
          std::vector<EffectGroup> everything;
          everything.reserve(names_.variable_names_.size());
          for (int i = 0; i < names_.variable_names_.size(); ++i) {
            everything.push_back(EffectGroup(
                i,
                names_.level_names_[i],
                names_.variable_names_[i]));
          }
          return everything;
        }
        std::vector<std::string>::iterator it = std::find(
            names_.variable_names_.begin(), names_.variable_names_.end(),
            variable_name);
        if (it == names_.variable_names_.end()) {
          std::ostringstream err;
          err << "Variable name [" << variable_name << "] not found.";
          report_error(err.str());
        }
        int position = it - names_.variable_names_.begin();
        std::vector<EffectGroup> ans;
        ans.push_back(EffectGroup(position,
                                  names_.level_names_[position],
                                  variable_name));
        return ans;
      }

      //======================================================================
      // Specializations for ContextualEffectGroup
      template<>
      struct NameStorage<BOOM::ContextualEffectGroup> {
        std::vector<std::string> experiment_variable_names_;
        std::vector<std::vector<std::string> > experiment_level_names_;
        std::vector<std::string> context_variable_names_;
        std::vector<std::vector<std::string> > context_level_names_;

        // Finds a variable name in either the experimental or
        // contextual variable names.
        // Args:
        //   variable_name:  The name to find.
        //
        // Return:
        //   The return value is a tuple with the following elements.
        //   0) A boolean flag indicating whether the variable name was
        //      found among the context names (true) or experiment
        //      names (false).
        //   1) The position in the sequence of variable names where
        //      the variable name was found.
        //   2) A pointer to the level names for the found variable.
        std::tuple<bool, int, const std::vector<std::string>* >
        FindVariableName(const std::string &variable_name) const {
          bool context = false;
          int position = -1;
          const std::vector<std::string> *level_names;
          // Search for the variable name in the experiment variables.
          std::vector<std::string>::const_iterator it = std::find(
              experiment_variable_names_.cbegin(),
              experiment_variable_names_.cend(),
              variable_name);
          if (it != experiment_variable_names_.cend()) {
            position = it - experiment_variable_names_.cbegin();
            context = false;
            level_names = &(experiment_level_names_[position]);
            return std::make_tuple(context, position, level_names);
          }

          // Search for the variable name in the context variables.
          it = std::find(context_variable_names_.cbegin(),
                         context_variable_names_.cend(),
                         variable_name);
          if (it != context_variable_names_.cend()) {
            position = it - context_variable_names_.cbegin();
            context = true;
            level_names = &(context_level_names_[position]);
            return std::make_tuple(context, position, level_names);
          }

          std::ostringstream err;
          err << "Variable name [" << variable_name << "] not found.";
          report_error(err.str());
          return std::make_tuple(context, position, level_names);
        }
      };

      //----------------------------------------------------------------------
      // Args:
      //   r_experiment_data: An R data frame containing the
      //     experimental variables to be used as part of a model
      //     formula.
      //   r_context_data: An R data frame containing the contextual
      //     variables to be used as part of a model formula.
      template<>
      ExpressionFactory<BOOM::ContextualEffectGroup>::ExpressionFactory(
          SEXP r_experiment_data,
          SEXP r_context_data)
      {
        names_.experiment_variable_names_ = getListNames(r_experiment_data);
        names_.context_variable_names_ = getListNames(r_context_data);
        FillLevelNames(names_.experiment_level_names_, r_experiment_data);
        FillLevelNames(names_.context_level_names_, r_context_data);
      }

      //----------------------------------------------------------------------
      // Args:
      //   r_expression is a single element, which is assumed to be a
      //     variable name.
      template<>
      std::vector<BOOM::ContextualEffectGroup>
      ExpressionFactory<BOOM::ContextualEffectGroup>::ParseSymbol(
          SEXP r_expression) {
        std::string variable_name = ToString(PRINTNAME(r_expression));
        int position;
        bool context;
        const std::vector<std::string> *level_names;
        std::tie(context, position, level_names) =
            names_.FindVariableName(variable_name);
        std::vector<BOOM::ContextualEffectGroup> ans;
        ans.push_back(BOOM::ContextualEffectGroup(
            position,
            *level_names,
            variable_name,
            context));
        return ans;
      }

      //======================================================================
      // Handle unary operators.  This is mainly parentheses.
      // Args:
      //   r_expression: A piece of an R formula.
      template<class EFFECT_GROUP>
      std::vector<EFFECT_GROUP>
      ExpressionFactory<EFFECT_GROUP>::ParseUnaryOperator(SEXP r_expression) {
        SEXP operation = CAR(r_expression);
        if (!Rf_isSymbol(operation)) {
          report_error("Expected 'operation' to be a Symbol.");
        }
        if (CDR(r_expression) == nullptr) {
          report_error("The argument to the operation is missing!");
        }
        SEXP r_argument = CADR(r_expression);

        if (operation == Rf_install("(")) {
          return ExpandFormulaRHS(r_argument);
        } else {
          ostringstream err;
          err << "Unexpected unary operator: "
              << ToString(Rf_asChar(operation));
          report_error(err.str());
        }
        return std::vector<EFFECT_GROUP>();
      }

      //======================================================================
      template <class EFFECT_GROUP>
      std::vector<EFFECT_GROUP>
      ExpressionFactory<EFFECT_GROUP>::ParseBinaryOperator(SEXP r_expression) {
        SEXP operation = CAR(r_expression);
        if (!Rf_isSymbol(operation)) {
          report_error("Expected 'operation' to be a Symbol.");
        }
        if (CDR(r_expression) == nullptr) {
          report_error("Left hand side of the formula is missing!");
        }
        SEXP r_lhs_expression = CADR(r_expression);
        std::vector<EFFECT_GROUP> lhs_effects =
            ExpandFormulaRHS(r_lhs_expression);

        if (CDDR(r_expression) == nullptr) {
          // TODO(stevescott):  nullptr or NULL?
          report_error("Right hand side of the formula is missing!");
        }

        SEXP r_rhs_expression = CADDR(r_expression);

        if (operation == Rf_install("+")) {

          std::vector<EFFECT_GROUP> rhs_effects =
              ExpandFormulaRHS(r_rhs_expression);
          Concatenate(lhs_effects, rhs_effects);
          return lhs_effects;

        } else if (operation == Rf_install(":")) {

          std::vector<EFFECT_GROUP> rhs_effects =
              ExpandFormulaRHS(r_rhs_expression);
          if (lhs_effects.size() != 1
              || rhs_effects.size() != 1) {
            report_error("The colon operator can only be applied "
                         "to individual effects.");
          }
          return std::vector<EFFECT_GROUP>(
              1, EFFECT_GROUP(lhs_effects[0], rhs_effects[0]));

        } else if (operation == Rf_install("*")) {
          std::vector<EFFECT_GROUP> rhs_effects =
              ExpandFormulaRHS(r_rhs_expression);
          std::vector<EFFECT_GROUP> interaction =
              ExpandInteraction(lhs_effects, rhs_effects);
          Concatenate(lhs_effects, rhs_effects);
          Concatenate(lhs_effects, interaction);
          return make_unique_preserve_order(lhs_effects);

        } else if (operation == Rf_install("^")) {

          int power = Rf_asInteger(Rf_asChar(r_rhs_expression));
          std::vector<EFFECT_GROUP> elements =
              ExpandFormulaRHS(r_lhs_expression);
          std::vector<EFFECT_GROUP> interaction = elements;
          for (int i = 0; i < power - 1; ++i) {
            Concatenate(interaction,
                        ExpandInteraction(interaction, elements));
          }
          return make_unique_preserve_order(interaction);

        }
        return std::vector<EFFECT_GROUP>();
      }

      //======================================================================
      // Expands the right hand side of an R formula (the part after
      // the tilde) into a vector of BOOM::ContextualEffectGroup objects.
      // Args:
      //   r_expression:  The right hand side of an R model formula.
      template <class EFFECT_GROUP>
      std::vector<EFFECT_GROUP>
      ExpressionFactory<EFFECT_GROUP>::ExpandFormulaRHS(SEXP r_expression) {
        if (Rf_isSymbol(r_expression)) {
          return ParseSymbol(r_expression);
        } else if (Rf_isLanguage(r_expression)) {
          int expression_length = Rf_length(r_expression);
          if (expression_length == 2) {
            return ParseUnaryOperator(r_expression);
          } else if (expression_length == 3) {
            // Handle binary operators.
            return ParseBinaryOperator(r_expression);
          } else {
            ostringstream err;
            err << "Unexpected expression length:  " << expression_length;
            report_error(err.str());
          }
        }
        return std::vector<EFFECT_GROUP>();
      }

    } // namespace

    //======================================================================
    BOOM::RowBuilder ParseModelFormulaRHS(SEXP r_formula_rhs,
                                          SEXP r_frame,
                                          bool intercept) {
      BOOM::RInterface::ExpressionFactory<BOOM::EffectGroup> factory(r_frame);
      std::vector<BOOM::EffectGroup> effects =
          factory.ExpandFormulaRHS(r_formula_rhs);
      return BOOM::RowBuilder(effects, intercept);
    }

    BOOM::ContextualRowBuilder ParseContextualModelFormulaRHS(
        SEXP r_formula_rhs,
        SEXP r_experimental_factors_data_frame,
        SEXP r_contextual_factors_data_frame,
        bool intercept) {
      BOOM::RInterface::ExpressionFactory<BOOM::ContextualEffectGroup> factory(
          r_experimental_factors_data_frame,
          r_contextual_factors_data_frame);
      std::vector<BOOM::ContextualEffectGroup> factor_effects =
          factory.ExpandFormulaRHS(r_formula_rhs);
      BOOM::ContextualRowBuilder ans;
      std::set<BOOM::ContextualEffect> already_seen;
      if (intercept) {
        BOOM::ContextualEffect intercept_effect;
        ans.add_effect(intercept_effect);
        already_seen.insert(intercept_effect);
      }
      for (int group = 0; group < factor_effects.size(); ++group) {
        const std::vector<BOOM::ContextualEffect> &effects(
            factor_effects[group].effects());
        for (int e = 0; e < effects.size(); ++e) {
          if (already_seen.find(effects[e]) == already_seen.end()) {
            ans.add_effect(effects[e]);
            already_seen.insert(effects[e]);
          }
        }
      }
      return ans;
    }
    //======================================================================
    // Args:
    //   factors: A vector of factors.  Conceptually this is a data frame with
    //     all factor data.  All factors are assumed to have the same length.
    //   which_row: The row to extract from the notional data frame represented
    //     by the first argument.
    //
    // Returns:
    //   A vector of integers giving the level of each factor in the first
    //   argument, at the specified row.
    std::vector<int> ExtractRow(const std::vector<BOOM::Factor> &factors,
                                int which_row) {
      std::vector<int> ans(factors.size());
      for (int j = 0; j < factors.size(); ++j) {
        ans[j] = factors[j][which_row];
      }
      return ans;
    }

    //======================================================================
    // Args:
    //   r_factor_data_frame:  An R data frame where all columns are factors.
    //
    // Returns:
    //   A vector where each element is the BOOM representation of the columns
    //   in the data frame.
    std::vector<BOOM::Factor> ExtractFactors(SEXP r_factor_data_frame) {
      int nvars = Rf_length(r_factor_data_frame);
      std::vector<BOOM::Factor> ans;
      ans.reserve(nvars);
      for (int i = 0; i < nvars; ++i) {
        ans.push_back(BOOM::Factor(VECTOR_ELT(r_factor_data_frame, i)));
      }
      return ans;
    }

    //======================================================================
    // Args:
    //   r_formula_rhs: The right hand side of an R formula describing a model
    //     in terms of factor variables contained in r_data_frame.
    //   r_data_frame:  An R data frame containing all factor data.
    //   r_add_intercept: Scalar logical value indicating whether an intercept
    //     should be added to the formula specified in the first argument.
    //
    // Returns:
    //   An R matrix, including column names describing the variables, formed by
    //   expanding the data frame into a design matrix.
    SEXP BuildDesignMatrix(SEXP r_formula_rhs,
                           SEXP r_data_frame,
                           SEXP r_add_intercept) {
      bool add_intercept = Rf_asLogical(r_add_intercept);
      BOOM::RowBuilder row_builder = ParseModelFormulaRHS(
          r_formula_rhs, r_data_frame, add_intercept);
      std::vector<BOOM::Factor> factors = ExtractFactors(r_data_frame);
      int nobs = factors[0].length();
      BOOM::Matrix design_matrix(nobs, row_builder.dimension());
      for (int i = 0; i < nobs; ++i) {
        design_matrix.row(i) = row_builder.build_row(ExtractRow(factors, i));
      }
      return ToRMatrix(BOOM::LabeledMatrix(
          design_matrix,
          std::vector<std::string>(),
          row_builder.variable_names()));
    }
    //======================================================================
    // Args:
    //   r_formula_rhs: The right hand side of an R formula describing a model
    //     in terms of factor variables contained in r_data_frame.
    //   r_experiment_data_only_factors: An R data frame, with all rows being
    //     factors, representing the experimental variables in an experiment
    //     with both experimental and context factors.
    //   r_context_data_only_factors: An R data frame, with all rows being
    //     factors, representing the contextual variables in an experiment with
    //     both experimental and context factors.
    //   r_add_intercept: Scalar logical value indicating whether an intercept
    //     should be added to the formula specified in the first argument.
    //
    // Returns:
    //   An R matrix, including column names describing the variables, formed by
    //   expanding the data frame into a design matrix.
    SEXP BuildContextualDesignMatrix(
        SEXP r_formula_rhs,
        SEXP r_experiment_data_only_factors,
        SEXP r_context_data_only_factors,
        SEXP r_add_intercept) {
      bool add_intercept = Rf_asLogical(r_add_intercept);

      BOOM::ContextualRowBuilder row_builder = ParseContextualModelFormulaRHS(
          r_formula_rhs,
          r_experiment_data_only_factors,
          r_context_data_only_factors,
          add_intercept);
      std::vector<BOOM::Factor> experiment_factors =
          ExtractFactors(r_experiment_data_only_factors);
      std::vector<BOOM::Factor> context_factors = ExtractFactors(
          r_context_data_only_factors);
      int nobs = experiment_factors[0].length();
      BOOM::Matrix design_matrix(nobs, row_builder.dimension());
      for (int i = 0; i < nobs; ++i) {
        design_matrix.row(i) = row_builder.build_row(
            ExtractRow(experiment_factors, i),
            ExtractRow(context_factors, i));
      }
      return ToRMatrix(BOOM::LabeledMatrix(
          design_matrix,
          std::vector<std::string>(),
          row_builder.variable_names()));
    }

  }   // namespace RInterface
}  // namespace BOOM
