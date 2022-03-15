#ifndef BOOM_IMPUTATION_MIXED_DATA_IMPUTER_WITH_ERROR_CORRECTION_HPP_
#define BOOM_IMPUTATION_MIXED_DATA_IMPUTER_WITH_ERROR_CORRECTION_HPP_

/*
  Copyright (C) 2020 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "stats/DataTable.hpp"
#include "stats/Encoders.hpp"
#include "Models/Impute/MixedDataImputer.hpp"
#include "Models/Impute/MvRegCopulaDataImputer.hpp"

namespace BOOM {
  namespace MixedImputation {

    //-------------------------------------------------------------------------
    // A model for a possibly-erroneous single semicontinuous numeric entry in a
    // mixed data observation.  The model gives the marginal distribution of a
    // semicontinuous scalar with respect to a known collection of of fixed
    // "atoms".
    //
    // A thin wrapper around the ErrorCorrectionModel found in
    // MvRegCopulaDataImputer.hpp.
    class NumericErrorCorrectionModel
        : public ScalarModelBase,
          public CompositeParamPolicy,
          public NullDataPolicy,
          public NullPriorPolicy
    {
     public:
      // Args:
      //   index:  The position of the numeric variable to be modeled.
      //   atoms: A vector numeric values to receive special attention.  These
      //     are frequently occurring values that might also be errors.
      NumericErrorCorrectionModel(int index, const Vector &atoms);

      NumericErrorCorrectionModel(const NumericErrorCorrectionModel &rhs);
      NumericErrorCorrectionModel &operator=(
          const NumericErrorCorrectionModel &rhs);
      NumericErrorCorrectionModel(NumericErrorCorrectionModel &&rhs) = default;
      NumericErrorCorrectionModel &operator=(
          NumericErrorCorrectionModel &&rhs) = default;

      NumericErrorCorrectionModel *clone() const override;

      // The marginal log density of the appropriate element of 'data'.  This
      // density only measures the atom probabilities.  If a data element comes
      // from the continuous portion of the density then only the fact that it
      // is continuous contributes to the log probability.
      double logp(const MixedMultivariateData &data) const override;

      // The log density of the observed value.  Missing values are indicated by
      // NaN and can have positive density.
      double logp(double observed) const { return impl_->logp(observed);}

      VariableType variable_type() const override {
        return VariableType::numeric;
      }

      void set_conjugate_prior_for_true_categories(const Vector &prior_counts) {
        impl_->set_conjugate_prior_for_true_categories(prior_counts);
      }

      void set_conjugate_prior_for_observation_categories(
          const Matrix &prior_counts) {
        impl_->set_conjugate_prior_for_observation_categories(prior_counts);
      }

      void sample_posterior() override { impl_->sample_posterior(); }
      double logpri() const override {return impl_->logpri();}
      void clear_data() override {impl_->clear_data();}
      int impute_atom(double observed_value, RNG &rng, bool update) {
        return impl_->impute_atom(observed_value, rng, update);
      }

      const Vector &atom_probs() const {return impl_->atom_probs();}
      void set_atom_probs(const Vector &atom_probs) {
        impl_->set_atom_probs(atom_probs);
      }

      Matrix atom_error_probs() const {return impl_->atom_error_probs();}
      void set_atom_error_probs(const Matrix &atom_error_probs) {
        impl_->set_atom_error_probs(atom_error_probs);
      }

      double true_value(int true_atom, double observed) const {
        return impl_->true_value(true_atom, observed);
      }

      double numeric_value(int true_atom, double observed) const {
        return impl_->numeric_value(true_atom, observed);
      }

     private:
      Ptr<ErrorCorrectionModel> impl_;
    };

    //-------------------------------------------------------------------------
    class CategoricalErrorCorrectionModel
        : public ScalarModelBase,
          public CompositeParamPolicy,
          public NullDataPolicy,
          public NullPriorPolicy
    {
     public:
      // Args:
      //   key: The set of observed levels.  "Missing," as set by the Data base
      //     class, is an implicit level.
      //
      // The full set of observed values is the union of everything in "key"
      // with everything in "atoms" (there may be some overlap) and the missing
      // data flag in a data element.
      CategoricalErrorCorrectionModel(int index, const Ptr<CatKey> &levels);

      CategoricalErrorCorrectionModel(
          const CategoricalErrorCorrectionModel &rhs);
      CategoricalErrorCorrectionModel &operator=(
          const CategoricalErrorCorrectionModel &rhs);
      CategoricalErrorCorrectionModel(
          CategoricalErrorCorrectionModel &&rhs) = default;
      CategoricalErrorCorrectionModel &operator=(
          CategoricalErrorCorrectionModel &&rhs) = default;

      CategoricalErrorCorrectionModel *clone() const override;

      // The log density of the observed value, averaging over the atom
      // distribution.  If 'label' corresponds to an atom or a missing value it
      // may still have positive density.
      double logp(const std::string &label) const;

      double logp(const MixedMultivariateData &data) const override;

      void clear_data() override;
      void sample_posterior() override;
      double logpri() const override;
      VariableType variable_type() const override {
        return VariableType::categorical;
      }

      void update_complete_data_suf(int true_level, int observed_level);

      // The log conditional probability distribution of the true value, given
      // the obseved value that each atom is the true value.
      Vector true_level_log_probability(
          const LabeledCategoricalData &observed_value);

      // A mapping between a label or categorical data value and a column of the
      // "joint distribution" between the true and observed data.
      int atom_index(const LabeledCategoricalData &data) const;
      int atom_index(const std::string &label) const;

      void set_conjugate_prior_for_levels(const Vector &counts);
      void set_conjugate_prior_for_observations(const Matrix &counts);

      const Vector &level_probs() const {
        return truth_model_->pi();
      }
      void set_level_probs(const Vector &level_probs) {
        truth_model_->set_pi(level_probs);
      }

      Matrix level_observation_probs() const {
        Matrix ans(obs_models_.size(), obs_models_[0]->dim());
        for (int i = 0; i < ans.nrow(); ++i) {
          ans.row(i) = obs_models_[i]->pi();
        }
        return ans;
      }

      void set_level_observation_probs(const Matrix &level_observation_probs) {
        for (int i = 0; i < obs_models_.size(); ++i) {
          obs_models_[i]->set_pi(level_observation_probs.row(i));
        }
      }

     private:
      // The set of levels observed in the data.  The "missing" level is
      // implicitly present as well.  It is not included in the set of levels
      // contained in the key, but it is notionally tacked onto the end.
      Ptr<CatKey> levels_;

      // A mapping between level labels and the columns of the joint
      // distribution between true and observed data.
      std::map<std::string, int> atom_index_;

      // The marginal distribution of the true value.
      Ptr<MultinomialModel> truth_model_;

      // The conditional distribution of the observed data, given the true
      // value.  The index of the vector corresponds to the "true" value's index
      // in 'levels_'.
      std::vector<Ptr<MultinomialModel>> obs_models_;

      mutable Vector wsp_;

      // Rows are "true value" levels.  Columns are "observed_value" levels,
      // with the final column representing an implicit "missing" level.
      mutable Matrix log_joint_distribution_;
      mutable Vector log_marginal_observed_;

      // A flag to be set by observers.
      mutable bool workspace_is_current_;
      void ensure_workspace_is_current() const;

      // Functions to be called during construction.
      void set_observers();
      void build_atom_index();
    };

    //==========================================================================
    // Component model for MixedDataImputerWithErrorCorrection.  The model describes the
    // categorical component of an observation of mixed-type data.  This
    // includes the explicitly categorical variables, and the categorical
    // component of semicontinuous variables.
    class RowModelWithErrorCorrection : public RowModelBase
    {
     public:
      RowModelWithErrorCorrection();
      RowModelWithErrorCorrection(const RowModelWithErrorCorrection &rhs);
      RowModelWithErrorCorrection & operator=(
          const RowModelWithErrorCorrection &rhs);
      RowModelWithErrorCorrection *clone() const override;

      void add_numeric(const Ptr<NumericErrorCorrectionModel> &model);
      void add_categorical(const Ptr<CategoricalErrorCorrectionModel> &model);

      // Fill in the "true category" data member of 'row'.  The draw is made
      // conditional on the imputed y_numeric values, and on the mixture mixture
      // indicator.
      void impute_categorical(
          Ptr<MixedImputation::CompleteData> &row,
          RNG &rng,
          bool update_complete_data_suf,
          const Ptr<DatasetEncoder> &encoder,
          const std::vector<Ptr<EffectsEncoder>> &encoders,
          const Ptr<MultivariateRegressionModel> &numeric_model) override;

      // For numeric variables, impute the indicator variables describing which
      // atom is responsible for each variable.
      void impute_atoms(
          Ptr<MixedImputation::CompleteData> &row,
          RNG & rng,
          bool update_complete_data_suf) override;

      // Return the requested numeric model.  The index counts only the numeric
      // variables, regardless of any intervening non-numeric variables.
      Ptr<NumericErrorCorrectionModel> numeric_model(int numeric_index) {
        return numeric_ec_models_[numeric_index];
      }

      // Return the requested categorical model.  The index counts only the
      // categorical variables, regardless of any intervening non-categorical
      // variables.
      Ptr<CategoricalErrorCorrectionModel> categorical_model(
          int categorical_index) {
        return categorical_ec_models_[categorical_index];
      }

     private:
      // categorical_models_ and numeric_models_ contain the same pointers found
      // in scalar_models_, but they are separated out by type to avoid
      // downcasting.  For example categorical_models_[3] is the model for
      // categorical variable 3, which might be the 7th variable in the data
      // frame.
      std::vector<Ptr<CategoricalErrorCorrectionModel>> categorical_ec_models_;
      std::vector<Ptr<NumericErrorCorrectionModel>> numeric_ec_models_;

      // A utility to be called during copy construction and assignment.
      // Populate the type-specific data members based on the type-agnostic data
      // in the base class.
      void populate_ec_scalar_models();
    };


  }  // namespace MixedImputation

  //===========================================================================
  // A model for imputing missing values and correcting likely errors.
  //
  // The model has components for categorical and numeric varibles, with joint
  // distribution p(categorical) * p(numeric | categorical).  The categorical
  // component is a "pattern matching" mixture model.  The categorical variables
  // V1, ..., Vm are modeled as conditionally product multinomial given a
  // cluster indcator z.  That is, the conditional probability that V1 = v1,
  // ... Vm = vm given z is pi_1(z)[v1] * ... * pi_m(z)[vm].
  //
  // The conditional distribution of the numeric variables given the categorical
  // variables is a transformed multivariate regression.  If the vector of
  // numeric variables is Y, and the dummy variable expansion of the categorical
  // variables is X, then there is a transformation h such that h(Y) ~ N(X*beta,
  // Sigma).  The transformation we use is the Gaussian copula: h(Y_j) =
  // Phi^{-1}(F_j(Y_j)), where F_j is the empirical CDF of variable j.  This
  // produces h(Y) values that are marginally standard normal.
  //
  //
  class MixedDataImputerWithErrorCorrection : public MixedDataImputerBase
  {
   public:
    MixedDataImputerWithErrorCorrection(
        int num_clusters,
        const DataTable &table,
        const std::vector<Vector> &atoms,
        RNG &seeding_rng = GlobalRng::rng);

    MixedDataImputerWithErrorCorrection(
        const MixedDataImputerWithErrorCorrection &rhs);
    MixedDataImputerWithErrorCorrection &operator=(
        const MixedDataImputerWithErrorCorrection &rhs);
    MixedDataImputerWithErrorCorrection(
        MixedDataImputerWithErrorCorrection &&rhs) = default;
    MixedDataImputerWithErrorCorrection &operator=(
        MixedDataImputerWithErrorCorrection &&rhs) = default;
    MixedDataImputerWithErrorCorrection *clone() const override;

    // void impute_row(Ptr<MixedImputation::CompleteData> &row,
    //                 RNG &rng,
    //                 bool update_complete_data_suf) override;
    MixedImputation::RowModelWithErrorCorrection *
    row_model(int cluster) override {
      return mixture_components_[cluster].get();
    }

    const MixedImputation::RowModelWithErrorCorrection *
    row_model(int cluster) const override {
      return mixture_components_[cluster].get();
    }

    int number_of_mixture_components() const override {
      return mixture_components_.size();
    }

   private:
    void impute_numerics_given_atoms(Ptr<MixedImputation::CompleteData> &data,
                                     RNG &rng,
                                     bool update_complete_data_suf) override;

    void initialize_mixture(int num_clusters,
                            const std::vector<Vector> &atoms,
                            const std::vector<Ptr<CatKey>> &levels,
                            const std::vector<VariableType> &variable_type);

    // A utility to be called during copy construction.
    void populate_mixture_components();
    // ----------------------------------------------------------------------
    // Data section
    // ----------------------------------------------------------------------
    std::vector<Ptr<MixedImputation::RowModelWithErrorCorrection>>
    mixture_components_;
  };


}  // namespace BOOM




#endif  // BOOM_IMPUTATION_MIXED_DATA_IMPUTER_WITH_ERROR_CORRECTION_HPP_
