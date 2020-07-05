#ifndef BOOM_MIXED_DATA_IMPUTER_HPP_
#define BOOM_MIXED_DATA_IMPUTER_HPP_
/*
  Copyright (C) 2005-2020 Steven L. Scott

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
#include "Models/Impute/MvRegCopulaDataImputer.hpp"

namespace BOOM {

  namespace MixedImputation {

    // CompleteData represents the true error-corrected and imputed data values.
    class CompleteData : public Data {
     public:

      // Args:
      //   data:  The observed data to be corrected.
      CompleteData(const Ptr<MixedMultivariateData> &observed);
      CompleteData(const CompleteData &rhs);
      CompleteData & operator=(const CompleteData &rhs);

      CompleteData(CompleteData &&rhs) = default;
      CompleteData & operator=(CompleteData &&rhs) = default;

      CompleteData *clone() const override;
      std::ostream &display(std::ostream &out) const override;

      const MixedMultivariateData &observed_data() const {
        return *observed_data_;
      }

      // ======================================================================
      // Observed and latent data.
      // The numeric variables have 3 representations.
      // - The "observed" values are contained in y_observed.  These are the
      //   values witnessed in the raw data.
      Vector y_observed() const {return observed_data_->numeric_data();}

      // These are the imputed values, on their original scale.  They will often
      // agree with y_observed in cases where y_observed is obviously correct.
      // The non-atomic imputations will be inverse-transformed values of
      // y_numeric.
      const Vector &y_true() const {return y_true_;}
      void set_y_true(const Vector &y_true) {y_true_ = y_true;}
      void set_y_true(int i, double true_value) {y_true_[i] = true_value;}

      // y_numeric contains imputed values on the TRANSFORMED scale for the
      // non-atomic component of the data mixtuer.  These are the values
      // modeled by the "numeric data model" contained in MixedDataImputer.
      const Vector &y_numeric() const {return y_numeric_;}
      void set_y_numeric(const Vector &y_numeric) {y_numeric_ = y_numeric;}
      void set_y_numeric(int i, double numeric) {y_numeric_[i] = numeric;}

      // Levels of the categorical variables, expressed as integers.  These
      // correspond to the levels in the level key of the corresponding
      // categorical variable.
      const std::vector<int> true_categories() const {return true_categories_;}
      void set_true_categories(const std::vector<int> &truth) {
        true_categories_ = truth;
      }

      // Observed categorical data.  Missing values are indicated by the
      // "missing" mechanism in the Data base class.
      const std::vector<Ptr<CategoricalData>> observed_categories() const {
        return observed_categories_;
      }

      Vector &x() {return predictors_;}

     private:
      Ptr<MixedMultivariateData> observed_data_;

      // y_true_ holds the vector of imputed Y values for the numeric data.
      Vector y_true_;

      // y_numeric_ holds the output of the transformed regression model, on the
      // copula transformed scale.
      Vector y_numeric_;

      // For categorical variables, the entry indicates which category holds.
      std::vector<int> true_categories_;

      // The categorical data from observed_data_;
      std::vector<Ptr<CategoricalData>> observed_categories_;

      // The vector of predictors formed by encoding the categorical data.
      Vector predictors_;
    };

    //=========================================================================
    // Interface class describing on element of a collection of conditionally
    // independent distributions for elements of a mixed category data vector.
    class ErrorCorrectionModelBase : virtual public Model {
     public:

      // Args:
      //   index: The postion of the categorical or numeric variable being
      //     modeled.  Used to extract the relevant element from a row of
      //     MixedMultivariateData.
      ErrorCorrectionModelBase(int index)
          : index_(index)
      {}

      ErrorCorrectionModelBase *clone() const override = 0;

      // Pick the relevant entry out of 'data' and return its log density under
      // the model.
      virtual double logp(const MixedMultivariateData &data) const = 0;

      // The position (column number) of the variable that this model describes.
      int index() const {return index_;}

     private:
      int index_;
    };

    //-------------------------------------------------------------------------
    // A model for a possibly-erroneous single semicontinuous numeric entry in a
    // mixed data observation.  The model gives the marginal distribution of a
    // semicontinuous scalar with respect to a known collection of of fixed
    // "atoms".
    //
    // This class is a thin wrapper around the ErrorCorrectionModel found in
    // MvRegCopulaDataImputer.hpp.
    class NumericErrorCorrectionModel
        : public ErrorCorrectionModelBase,
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
      Matrix atom_error_probs() const {return impl_->atom_error_probs();}

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
        : public ErrorCorrectionModelBase,
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

      void sample_posterior() override;
      double logpri() const override;
      void clear_data() override;

      void update_complete_data_suf(int true_level, int observed_level);

      // The log conditional probability distribution of the true value, given
      // the obseved value that each atom is the true value.
      Vector true_level_log_probability(const CategoricalData &observed_value);

      // A mapping between a label or categorical data value and a column of the
      // "joint distribution" between the true and observed data.
      int atom_index(const CategoricalData &data) const;
      int atom_index(const std::string &label) const;

      void set_conjugate_prior_for_levels(const Vector &counts);
      void set_conjugate_prior_for_observations(const Matrix &counts);

      const Vector &level_probs() const {
        return truth_model_->pi();
      }

      Matrix level_observation_probs() const {
        Matrix ans(obs_models_.size(), obs_models_[0]->dim());
        for (int i = 0; i < ans.nrow(); ++i) {
          ans.row(i) = obs_models_[i]->pi();
        }
        return ans;
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
    struct InitInfo {
      Vector numeric_atoms;
      Ptr<CatKey> levels;
      Ptr<CatKey> categorical_atoms;
    };

    //==========================================================================
    class RowModel : public CompositeParamPolicy,
                     public NullDataPolicy,
                     public NullPriorPolicy {
     public:
      RowModel();
      RowModel(const RowModel &rhs);

      RowModel *clone() const override;

      void add_numeric(const Ptr<NumericErrorCorrectionModel> &model);
      void add_categorical(const Ptr<CategoricalErrorCorrectionModel> &model);

      double logp(const MixedMultivariateData &data) const;
      void clear_data() override;

      int ydim() const {return scalar_models_.size();}

      // Fill in the "true category" data member of 'row'.  The draw is made
      // conditional on the imputed y_numeric values, and on the mixture mixture
      // indicator.
      void impute_categorical(
          Ptr<MixedImputation::CompleteData> &row,
          RNG &rng,
          bool update_complete_data_suf,
          const Ptr<DatasetEncoder> &encoder,
          const std::vector<Ptr<EffectsEncoder>> &encoders,
          const Ptr<MultivariateRegressionModel> &numeric_model);

      void impute_atoms(
          Ptr<MixedImputation::CompleteData> &row,
          RNG & rng,
          bool update_complete_data_suf);

      // Return the requested numeric model.  The index counts only the numeric
      // variables, regardless of any intervening non-numeric variables.
      Ptr<NumericErrorCorrectionModel> numeric_model(int numeric_index) {
        return numeric_models_[numeric_index];
      }

      // Return the requested categorical model.  The index counts only the
      // categorical variables, regardless of any intervening non-categorical
      // variables.
      Ptr<CategoricalErrorCorrectionModel> categorical_model(
          int categorical_index) {
        return categorical_models_[categorical_index];
      }

      void sample_posterior() override;

     private:
      std::vector<Ptr<ErrorCorrectionModelBase>> scalar_models_;
      std::vector<Ptr<CategoricalErrorCorrectionModel>> categorical_models_;
      std::vector<Ptr<NumericErrorCorrectionModel>> numeric_models_;
    };

  }  // mixedimputation

  // A model responsible for imputing missing values and correcting likely
  // errors.
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
  class MixedDataImputer
      : public CompositeParamPolicy,
        public IID_DataPolicy<MixedMultivariateData>,
        public NullPriorPolicy
  {
   public:
    MixedDataImputer(int num_clusters,
                     const DataTable &table,
                     const std::vector<Vector> &atoms,
                     RNG &seeding_rng = GlobalRng::rng);

    MixedDataImputer(const MixedDataImputer &rhs);
    MixedDataImputer &operator=(const MixedDataImputer &rhs);
    MixedDataImputer(MixedDataImputer &&rhs) = default;
    MixedDataImputer &operator=(MixedDataImputer &&rhs) = default;
    MixedDataImputer *clone() const override;

    // Data management.
    void clear_data() override;
    void clear_client_data();
    void add_data(const Ptr<MixedMultivariateData> &data_point);

    void impute_row(Ptr<MixedImputation::CompleteData> &row,
                    RNG &rng,
                    bool update_complete_data_suf);
    void impute_row(Ptr<MixedImputation::CompleteData> &row,
                    RNG &rng) const;
    void impute_all_rows();

    void sample_posterior();

    Ptr<MultivariateRegressionModel> numeric_data_model() const {
      return numeric_data_model_;
    }
    int xdim() const { return numeric_data_model_->xdim();}
    int ydim() const { return numeric_data_model_->ydim();}
    int nclusters() const {return mixing_distribution_->dim();}

    Ptr<MultinomialModel> mixing_distribution() {
      return mixing_distribution_;
    }

    Ptr<MultivariateRegressionModel> numeric_data_model() {
      return numeric_data_model_;
    }

    Ptr<MixedImputation::RowModel> row_model(int cluster) {
      return mixture_components_[cluster];
    }

    RNG &rng() {return rng_;}

   private:
    int impute_cluster(Ptr<MixedImputation::CompleteData> &row,
                       RNG &rng,
                       bool update_complete_data_suf);
    int impute_cluster(Ptr<MixedImputation::CompleteData> &row, RNG &rng) const;

    void impute_numerics_given_atoms(Ptr<MixedImputation::CompleteData> &data,
                                     RNG &rng,
                                     bool update_complete_data_suf);

    void create_encoders(const DataTable &table);
    void initialize_empirical_distributions(const DataTable &data,
                                            const std::vector<Vector> &atoms);
    void initialize_regression_component();
    void initialize_mixture(int num_clusters,
                            const std::vector<Vector> &atoms,
                            const std::vector<Ptr<CatKey>> &levels,
                            const std::vector<VariableType> &variable_type);

    //===========================================================================
    // Data section
    //===========================================================================
    Ptr<MultinomialModel> mixing_distribution_;
    std::vector<Ptr<MixedImputation::RowModel>> mixture_components_;
    Ptr<MultivariateRegressionModel> numeric_data_model_;

    // The empirical distributions summarize the non-atomic parts of the
    // marginal distributions of the numeric variables.
    std::vector<IQagent> empirical_distributions_;

    // Encoders expand the categorical variables into dummy variables so they
    // can be used in the regression model.
    std::vector<Ptr<EffectsEncoder>> encoders_;
    Ptr<DatasetEncoder> encoder_;

    std::vector<Ptr<MixedImputation::CompleteData>> complete_data_;

    mutable RNG rng_;

    // ======================================================================
    // Mutable workspace
    // ======================================================================
    mutable SweptVarianceMatrix swept_sigma_;
    mutable bool swept_sigma_current_;
    void ensure_swept_sigma_current() const;

    // Set an observer that will flip swept_sigma_current_ to false when the
    // Sigma parameter changes.  This function is to be called during
    // construction.
    void set_observers();
    mutable Vector wsp_;
  };


}  // namespace BOOM

#endif  // BOOM_MIXED_DATA_IMPUTER_HPP_
