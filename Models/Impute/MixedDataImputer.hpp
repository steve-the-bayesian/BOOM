#ifndef BOOM_MIXED_DATA_IMPUTER_HPP_
#define BOOM_MIXED_DATA_IMPUTER_HPP_
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
#include "stats/summary.hpp"
#include "Models/Impute/MvRegCopulaDataImputer.hpp"

namespace BOOM {

  namespace MixedImputation {

    // CompleteData represents the true error-corrected and imputed data values
    // in a data set with missing or erroneous data.
    //
    //
    class CompleteData : public Data {
     public:
      // Args:
      //   data:  The observed data to be corrected.
      explicit CompleteData(const Ptr<MixedMultivariateData> &observed);
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
      const std::vector<Ptr<LabeledCategoricalData>>
      observed_categories() const {
        return observed_categories_;
      }

      // The dummy-variable encoded categorical data used to predict numeric
      // outcomes from categorical ones.
      Vector &x() {return predictors_;}

      // Fill a specific row of the data table with the imputed values for this
      // observation.
      //
      // Args:
      //   table:  The table to be filled.
      //   row:  The index of the row to be filled.
      void fill_data_table_row(DataTable &table, int row);

      // Fill a MixedMultivariateData object with the imputed values.
      MixedMultivariateData to_mixed_multivariate_data() const;

     private:
      Ptr<MixedMultivariateData> observed_data_;

      // y_true_ holds the vector of imputed Y values for the numeric data.
      Vector y_true_;

      // y_numeric_ holds the output of the transformed regression model, on the
      // copula transformed scale.
      Vector y_numeric_;

      // true_categories_[i] indicates the category responsible for the ith
      // categorical variable.  The length of this vector matches
      // 'observed_categories_'.
      std::vector<int> true_categories_;

      // The categorical data from observed_data_.  There is one element for
      // each categorical variable.
      std::vector<Ptr<LabeledCategoricalData>> observed_categories_;

      // The vector of predictors formed by encoding the categorical data.
      Vector predictors_;
    };

    //=========================================================================
    // Interface describing a single variable in a mixed category data vector.
    class ScalarModelBase : virtual public Model {
     public:
      // Args:
      //   index: The postion of the categorical or numeric variable being
      //     modeled.  Used to extract the relevant element from a row of
      //     MixedMultivariateData.
      ScalarModelBase(int index)
          : index_(index)
      {}

      ScalarModelBase *clone() const override = 0;

      // Pick the relevant entry out of 'data' and return its log density under
      // the model.  If the scalar value is missing, then return the expected
      // value of logp (aka the entropy).
      virtual double logp(const MixedMultivariateData &data) const = 0;

      // The position (column number) of the variable that this model describes.
      int index() const {return index_;}

      // The type of variable the scalar model describes.
      virtual VariableType variable_type() const = 0;

     private:
      int index_;
    };

    // -------------------------------------------------------------------------
    // A model describing the categorical component of a semicontinuous numeric
    // variable.
    //
    // The model is multinomial, with explicit categories given by specific
    // numeric values ("atoms") known a priori, and an implicit "numeric"
    // category indicating the response obeys a multivariate regression
    // relationship with the categorical variables.
    class NumericScalarModel
        : public ScalarModelBase,
          public CompositeParamPolicy,
          public NullDataPolicy,
          public NullPriorPolicy
    {
     public:
      NumericScalarModel(int index, const Vector &atoms);
      NumericScalarModel(const NumericScalarModel &rhs);
      NumericScalarModel(NumericScalarModel &&rhs) = default;
      NumericScalarModel & operator=(const NumericScalarModel &rhs);
      NumericScalarModel & operator=(NumericScalarModel &&rhs) = default;
      NumericScalarModel * clone() const override;

      // Return the log probability of this observation's atom.  If the
      // observation is missing then the expected logp (entropy) is returned.
      double logp(const MixedMultivariateData &data) const override;
      double logp(double observed) const;

      // The dimension of the conjugate prior is K+1 where K is the number of
      // atoms.  The final dimension indicates the state of being determined by
      // the regression model.  At least one element of counts must be positive,
      // but any element can be zero.
      void set_conjugate_prior(const Vector &counts);

      void sample_posterior() override {atom_model_->sample_posterior();}
      double logpri() const override {return atom_model_->logpri();}
      VariableType variable_type() const override {
        return VariableType::numeric;
      }

      // Return the atom responsible for the observed value.  If the observed
      // value is missing then impute using the atom_model_.
      int impute_atom(double observed_value, RNG &rhg, bool update);

      // Return the index of the atom to which value belongs.  This value can be
      // atoms_.size() if 'value' does not correspond to any atom.
      // I.e. atoms._size() indicates the value is from the numeric component.
      // It can also be -1 if 'value' is NaN.
      int category_map(double value) const;

      // Return the value of y_true given the atom responsible for y_true.  This
      // is either equal to the atom value, or to the observed numeric value.
      // The latter might be NaN, in which case NaN is returned.
      virtual double true_value(int true_atom, double observed) const;

      // Return the value of y_numeric.  This is either NaN, or the observed
      // value.
      virtual double numeric_value(int true_atom, double observed) const;

      const Vector &atom_probs() const {return atom_model_->pi();}
      void set_atom_probs(const Vector &probs) {
        atom_model_->set_pi(probs);
      }

     private:
      Ptr<MultinomialModel> atom_model_;
      Vector atoms_;
    };

    //-------------------------------------------------------------------------
    // A multinomial distribution describing data in a single categorical
    // variable.
    class CategoricalScalarModel
        : public ScalarModelBase,
          public CompositeParamPolicy,
          public NullDataPolicy,
          public NullPriorPolicy
    {
     public:
      CategoricalScalarModel(int index, const Ptr<CatKey> &levels);
      CategoricalScalarModel(const CategoricalScalarModel &rhs);
      CategoricalScalarModel(CategoricalScalarModel &&rhs) = default;
      CategoricalScalarModel & operator=(const CategoricalScalarModel &rhs);
      CategoricalScalarModel & operator=(
          CategoricalScalarModel &&rhs) = default;
      CategoricalScalarModel * clone() const override;

      double logp(const std::string &label) const;
      double logp(const MixedMultivariateData &data) const override;
      const Vector &log_probs() const {return model_->logpi();}

      void sample_posterior() override {model_->sample_posterior();}
      double logpri() const override {return model_->logpri();}
      void clear_data() override {model_->clear_data();}
      VariableType variable_type() const override {
        return VariableType::categorical;
      }

      void update_complete_data_suf(int observed_level);

      void set_conjugate_prior(const Vector &counts);
      const Vector &level_probs() const { return model_->pi(); }
      void set_level_probs(const Vector &probs) {
        model_->set_pi(probs);
      }

     private:
      // The set of levels observed in the data.
      Ptr<CatKey> levels_;

      // A mapping between level labels and numeric index values.
      //
      // The atom index may also contain a missing data symbol.  If so it will
      // be associated to the value -1.
      std::map<std::string, int> atom_index_;

      // The marginal distribution of the data.
      Ptr<MultinomialModel> model_;

      void build_atom_index();
      int atom_index(const std::string &label) const;
    };

    //==========================================================================
    struct InitInfo {
      Vector numeric_atoms;
      Ptr<CatKey> levels;
      Ptr<CatKey> categorical_atoms;
    };

    //==========================================================================
    class RowModelBase
        : public CompositeParamPolicy,
          public NullDataPolicy,
          public NullPriorPolicy
    {
     public:
      RowModelBase();
      RowModelBase(const RowModelBase &rhs);
      RowModelBase(RowModelBase &&rhs) = default;
      RowModelBase & operator=(const RowModelBase &rhs);
      RowModelBase & operator=(RowModelBase &&rhs) = default;
      RowModelBase * clone() const override = 0;

      void add_scalar_model(const Ptr<ScalarModelBase> &model);

      double logp(const MixedMultivariateData &data) const;
      void clear_data() override;
      void sample_posterior() override;

      // For numeric variables, impute the latent variables indicating which
      // atom is responsible for each variable.
      virtual void impute_atoms(
          Ptr<CompleteData> &row,
          RNG & rng,
          bool update_complete_data_suf) = 0;

      // Impute missing values for categorical variables.  The imputation is
      // done conditional on numeric data.
      //
      // Args:
      //   row:  The data to be imputed.
      //   rng:  The random number generator used to generate imputed values.
      //   update_complete_data_suf: If true, then the imputed, complete data
      //     values are added to the complete data sufficient statistics for the
      //     relevant component models.  If false then no update is done.
      //   encoder: A dataset encoder for transforming 'row' into a set of
      //     predictors.
      //   encoders: The specific encoders that are components of 'encoder'.
      //   numeric_model: The model used to predict the regression component of
      //     the numeric variables, given categorical variables.
      //
      // Effects:
      //   (1) 'row->set_true_categories' is called with the imputed categorical
      //     data.
      //   (2) If 'update_complete_data_suf' is true, the complete data
      //     sufficient statistics of the component models are updated with the
      //     complete data.
      virtual void impute_categorical(
          Ptr<CompleteData> &row,
          RNG &rng,
          bool update_complete_data_suf,
          const Ptr<DatasetEncoder> &encoder,
          const std::vector<Ptr<EffectsEncoder>> &encoders,
          const Ptr<MultivariateRegressionModel> &numeric_model) = 0;

     protected:
      std::vector<Ptr<ScalarModelBase>> scalar_models() {
        return scalar_models_;
      }

      const std::vector<Ptr<ScalarModelBase>> scalar_models() const {
        return scalar_models_;
      }

     private:
      std::vector<Ptr<ScalarModelBase>> scalar_models_;
    };

    //==========================================================================
    class RowModel : public RowModelBase
    {
     public:
      RowModel();
      RowModel(const RowModel &rhs);
      RowModel(RowModel &&rhs) = default;
      RowModel & operator=(const RowModel &rhs);
      RowModel & operator=(RowModel &&rhs) = default;
      RowModel *clone() const override;

      void add_numeric(const Ptr<NumericScalarModel> &model);
      void add_categorical(const Ptr<CategoricalScalarModel> &model);

      int ydim() const {return numeric_models_.size();}

      // Impute missing values for categorical variables.  The imputation is
      // done conditional on numeric data.
      //
      // Args:
      //   row:  The data to be imputed.
      //   rng:  The random number generator used to generate imputed values.
      //   update_complete_data_suf: If true, then the imputed, complete data
      //     values are added to the complete data sufficient statistics for the
      //     relevant component models.  If false then no update is done.
      //   encoder: A dataset encoder for transforming 'row' into a set of
      //     predictors.
      //   encoders: The specific encoders that are components of 'encoder'.
      //   numeric_model: The model used to predict the regression component of
      //     the numeric variables, given categorical variables.
      //
      // Effects:
      //   (1) 'row->set_true_categories' is called with the imputed categorical
      //     data.
      //   (2) If 'update_complete_data_suf' is true, the complete data
      //     sufficient statistics of the component models are updated with the
      //     complete data.
      void impute_categorical(
          Ptr<MixedImputation::CompleteData> &row,
          RNG &rng,
          bool update_complete_data_suf,
          const Ptr<DatasetEncoder> &encoder,
          const std::vector<Ptr<EffectsEncoder>> &encoders,
          const Ptr<MultivariateRegressionModel> &numeric_model) override;

      // For numeric variables, impute the latent variables indicating which
      // atom is responsible for each variable.
      void impute_atoms(Ptr<CompleteData> &row,
                        RNG & rng,
                        bool update_complete_data_suf) override;

      Ptr<NumericScalarModel> numeric_model(int numeric_index) {
        return numeric_models_[numeric_index];
      }

      Ptr<CategoricalScalarModel> categorical_model(int categorical_index) {
        return categorical_models_[categorical_index];
      }

     private:
      std::vector<Ptr<CategoricalScalarModel>> categorical_models_;
      std::vector<Ptr<NumericScalarModel>> numeric_models_;

      // A utility to be called during copy construction.  Fill in the entries
      // of numeric_models_ and categorical_models_ from scalar_models_ using
      // type deduction.
      void populate_numeric_and_categorical_models();
    };

  }  // namespace MixedImputation

  //===========================================================================
  // There are at least 2 versions of "mixed data imputer" classes that differ
  // in the way they treat categorical variables.  This base class combines
  // their common code.
  class MixedDataImputerBase
      : public CompositeParamPolicy,
        public IID_DataPolicy<MixedMultivariateData>,
        public NullPriorPolicy
  {
   public:
    MixedDataImputerBase(int num_clusters,
                         const DataTable &data,
                         const std::vector<Vector> &atoms,
                         RNG &seeding_rng = GlobalRng::rng);
    MixedDataImputerBase(const MixedDataImputerBase &rhs);
    MixedDataImputerBase & operator=(const MixedDataImputerBase &rhs);
    MixedDataImputerBase(MixedDataImputerBase &&rhs) = default;
    MixedDataImputerBase & operator=(MixedDataImputerBase &&rhs) = default;

    MixedDataImputerBase * clone() const override = 0;
    // Setup functions that require virtual functions.  Clients should call this
    // function immediately after construction.
    void initialize(const std::vector<Vector> &atoms);

    //--------------------------------------------------------------------------
    // Data management.
    //--------------------------------------------------------------------------
    void clear_data() override;
    void clear_client_data();
    using IID_DataPolicy::add_data;
    void add_data(const Ptr<MixedMultivariateData> &data_point) override;

    // The dimension of the dummy-expanded categorical variables.
    int xdim() const { return numeric_data_model_->xdim();}

    // Dimension of the numeric variables.
    int ydim() const { return numeric_data_model_->ydim();}

    //--------------------------------------------------------------------------
    // Handling imputation.
    //--------------------------------------------------------------------------
    void impute_data_set(
        std::vector<Ptr<MixedImputation::CompleteData>> &rows);

    virtual void impute_row(Ptr<MixedImputation::CompleteData> &row,
                            RNG &rng,
                            bool update_complete_data_suf);
    void impute_all_rows();

    void sample_posterior() override;

    //--------------------------------------------------------------------------
    // Accessing the component models.
    //--------------------------------------------------------------------------
    Ptr<MultivariateRegressionModel> numeric_data_model() const {
      return numeric_data_model_;
    }
    int nclusters() const {return mixing_distribution_->dim();}

    Ptr<MultinomialModel> mixing_distribution() {
      return mixing_distribution_;
    }

    Ptr<MultivariateRegressionModel> numeric_data_model() {
      return numeric_data_model_;
    }

    // Child classes will return the correct model.  The stored model will be
    // stored by a Ptr, but the return value is a raw pointer, which should be
    // caught by a Ptr unless expected to be ephemeral.
    virtual MixedImputation::RowModelBase * row_model(int cluster) = 0;
    virtual const MixedImputation::RowModelBase * row_model(int cluster) const = 0;
    virtual int number_of_mixture_components() const = 0;

    RNG &rng() {return rng_;}

    const IQagent & empirical_distribution(int i) const {
      return empirical_distributions_[i];
    }

    // Sample means of the observed numeric variables.  Missing data are ignored
    // on a variable-by-variable basis.
    Vector ybar() const;

   protected:
    void ensure_swept_sigma_current() const;
    SweptVarianceMatrix & swept_sigma() {return swept_sigma_;}

    IQagent & empirical_distribution(int i) {
      return empirical_distributions_[i];
    }

    // Re-initialie the numeric_data_model_ using the specified model.
    void set_numeric_data_model(
        const Ptr<MultivariateRegressionModel> &model) {
      numeric_data_model_ = model;
      set_numeric_data_model_observers();
    }

   private:
    // Simulate the cluster (i.e. the mixture component index) responsible for
    // the observation supplied in the first argument.
    //
    // Args:
    //   row:  The observation.
    //   rng:  The random number generator to use for the simulation.
    //   update_complete_data_suf: Indicates whether the complete data
    //     sufficient statistics should be updated in light of the draw.  True
    //     for training data, false for imputation after training.
    int impute_cluster(const Ptr<MixedImputation::CompleteData> &row,
                       RNG &rng,
                       bool update_complete_data_suf);
    int impute_cluster(const Ptr<MixedImputation::CompleteData> &row,
                       RNG &rng) const;

    // Fill numeric_summaries_ and categorical_summaries_ with summaries of the
    // supplied data.
    void summarize_data(const DataTable &data);

    virtual void impute_numerics_given_atoms(Ptr<MixedImputation::CompleteData> &data,
                                             RNG &rng,
                                             bool update_complete_data_suf) = 0;

    void clear_mixture_component_data();

    void create_encoders(const DataTable &table);
    void initialize_empirical_distributions(const DataTable &data,
                                            const std::vector<Vector> &atoms);
    void initialize_regression_component();

    // -------------------------------------------------------------------------
    // Data section
    // -------------------------------------------------------------------------
    DataTypeIndex data_types_;
    Ptr<MultinomialModel> mixing_distribution_;
    Ptr<MultivariateRegressionModel> numeric_data_model_;

    // The empirical distributions summarize the non-atomic parts of the
    // marginal distributions of the numeric variables.
    std::vector<IQagent> empirical_distributions_;

    // Summaries of the numeric and categorical training data.
    std::map<std::string, NumericSummary> numeric_summaries_;
    std::map<std::string, CategoricalSummary> categorical_summaries_;

    // Encoders expand the categorical variables into dummy variables so they
    // can be used in the regression model.
    std::vector<Ptr<EffectsEncoder>> encoders_;
    Ptr<DatasetEncoder> encoder_;

    std::vector<Ptr<MixedImputation::CompleteData>> complete_data_;

    mutable RNG rng_;

    // ----------------------------------------------------------------------
    // Mutable workspace
    // ----------------------------------------------------------------------
    mutable SweptVarianceMatrix swept_sigma_;
    mutable bool swept_sigma_current_;

    // Set an observer that will flip swept_sigma_current_ to false when the
    // Sigma parameter changes.  This function is to be called whenever a new
    // numeric_data_model_ object is constructed.
    void set_numeric_data_model_observers();
    mutable Vector wsp_;
  };

  //===========================================================================
  // A multivariate model for imputing missing values for independent
  // observations (or "rows").  The model breaks a multivariate observation d =
  // (y, x), where y is numeric and x is categorical.  The distribution is
  // defined according to the decomposition p(y | x) p(x).
  //
  // The categorical part is modeled as a mixture of independent multinomial
  // distributions.  Let z be the mixture indicator, which is shared by all
  // variables in the observation.  Let Xj be the j'the variable in x.  Pr(Xj =
  // xj | z) = pi[z, j, xj].  This is a collection of distributions that differs
  // across mixture components z and data columns, j, and which sums to 1 across
  // xj.
  //
  // The numeric portion of the data is modeled as semicontinuous.  Let Yj be
  // the jth variable in y.  Yj is either one of k 'atoms' (particular numeric
  // values known in advance) or else a transformed version of Yj obeys a
  // multivariate linear regression with x.  The atomic part of the distribution
  // is handled in the same way as the categorical variables.  Each variable has
  // its own "atom list".  The conditional distribution of the atoms is indexed
  // by the same latent mixture indicator z as the categorical data.
  class MixedDataImputer : public MixedDataImputerBase
  {
   public:
    MixedDataImputer(int num_clusters,
                     const DataTable &data,
                     const std::vector<Vector> &atoms,
                     RNG &seeding_rng = GlobalRng::rng);
    MixedDataImputer(const MixedDataImputer &rhs);
    MixedDataImputer & operator=(const MixedDataImputer &rhs);
    MixedDataImputer(MixedDataImputer &&rhs) = default;
    MixedDataImputer & operator=(MixedDataImputer &&rhs) = default;

    MixedDataImputer * clone() const override;

    MixedImputation::RowModel * row_model(int cluster) override {
      return mixture_components_[cluster].get();
    }
    const MixedImputation::RowModel * row_model(int cluster) const override {
      return mixture_components_[cluster].get();
    }
    int number_of_mixture_components() const override {
      return mixture_components_.size();
    }

   private:
    std::vector<Ptr<MixedImputation::RowModel>> mixture_components_;

    int impute_cluster(Ptr<MixedImputation::CompleteData> &row,
                       RNG &rng,
                       bool update_complete_data_suf);
    int impute_cluster(Ptr<MixedImputation::CompleteData> &row, RNG &rng) const;

    void impute_numerics_given_atoms(Ptr<MixedImputation::CompleteData> &data,
                                     RNG &rng,
                                     bool update_complete_data_suf) override;

    void initialize_mixture(int num_clusters,
                            const std::vector<Vector> &atoms,
                            const std::vector<Ptr<CatKey>> &levels,
                            const std::vector<VariableType> &variable_type);
  };

}  // namespace BOOM

#endif  // BOOM_MIXED_DATA_IMPUTER_HPP_
