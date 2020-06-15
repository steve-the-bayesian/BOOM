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

    class CompleteData : public Data {
     public:
      CompleteData(const Ptr<MixedMultivariateData> &data);
      CompleteData(const CompleteData &rhs);
      CompleteData & operator=(const CompleteData &rhs);

      CompleteData(CompleteData &&rhs) = default;
      CompleteData & operator=(CompleteData &&rhs) = default;

      CompleteData *clone() const override;
      std::ostream &display(std::ostream &out) const override;

      const MixedMultivariateData &observed_data() const {
        return *observed_data_;
      }

     private:
      Ptr<MixedMultivariateData> observed_data_;

      // y_true_ holds the vector of imputed Y values for the numeric data.
      Vector y_true_;

      // y_numeric_ holds the output of the transformed regression model, on the
      // copula transformed scale.
      Vector y_numeric_;

      // There are as many entries in true_categories_ as in data_.  For numeric
      // variables, the true_category is the category indicating which atom
      // holds.  For categorical variables, the entry indicates which category
      // holds.
      std::vector<int> true_categories_;
    };

    //=========================================================================
    class ErrorCorrectionModelBase : virtual public Model {
     public:
      ErrorCorrectionModelBase(int index)
          : index_(index)
      {}

      ErrorCorrectionModelBase *clone() const override = 0;

      // Pick the relevant entry out of 'data' and return its log density under
      // the model.
      virtual double logp(const MixedMultivariateData &data) const = 0;

      int index() const {return index_;}

     private:
      int index_;
    };

    //-------------------------------------------------------------------------
    // A model for a possibly-erroneous single semicontinuous numeric entry in a
    // mixed data observation.
    //
    // The model tracks a known set of atoms.
    class NumericErrorCorrectionModel
        : public ErrorCorrectionModelBase,
          public CompositeParamPolicy,
          public NullDataPolicy,
          public NullPriorPolicy
    {
     public:
      NumericErrorCorrectionModel(int index, const Vector &atoms);

      NumericErrorCorrectionModel(const NumericErrorCorrectionModel &rhs);
      NumericErrorCorrectionModel &operator=(
          const NumericErrorCorrectionModel &rhs);
      NumericErrorCorrectionModel(NumericErrorCorrectionModel &&rhs) = default;
      NumericErrorCorrectionModel &operator=(
          NumericErrorCorrectionModel &&rhs) = default;

      NumericErrorCorrectionModel *clone() const override;

      double logp(const MixedMultivariateData &data) const override;
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
      // The key describes all possible observed values.  The atoms are a subset
      // of the values in the key that might represent errors.
      CategoricalErrorCorrectionModel(int index,
                                      const Ptr<CatKey> &key,
                                      const Ptr<CatKey> &atoms);

      CategoricalErrorCorrectionModel(
          const CategoricalErrorCorrectionModel &rhs);
      CategoricalErrorCorrectionModel &operator=(
          const CategoricalErrorCorrectionModel &rhs);
      CategoricalErrorCorrectionModel(
          CategoricalErrorCorrectionModel &&rhs) = default;
      CategoricalErrorCorrectionModel &operator=(
          CategoricalErrorCorrectionModel &&rhs) = default;

      CategoricalErrorCorrectionModel *clone() const override;

      double logp(const std::string &label) const;
      double logp(const MixedMultivariateData &data) const override;

      void sample_posterior() override;
      double logpri() const override;
      void clear_data() override;
      int impute_atom(double observed_value, RNG &rng, bool update);

      int number_of_atoms() const {return atoms_->max_levels();}

     private:
      Ptr<CatKey> levels_;
      Ptr<CatKey> atoms_;
      Ptr<MultinomialModel> truth_model_;

      // There are number_of_atoms() + 1 models in obs_models_.  The final model
      // is the "not_an_atom" category.  Each model describes
      std::vector<Ptr<MultinomialModel>> obs_models_;

      mutable Vector wsp_;
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
      RowModel(const std::vector<InitInfo> &init);
      RowModel(const RowModel &rhs);

      RowModel *clone() const override;
      double logp(const MixedMultivariateData &data) const;
      void clear_data() override;

      int ydim() const {return scalar_models_.size();}

      // Fill in the "true category" data member of 'row'.  The draw is made
      // conditional on the imputed y_numeric values, and on the mixture mixture
      // indicator.
      void impute_categorical(Ptr<MixedImputation::CompleteData> &row,
                              RNG &rng,
                              bool update_complete_data_suf);

     private:
      std::vector<Ptr<ErrorCorrectionModelBase>> scalar_models_;
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

   private:
    int impute_cluster(Ptr<MixedImputation::CompleteData> &row,
                       RNG &rng,
                       bool update_complete_data_suf);
    int impute_cluster(Ptr<MixedImputation::CompleteData> &row, RNG &rng) const;

    void impute_numeric_given_categorical(
        Ptr<MixedImputation::CompleteData> &row, RNG &rng,
        bool update_complete_data_suf);

    void initialize_empirical_distributions(int ydim);

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
