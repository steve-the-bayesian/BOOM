#ifndef BOOM_MVREG_COPULA_DATA_IMPUTER_HPP_
#define BOOM_MVREG_COPULA_DATA_IMPUTER_HPP_
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

#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/NullPriorPolicy.hpp"
#include "Models/Glm/MultivariateRegression.hpp"
#include "Models/FiniteMixtureModel.hpp"
#include "Models/MultinomialModel.hpp"
#include "Models/WishartModel.hpp"
#include "Models/ChisqModel.hpp"

#include "stats/IQagent.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  //===========================================================================
  // Describes the joint distribution of the true and observed data for a single
  // variable.
  //
  // The idea is that each observation has a true value and an observed value.
  // The true value is one of several known atoms, or else it is a continuous
  // value.  We don't model the specific continuous value, just the fact that it
  // is continuous.
  //
  // The observed value falls into any of the categories of the true value, plus
  // "missing".  The model has a marginal component for the true value, and a
  // conditional component for the observed value given the truth.
  class ErrorCorrectionModel
      : public CompositeParamPolicy,
        public NullDataPolicy,
        public NullPriorPolicy
  {
   public:
    ErrorCorrectionModel(const Vector &atoms);

    ErrorCorrectionModel * clone() const override;

    // The log probability of the observed value.
    double logp(double observed) const;

    int number_of_observations() const {
      return marginal_of_true_data_->number_of_observations();
    }

    // Set a conjugate prior distribution on the marginal distribution of the
    // true categories.  The length of prior_counts must be atoms.size() + 1,
    // with the extra category at the end corresponding to the continuous cell.
    void set_conjugate_prior_for_true_categories(const Vector &prior_counts);

    // Set a conjugate prior distribution on each conditional distribution of
    // the observed category given the true category.
    //
    // Args:
    //   prior_counts: A matrix of non-negative real numbers.  Each row
    //     corresponds to a "true" category (so the number of rows is
    //     atoms.size() + 1).  Each column corresponds to an "observed"
    //     category, so the number of columns must be atoms.size() + 2.
    void set_conjugate_prior_for_observation_categories(
        const Matrix &prior_counts);

    void sample_posterior() override;
    double logpri() const override;

   private:
    // A collection of point mass values from the observed data.  Some of these
    // are likely to represent errors or missing data codes.  Think 0 or 99999
    // as a place holder for missing data.
    //
    // In addition to the atoms specified here, there are two other implicit
    // categories.  "Continuous" is the first category after the atoms.  "NA" is
    // the final category.  NA is not a possible value for the truth.
    Vector atoms_;

    // Returns the atom to which y corresponds.
    int category_map(double y) const;

    // The marginal distribution of the true data category.  The first
    // atoms_.size() elements are marginal probabilities for the atoms.  The
    // terminal element is the marginal probability for the
    Ptr<MultinomialModel> marginal_of_true_data_;

    // Note: some probabilities in these models will be 0.
    std::vector<Ptr<MultinomialModel>> conditional_observed_given_true_;

    mutable Vector observed_log_probability_table_;
    mutable bool observed_log_probability_table_current_;
    void ensure_observed_log_probability_table_current() const;
  };

  //===========================================================================
  // This model is a mixture component for the MvRegCopulaDataImputer.  It
  // describes the conditional distribution of the categorical part of the
  // observed data given the observation-level mixture class indicator.
  class ConditionallyIndependentCategoryModel
      : public CompositeParamPolicy,
        public IID_DataPolicy<VectorData>,
        public PriorPolicy,
        public MixtureComponent {
   public:
    ConditionallyIndependentCategoryModel(const std::vector<Vector> &atoms);

    ConditionallyIndependentCategoryModel * clone() const override {
      return new ConditionallyIndependentCategoryModel(*this);
    }

    double pdf(const Data *dp, bool logscale) const override;
    int number_of_observations() const override {
      return observed_data_models_[0]->number_of_observations();
    }

   private:
    // A model for the observed data (just the categorical parts of the numeric
    // parts).
    std::vector<Ptr<ErrorCorrectionModel>> observed_data_models_;
  };

  //===========================================================================
  // Given a collection of continuous variables Y, which contain missing values,
  // and a collection of fully observed predictors X (which might be empty),
  // impute the missing values in Y from their posterior distribution using the
  // following algorithm. Let Z ~ Mvn(X * Beta, Sigma), where Z is the Gaussian
  // copula transformed, fully observed, version of Y.
  //
  // 1) Simulate Beta, Sigma ~ p(Beta, Sigma | Z).
  // 2) Simulate Zmis | Zobs, Beta, Sigma.
  // 3) Transform Z -> Y, and recompute the copula transform.
  // 4) Transform back to Z, then repeat.
  class MvRegCopulaDataImputer
      : public IID_DataPolicy<MvRegData>,
        public CompositeParamPolicy,
        public NullPriorPolicy {
   public:
    // Args:
    //   num_clusters: The number of clusters to use in the finite mixture model
    //     for the categorical part of the analysis.
    //   atoms: The set of atoms to recieve special treatment for each variable.
    //     The length of this vector is used to determine the number of
    //     variables in the model.
    //   xdim:  The dimension of the fully observed predictor variable.
    MvRegCopulaDataImputer(int num_clusters,
                           const std::vector<Vector> &atoms,
                           int xdim,
                           RNG &seeding_rng = GlobalRng::rng);

    // Given an input, return a draw from the imputation distribution.  This
    // will contain any relevant atoms.
    //
    // Args:
    //   input: The actually observed values in the data, including errors and
    //     NaN's.
    //   predictors:  The vector of predictor variables.
    //   rng: Random number generator used to make the prediction.
    Vector impute_row(const Vector &input,
                      const ConstVectorView &predictors,
                      RNG &rng) const;

    Vector impute_continuous_values(
        const Vector &y, const ConstVectorView &x, RNG &rng) const;

    // Posterior samplers need to be assigned to the components of the cluster
    // model.
    Ptr<FiniteMixtureModel> cluster_model() {return cluster_model_;}

    // Need access to the regression model so we can set a prior.
    Ptr<MultivariateRegressionModel> regression() {return complete_data_model_;}

    double logpri() const override;
    void sample_posterior() override;

   private:
    // Describes the component to which each observation belongs.  This model
    // controls the sharing of information across variables.
    //
    // The "emission distribution" for this model is a
    // ConditionallyIndependentCategoryModel.
    Ptr<FiniteMixtureModel> cluster_model_;

    // The mixture components in cluster_model_;
    std::vector<Ptr<ConditionallyIndependentCategoryModel>> mixture_components_;

    // The complete data model describes the relationship among the continuous
    // variables.  In the event that an observation is driven by an atom, the
    // unobserved continuous part is to be imputed and used to fit the model.
    Ptr<MultivariateRegressionModel> complete_data_model_;

    std::vector<IQagent> empirical_distributions_;

    RNG rng_;
  };

}  // namespace BOOM

#endif  // BOOM_MVREG_COPULA_DATA_IMPUTER_HPP_
