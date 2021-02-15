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
#include "Models/MultinomialModel.hpp"
#include "Models/WishartModel.hpp"
#include "Models/ChisqModel.hpp"

#include "LinAlg/SWEEP.hpp"

#include "stats/IQagent.hpp"
#include "distributions/rng.hpp"

#include "cpputil/ThreadTools.hpp"

namespace BOOM {

  namespace Imputer {

    //======================================================================
    // Multivariate regression data augmented with slots for error-corrected
    // data (possibly containing discrete atoms), and error corrected
    // transformed data.
    class CompleteData : public Data {
     public:
      // Args:
      //   observed: The observed data on which the augmented data should be
      //     based.
      explicit CompleteData(const Ptr<MvRegData> &observed);

      CompleteData * clone() const override;
      std::ostream &display(std::ostream &out) const override;

      const MvRegData *observed_data() const {return observed_data_.get();}

      // The observed data.
      const Vector &y_observed() const { return observed_data_->y(); }
      const Vector &x() const { return observed_data_->x(); }

      // The complete data, on the original scale.  This is the data that should
      // be recorded as an imputation.
      const Vector &y_true() const { return y_true_; }
      void set_y_true(int i, double y) { y_true_[i] = y; }
      void set_y_true(const Vector &y) { y_true_ = y; }

      // The complete data (no atoms) on the transformed scale.  This is the bit
      // that should be multivariate normal.
      const Vector &y_numeric() const { return y_numeric_; }
      void set_y_numeric(const Vector &numeric) { y_numeric_ = numeric; }
      void set_y_numeric(int i, double y) { y_numeric_[i] = y; }

     private:
      Ptr<MvRegData> observed_data_;
      Vector y_true_;
      Vector y_numeric_;
    };
  };

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
    // Args:
    //   atoms: A vector of numeric values expected to occur multiple times in
    //     the observed data.  Some values that appear as atoms in the observed
    //     data are actually errors.
    explicit ErrorCorrectionModel(const Vector &atoms);
    ErrorCorrectionModel(const ErrorCorrectionModel &rhs);
    ErrorCorrectionModel * clone() const override;

    // The log probability of the observed value.
    double logp(double observed) const;

    // Set a conjugate prior distribution on the marginal distribution of the
    // true categories.  The length of prior_counts must be atoms.size() + 1,
    // with the extra category at the end corresponding to the continuous cell.
    //
    // The prior can specify that an atom should have zero probability of being
    // the true value.  The signal for this is a negative count.
    void set_conjugate_prior_for_true_categories(const Vector &prior_counts);

    // Set a conjugate prior distribution on each conditional distribution of
    // the observed category given the true category.
    //
    // Args:
    //   prior_counts: A matrix of non-negative real numbers.  Each row
    //     corresponds to a "true" category (so the number of rows is
    //     atoms.size() + 1).  Each column corresponds to an "observed"
    //     category, so the number of columns must be atoms.size() + 2.
    //
    // The prior can specify that a given atom can never be observed when a
    // specific atom (either the same or another) is true.  The signal for this
    // is a negative prior count.
    void set_conjugate_prior_for_observation_categories(
        const Matrix &prior_counts);

    void sample_posterior() override;
    double logpri() const override;
    void clear_data() override;

    // Impute the atom responsible for the true value of the given observed
    // value.
    //
    // Args:
    //   observed_value:  The value that was actually observed.  This might be NaN.
    //   rng:  A random number generator.
    //   update: If true, then the imputed value will be used to update the
    //     sufficient statistics of the relevant sub-models.  This is what you
    //     want if the imputation is done as part of an MCMC algorithm to learn
    //     the model parameters.  If learning model parameters is not part of
    //     the current workflow then set this to false.
    //
    // Returns:
    //   An int representing the atoms responsible for the true and observed
    //   value, and the numeric value of the observation.
    int impute_atom(double observed_value, RNG &rng, bool update);

    // Return the value of y_true given the atom responsible for y_true.  This
    // is either equal to the atom value, or to the observed numeric value.  The
    // latter might be NaN, in which case NaN is returned.
    double true_value(int true_atom, double observed) const;

    // Return the value of y_numeric.  This is either NaN, or the observed value
    // (which might be NaN).
    double numeric_value(int true_atom, double observed) const;

    const Vector &atoms() const {return atoms_;}
    int number_of_atoms() const {return atoms_.size();}

    // The marginal probability that the true value of an observation will be
    // one of the atoms.  The atom_probs vector has one extra element (at the
    // end) indicating the probability that the observation is from the smoothly
    // varying component.
    const Vector &atom_probs() const {
      return marginal_of_true_data_->pi();
    }
    void set_atom_probs(const Vector &probs) {
      marginal_of_true_data_->set_pi(probs);
    }

    // The atom error probs are a matrix.  Rows correspond to entries in
    // atom_probs.  Columns have number_of_atoms + 2 entries, corresponding to
    // the smoothly varying numeric component, and "NA".
    Matrix atom_error_probs() const;
    void set_atom_error_probs(const Matrix &probs);

    // Returns the atom to which y corresponds.  The return value is between 0
    // and number_of_atoms + 1.  A value equal to number_of_atoms() indicates
    // the "smoothly varying" component.  A value equal to number_of_atoms() + 1
    // indicates NA.
    int category_map(double y) const;

    // Combine the sufficient statistics from 'other' into those from this
    // model.
    void combine_sufficient_statistics(const ErrorCorrectionModel &other);

    // Replace the current set of model parameters with those from 'other'.
    void copy_parameters(const ErrorCorrectionModel &other);

    //===========================================================================
    // Methods intended for testing purposes only.
    const MultinomialModel &atom_prob_model() const {
      return *marginal_of_true_data_;
    }
    const MultinomialModel &atom_error_prob_model(int atom) const {
      return *conditional_observed_given_true_[atom];
    }

   private:
    // A collection of point mass values from the observed data.  Some of these
    // are likely to represent errors or missing data codes.  Think 0 or 99999
    // as a place holder for missing data.
    //
    // In addition to the atoms specified here, there are two other implicit
    // categories.  "Continuous" is the first category after the atoms.  "NA" is
    // the final category.  NA is not a possible value for the truth.
    Vector atoms_;

    // The marginal distribution of the true data category.  The first
    // atoms_.size() elements are marginal probabilities for the atoms.  The
    // terminal element is the marginal probability for the
    Ptr<MultinomialModel> marginal_of_true_data_;

    // Note: some probabilities in these models will be 0.
    std::vector<Ptr<MultinomialModel>> conditional_observed_given_true_;

    mutable Matrix joint_distribution_;
    mutable Vector observed_log_probability_table_;
    mutable bool workspace_is_current_;
    Vector wsp_;
    void ensure_workspace_current() const;

    // To be called during construction.
    void set_observers();
  };

  //===========================================================================
  // This model is a mixture component for the MvRegCopulaDataImputer.  It
  // describes the conditional distribution of the categorical part of the
  // observed data given the observation-level mixture class indicator.
  class ConditionallyIndependentCategoryModel
      : public CompositeParamPolicy,
        public NullDataPolicy,
        public NullPriorPolicy {
   public:
    explicit ConditionallyIndependentCategoryModel(
        const std::vector<Vector> &atoms);

    ConditionallyIndependentCategoryModel(
        const ConditionallyIndependentCategoryModel &rhs);


    ConditionallyIndependentCategoryModel * clone() const override {
      return new ConditionallyIndependentCategoryModel(*this);
    }

    void clear_data() override;

    int ydim() const {return observed_data_models_.size();}

    // Args:
    //   data:  The data point to impute.
    //   rng:  The random numer generator to use for the imputation.
    //   update_complete_data_suf: If true then the complete data sufficient
    //     statistics for the component models will be updated with the imputed
    //     values.
    //
    // Effects:
    //   The atom responsible for each variable is imputed.  The results are
    //   stored in y_true and y_numeric.  If a variable is attributed to an
    //   atom, then y_true is set to the atom value and y_numeric is set to NaN.
    //   If a variable is attributed to the continuous atom then y_true is set
    //   to the observed value and y_numeric is set to that value.  y_numeric is
    //   not transformed to normality.
    void impute_atoms(Imputer::CompleteData &data, RNG &rng,
                      bool update_complete_data_suf);

    double logp(const Vector &observed) const;

    const ErrorCorrectionModel &model(int variable_index) const {
      return *observed_data_models_[variable_index];
    }

    Ptr<ErrorCorrectionModel> mutable_model(int variable_index) {
      return observed_data_models_[variable_index];
    }

    void sample_posterior() override {
      for (size_t i = 0; i < observed_data_models_.size(); ++i) {
        observed_data_models_[i]->sample_posterior();
      }
    }

    void combine_sufficient_statistics(
        const ConditionallyIndependentCategoryModel &other);
    void copy_parameters(const ConditionallyIndependentCategoryModel &other);

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

    // The copy constructor is especially important for this model because it is
    // used to generate workers for multi-threaded runs.
    MvRegCopulaDataImputer(const MvRegCopulaDataImputer &rhs);

    ~MvRegCopulaDataImputer();

    MvRegCopulaDataImputer *clone() const override;

    int xdim() const {return complete_data_model_->xdim();}
    int ydim() const {return complete_data_model_->ydim();}

    std::vector<Vector> atoms() const;

    // Data management needs overrides because the class maintains a separate
    // vector of complete data.
    void clear_data() override;
    void add_data(const Ptr<MvRegData> &data) override;
    void add_data(const Ptr<Data> &data) override {return add_data(DAT(data));}
    void add_data(MvRegData *data) override {return add_data(Ptr<MvRegData>(data));}
    void remove_data(const Ptr<Data> &data) override;

    // Clears the data from the clustering and complete data models.
    void clear_client_data();

    // Given an input, return a draw from the imputation distribution.  This
    // will contain any relevant atoms.
    //
    // Args:
    //   data:  An instance of CompleteData to be imputed.
    //   rng: The random number generator used to simulate the missing
    //     components of 'data'.
    //   update_complete_data_suf: If true then update the complete data
    //     sufficient statistics of the component models.
    //
    // Effects:
    //   The missing elements of 'data' are filled with random draws from the
    //   posterior distribution.
    void impute_row(Ptr<Imputer::CompleteData> &data,
                    RNG &rng,
                    bool update_complete_data_suf);

    int impute_cluster(Ptr<Imputer::CompleteData> &data, RNG &rng) const;
    int impute_cluster(Ptr<Imputer::CompleteData> &data, RNG &rng,
                       bool update_complete_data_suf);

    // Posterior samplers need to be assigned to the components of the cluster
    // model.
    Ptr<MultinomialModel> cluster_mixing_distribution() {
      return cluster_mixing_distribution_;
    }

    Ptr<ConditionallyIndependentCategoryModel> cluster_mixture_component(
        int component) {
      return cluster_mixture_components_[component];
    }

    // Need access to the regression model so we can set a prior.
    Ptr<MultivariateRegressionModel> regression() {return complete_data_model_;}

    double logpri() const override;
    void sample_posterior() override;

    int nclusters() const {
      return cluster_mixing_distribution_->dim();
    }

    const Vector &atom_probs(int cluster, int variable_index) const;
    void set_atom_probs(int cluster, int variable_index, const Vector &probs);

    Matrix atom_error_probs(int cluster, int variable_index) const;
    void set_atom_error_probs(int cluster, int variable_index,
                              const Matrix &probs);

    // Set default priors for everything.
    void set_default_priors();

    void set_atom_prior(const Vector &prior_counts, int variable_index);
    void set_atom_error_prior(const Matrix &prior_counts, int variable_index);
    void set_default_prior_for_mixing_weights();
    void set_default_regression_prior();

    // Return the imputed values of the numeric variables.
    Matrix imputed_data() const;

    Matrix impute_data_set(const std::vector<Ptr<MvRegData>> &data);

    //--------------------------------------------------------------------------
    // Code needed to save/restore models.
    const std::vector<IQagent> empirical_distributions() const {
      return empirical_distributions_;
    }
    void set_empirical_distributions(
        const std::vector<IQagent> &empirical_distributions) {
      empirical_distributions_ = empirical_distributions;
    }

    std::vector<IqAgentState> empirical_distribution_state() const;
    void restore_empirical_distributions(
        const std::vector<IqAgentState> &state);

    //--------------------------------------------------------------------------
    void setup_worker_pool(int nworkers);
    void shut_down_worker_pool();

    const ConditionallyIndependentCategoryModel &
    cluster_mixture_component(int s) const {
      return *cluster_mixture_components_[s];
    }

    int id() const {return worker_id_;}

   private:
    // Describes the component to which each observation belongs.  This model
    // controls the sharing of information across variables.
    //
    // The "emission distribution" for this model is a
    // ConditionallyIndependentCategoryModel.
    Ptr<MultinomialModel> cluster_mixing_distribution_;

    // The mixture components in cluster_model_;
    std::vector<Ptr<ConditionallyIndependentCategoryModel>> cluster_mixture_components_;

    // The complete data model describes the relationship among the continuous
    // variables.  In the event that an observation is driven by an atom, the
    // unobserved continuous part is to be imputed and used to fit the model.
    Ptr<MultivariateRegressionModel> complete_data_model_;

    std::vector<IQagent> empirical_distributions_;
    void initialize_empirical_distributions(int ydim);

    std::vector<Ptr<Imputer::CompleteData>> complete_data_;

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

    // ======================================================================
    // Threading section
    // ======================================================================

    // If the object is a worker then the workers_ vector is empty and the
    // thread pool has no threads.
    std::vector<Ptr<MvRegCopulaDataImputer>> workers_;
    ThreadWorkerPool thread_pool_;
    int worker_id_;

    // These methods are here to implemente multi-threading.
    void impute_latent_data_multithreaded();

    void distribute_data_to_workers();
    void ensure_data_distribution();
    void broadcast_parameters();
    void reduce_sufficient_statistics();
    void impute_all_rows();
  };

}  // namespace BOOM

#endif  // BOOM_MVREG_COPULA_DATA_IMPUTER_HPP_
