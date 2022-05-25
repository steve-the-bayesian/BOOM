#ifndef BOOM_STATE_SPACE_SHARED_LOCAL_LEVEL_HPP_
#define BOOM_STATE_SPACE_SHARED_LOCAL_LEVEL_HPP_

/*
  Copyright (C) 2005-2021 Steven L. Scott

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

#include "Models/ZeroMeanGaussianModel.hpp"
#include "Models/Glm/MultivariateRegression.hpp"
#include "Models/Glm/WeightedRegressionModel.hpp"
#include "Models/StateSpace/Multivariate/MultivariateStateSpaceModelBase.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

#include "Models/StateSpace/Multivariate/StateModels/SharedStateModel.hpp"

namespace BOOM {

  //===========================================================================
  // A local level model for describing multivariate outcomes.  The latent state
  // consists of K independent random walks which are the 'factors'.  The series
  // are linked to the factors accorrding to
  //
  //     E( y[t] | alpha[t] ) = Z * alpha[t].
  //
  // Conditional on the the state and the observed data are almost a
  // multivariate regression model.  However some constraints are needed in
  // order to identify the model.  These are often expressed as the coefficient
  // matrix needing to be zero on one side of the diagonal, with a unit
  // diagonal.  The posterior sampler for this model will handle the
  // constraints, as different constraints might be relevant for different
  // modeling strategies.
  //
  // This model is a base class because different types of host models will make
  // different assumptions about the form of the residual variance matrix in the
  // regression of the observed data on latent state.
  class SharedLocalLevelStateModelBase
      : public SharedStateModel,
        public CompositeParamPolicy,
        public NullDataPolicy,
        public PriorPolicy
  {
   public:
    // Args:
    //   number_of_factors: The number of independent random walks to use in
    //     this state model.  The number of factors is the state dimension.
    //   nseries:  The number of observed time series being modeled.
    SharedLocalLevelStateModelBase(int number_of_factors,
                                   int nseries);
    SharedLocalLevelStateModelBase(const SharedLocalLevelStateModelBase &rhs);
    SharedLocalLevelStateModelBase(SharedLocalLevelStateModelBase &&rhs);
    SharedLocalLevelStateModelBase &operator=(const SharedLocalLevelStateModelBase &rhs);
    SharedLocalLevelStateModelBase &operator=(SharedLocalLevelStateModelBase &&rhs);

    SharedLocalLevelStateModelBase * clone() const override = 0;

    //----------------------------------------------------------------------
    // Sizes of things.  The state dimension and the state_error_dimension are
    // both just the number of factors.
    uint state_dimension() const override {return innovation_models_.size();}
    uint state_error_dimension() const override {return state_dimension();}
    virtual int nseries() const = 0;

    // Syntactic sugar.
    int number_of_factors() const {return state_dimension();}

    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;
    void simulate_initial_state(RNG &rng, VectorView eta) const override;

    virtual MultivariateStateSpaceModelBase *host() = 0;
    virtual const MultivariateStateSpaceModelBase *host() const = 0;

    //--------------------------------------------------------------------------
    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override {
      return state_transition_matrix_;
    }

    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override {
      return state_variance_matrix_;
    }

    // The state error expander matrix is an identity matrix of the same
    // dimension as the state_transition_matrix, so we just return that matrix.
    Ptr<SparseMatrixBlock> state_error_expander(int t) const override {
      return state_transition_matrix_;
    }

    // Because the error expander is the identity, the state variance matrix and
    // the state error variance are the same thing.
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override {
      return state_variance_matrix_;
    }

    //--------------------------------------------------------------------------
    // Initial state mean and variance.
    Vector initial_state_mean() const override;
    void set_initial_state_mean(const Vector &m);
    SpdMatrix initial_state_variance() const override;
    void set_initial_state_variance(const SpdMatrix &v);

    void observe_state(const ConstVectorView &then,
                       const ConstVectorView &now,
                       int time_now) override;

    //--------------------------------------------------------------------------
    // Tools for working with the EM algorithm and numerical optimization.
    // These are not currently implemented.
    void update_complete_data_sufficient_statistics(
        int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    void increment_expected_gradient(
        VectorView gradient, int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    //----------------------------------------------------------------------
    // Methods intended for use with the posterior samplers managing this model.

    Ptr<ZeroMeanGaussianModel> innovation_model(int i) {
      return innovation_models_[i];
    }

    // Convert the regression coefficients linking the state to the observation
    // equation so that they are lower triangular.  That is, in the equation y =
    // Z * alpha + error, Z is lower triangular with 1's on the diagonal.  Z is
    // the transpose of the coefficients in coefficient_model_.
    void impose_identifiability_constraint() override;

   protected:
    void clear_state_transition_data();
    virtual void initialize_model_matrices();

   private:
    // The innovation models describe the movement of the individual factors
    // from one time period to the next.  The model consists of
    // number_of_factors latent series.  innovation_models_[i] describes the
    // innovation errors for series i.
    std::vector<Ptr<ZeroMeanGaussianModel>> innovation_models_;

    // The state transition matrix is a number_of_factors * number_of_factors
    // identity matrix.
    Ptr<IdentityMatrix> state_transition_matrix_;

    // The state variance matrix is a view into variance parameters of the
    // innovation models.
    Ptr<DiagonalMatrixParamView> state_variance_matrix_;

    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;
    Matrix initial_state_variance_cholesky_;

    // Args:
    //   residual_y: The observed data, after having
    virtual void record_observed_data_given_state(const Vector &residual_y,
                                                  const ConstVectorView &now,
                                                  int time_now) = 0;
  };

  //===========================================================================
  // A SharedLocalLevelStateModel where the host is a
  // ConditionallyIndependentMultivariateStateSpaceModelBase.  The observation
  // model here uses a diagonal covariance matrix for the observation error.
  class ConditionallyIndependentSharedLocalLevelStateModel
      : public SharedLocalLevelStateModelBase
  {
   public:

    ConditionallyIndependentSharedLocalLevelStateModel(
        ConditionallyIndependentMultivariateStateSpaceModelBase *host,
        int number_of_factors,
        int nseries);

    ConditionallyIndependentSharedLocalLevelStateModel(
        const ConditionallyIndependentSharedLocalLevelStateModel &rhs);
    ConditionallyIndependentSharedLocalLevelStateModel(
        ConditionallyIndependentSharedLocalLevelStateModel &&rhs);
    ConditionallyIndependentSharedLocalLevelStateModel & operator=(
        const ConditionallyIndependentSharedLocalLevelStateModel &rhs);
    ConditionallyIndependentSharedLocalLevelStateModel & operator=(
        ConditionallyIndependentSharedLocalLevelStateModel &&rhs);
    ConditionallyIndependentSharedLocalLevelStateModel * clone() const override;

    void clear_data() override;

    Ptr<SparseMatrixBlock> observation_coefficients(
        int t, const Selector &observed) const override;
    int nseries() const override;

    ConditionallyIndependentMultivariateStateSpaceModelBase *
    host() override {
      return host_;
    }

    const ConditionallyIndependentMultivariateStateSpaceModelBase *
    host() const override {
      return host_;
    }

    Ptr<GlmCoefs> raw_observation_coefficients(int i) {
      return raw_observation_coefficients_[i];
    }

    const Ptr<GlmCoefs> raw_observation_coefficients(int i) const {
      return raw_observation_coefficients_[i];
    }

    const Ptr<WeightedRegSuf>& suf(int i) const {
      return sufficient_statistics_[i];
    }

   private:
    void record_observed_data_given_state(const Vector &residual_y,
                                          const ConstVectorView &now,
                                          int time_now) override;

    void ensure_observation_coefficients_current() const;
    void set_observation_coefficients_observer();

    ConditionallyIndependentMultivariateStateSpaceModelBase *host_;

    // raw_observation_coefficients_[i] contains the observation coefficients
    // for time series i on the shared state factors.
    std::vector<Ptr<GlmCoefs>> raw_observation_coefficients_;

    // Sufficient statistics for the model.  These get recorded when we
    // record_observed_data_given_state, and set to zero when we call
    // clear_data.
    //
    // Each element contains X'X and X'y for each series, where X is the time
    // series of factors.  If every observation is present at every time point
    // then all the X'X entries will be the same, but if some series are
    // partially observed then they will have different X'X entries, so we need
    // a series-by-series
    std::vector<Ptr<WeightedRegSuf>> sufficient_statistics_;

    mutable Ptr<DenseMatrix> observation_coefficients_;
    mutable bool observation_coefficients_current_;
  };

  //===========================================================================
  // A shared local level state model where the observation model uses a full
  // general covariance matrix.
  class GeneralSharedLocalLevelStateModel
      : public SharedLocalLevelStateModelBase
  {
   public:
    GeneralSharedLocalLevelStateModel(MultivariateStateSpaceModelBase *host,
                                      int number_of_factors, int nseries);
    GeneralSharedLocalLevelStateModel(
        const GeneralSharedLocalLevelStateModel &rhs);
    GeneralSharedLocalLevelStateModel & operator=(
        const GeneralSharedLocalLevelStateModel &rhs);
    GeneralSharedLocalLevelStateModel(
        GeneralSharedLocalLevelStateModel &&rhs);
    GeneralSharedLocalLevelStateModel & operator=(
        GeneralSharedLocalLevelStateModel &&rhs);
    GeneralSharedLocalLevelStateModel * clone() const override;

    MultivariateStateSpaceModelBase *host() override {return host_;}
    const MultivariateStateSpaceModelBase *host() const override {return host_;}

    MultivariateRegressionModel * coefficient_model() {
      return coefficient_model_.get();
    }

    Ptr<SparseMatrixBlock> observation_coefficients(
        int t, const Selector &observed) const override;
    int nseries() const override {return host_->nseries();}

   private:
    // Helper functions to be called in the constructor.
    void initialize_observation_coefficient_matrix();
    void set_observation_coefficients_observer();
    void set_param_policy();
    void sync_observation_coefficients();

    void record_observed_data_given_state(const Vector &residual_y,
                                          const ConstVectorView &now,
                                          int time_now) override;

    // The host is the model object in which *this is a state component.  The
    // host is needed for this model to properly implement observe_state,
    // because the coefficient models needs to subtract away the contributions
    // from other state models.
    MultivariateStateSpaceModelBase *host_;

    // The coefficient model describes the contribution of this state model to
    // the observation equation.  The multivariate regression model is organized
    // as (xdim, ydim).  The 'X' in our case is the state, where we want y = Z *
    // state.  In the regression model things are set up so Y = X * beta.  So
    // our 'Z' is the transpose of the coefficient matrix from the regression.
    Ptr<MultivariateRegressionModel> coefficient_model_;
    mutable Ptr<DenseMatrix> observation_coefficients_;
    Ptr<SparseMatrixBlock> empty_;
  };

}  // namespace BOOM

#endif  //  BOOM_STATE_SPACE_SHARED_LOCAL_LEVEL_HPP_
