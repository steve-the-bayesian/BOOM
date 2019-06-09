#ifndef BOOM_STATE_SPACE_LOCAL_LEVEL_STATE_MODEL_HPP
#define BOOM_STATE_SPACE_LOCAL_LEVEL_STATE_MODEL_HPP
// Copyright 2019 Steven L. Scott.
// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2008 Steven L. Scott

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

#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"

#include "Models/Glm/MultivariateRegression.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/StateSpace/DynamicInterceptRegression.hpp"
#include "Models/StateSpace/MultivariateStateSpaceModelBase.hpp"

namespace BOOM {

  // The local level state model assumes
  //
  //      y[t] = mu[t] + epsilon[t]
  //   mu[t+1] = mu[t] + eta[t].
  //
  // That is, it is a random walk observed in noise.  It is the simplest useful
  // state model.
  class LocalLevelStateModel : virtual public StateModel, public ZeroMeanGaussianModel {
   public:
    explicit LocalLevelStateModel(double sigma = 1);
    LocalLevelStateModel(const LocalLevelStateModel &rhs);
    LocalLevelStateModel *clone() const override;
    void observe_state(const ConstVectorView &then, const ConstVectorView &now,
                       int time_now) override;

    uint state_dimension() const override;
    uint state_error_dimension() const override { return 1; }
    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;
    void simulate_initial_state(RNG &rng, VectorView eta) const override;

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_error_expander(int t) const override;
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override;

    SparseVector observation_matrix(int t) const override;

    Vector initial_state_mean() const override;
    SpdMatrix initial_state_variance() const override;

    void set_initial_state_mean(double m);
    void set_initial_state_mean(const Vector &m);
    void set_initial_state_variance(const SpdMatrix &v);
    void set_initial_state_variance(double v);

    void update_complete_data_sufficient_statistics(
        int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    void increment_expected_gradient(
        VectorView gradient, int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

   private:
    Ptr<IdentityMatrix> state_transition_matrix_;
    Ptr<ConstantMatrixParamView> state_variance_matrix_;
    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;
  };

  //===========================================================================
  class DynamicInterceptLocalLevelStateModel
      : public LocalLevelStateModel,
        public DynamicInterceptStateModel {
   public:
    explicit DynamicInterceptLocalLevelStateModel(double sigma = 1.0)
        : LocalLevelStateModel(sigma)
    {}

    DynamicInterceptLocalLevelStateModel *clone() const override {
      return new DynamicInterceptLocalLevelStateModel(*this);
    }
    
    bool is_pure_function_of_time() const override {return true;}

    Ptr<SparseMatrixBlock> observation_coefficients(
        int t,
        const StateSpace::TimeSeriesRegressionData &data_point) const override {
      // In single threaded code we could optimize here by creating a single
      // IdenticalRowsMatrix and changing the number of rows each time.
      //
      // In multi-threaded code that would create a race condition.
      return new IdenticalRowsMatrix(observation_matrix(t),
                                     data_point.sample_size());
    }
  };

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
  class SharedLocalLevelStateModel
      : virtual public SharedStateModel,
        public CompositeParamPolicy,
        public NullDataPolicy,
        public PriorPolicy
  {
   public:
    // Args:
    //   number_of_factors: The number of independent random walks to use in
    //     this state model.  The number of factors is the state dimension.
    //   ydim:  The dimension of the outcome variable at time t.
    //   host:  The model in which this object is a component of state.
    //   nseries:  The number of observed time series being modeled.
    SharedLocalLevelStateModel(int number_of_factors,
                               MultivariateStateSpaceModelBase *host,
                               int nseries);
    SharedLocalLevelStateModel(const SharedLocalLevelStateModel &rhs);
    SharedLocalLevelStateModel(SharedLocalLevelStateModel &&rhs);
    SharedLocalLevelStateModel &operator=(const SharedLocalLevelStateModel &rhs);
    SharedLocalLevelStateModel &operator=(SharedLocalLevelStateModel &&rhs);
    SharedLocalLevelStateModel *clone() const override;
    
    void clear_data() override;
    void observe_state(const ConstVectorView &then, const ConstVectorView &now,
                       int time_now) override;

    //----------------------------------------------------------------------
    // Sizes of things.
    uint state_dimension() const override {return innovation_models_.size();}
    uint state_error_dimension() const override {return state_dimension();}
    int nseries() const {return coefficient_model_->ydim();}
    
    // Syntactic sugar.
    int number_of_factors() const {return state_dimension();}
    
    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;
    void simulate_initial_state(RNG &rng, VectorView eta) const override;

    //--------------------------------------------------------------------------
    // Model matrices.
    Ptr<SparseMatrixBlock> observation_coefficients(
        int t, const Selector &observed) const override;

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
    Ptr<MultivariateRegressionModel> coefficient_model() {
      return coefficient_model_;
    }

    // A setter helps prevent the user from forgetting to transpose the
    // observation coefficients.
    //
    // Args:
    //   Z: The matrix of observation coefficients.  nseries rows by
    //     number_of_factors.  columns.
    void set_observation_coefficients(const Matrix &Z) {
      coefficient_model_->set_Beta(Z.transpose());
    }
    
    // Copy the observation coefficients from the regression model to the state
    // matrix.  Note that the state coefficient matrix is the transpose of the
    // regression coefficient matrix.
    void sync_observation_coefficients();
    
    Ptr<ZeroMeanGaussianModel> innovation_model(int i) {
      return innovation_models_[i];
    }

    // Convert the regression coefficients linking the state to the observation
    // equation so that they are lower triangular.  That is, in the equation y =
    // Z * alpha + error, Z is lower triangular with 1's on the diagonal.  Z is
    // the transpose of the coefficients in coefficient_model_.
    void impose_identifiability_constraint() override;
    
   private:
    // The model consists of number_of_factors latent series.
    // innovation_models_[i] describes the innovation errors for series i.

    // The host is the model object in which *this is a state component.  The
    // host is needed for this model to properly implement observe_state,
    // because the coefficient models needs to subtract away the contributions
    // from other state models.
    MultivariateStateSpaceModelBase *host_;

    // The innovation models describe the movement of the individual factors
    // from one time period to the next.
    std::vector<Ptr<ZeroMeanGaussianModel>> innovation_models_;

    // The coefficient model describes the contribution of this state model to
    // the observation equation.  The multivariate regression model is organized
    // as (xdim, ydim).  The 'X' in our case is the state, where we want y = Z *
    // state, so we need the transpose of the coefficient matrix from the
    // regression.
    Ptr<MultivariateRegressionModel> coefficient_model_;
    mutable Ptr<DenseMatrix> observation_coefficients_;
    Ptr<SparseMatrixBlock> empty_;

    // An observer to be placed on the observation coefficients to indicate that
    // they have changed.
    void set_observation_coefficients_observer();
    
    // The state transition matrix is a number_of_factors * number_of_factors
    // identity matrix.
    Ptr<IdentityMatrix> state_transition_matrix_;

    // The state variance matrix is a view into variance parameters of the
    // innovation models.
    Ptr<DiagonalMatrixParamView> state_variance_matrix_;

    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;
    Matrix initial_state_variance_cholesky_;

    // Helper functions to be called in the constructor.
    void set_param_policy();
    void initialize_model_matrices();
  };
}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_LOCAL_LEVEL_STATE_MODEL_HPP
