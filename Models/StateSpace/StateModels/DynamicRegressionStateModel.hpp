/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#ifndef BOOM_STATE_SPACE_DYNAMIC_REGRESSION_STATE_MODEL_HPP_
#define BOOM_STATE_SPACE_DYNAMIC_REGRESSION_STATE_MODEL_HPP_
#include <Models/StateSpace/StateModels/StateModel.hpp>
#include <Models/StateSpace/Filters/SparseMatrix.hpp>
#include <Models/StateSpace/Filters/SparseVector.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/Policies/NullDataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/ZeroMeanGaussianModel.hpp>

#include <LinAlg/Vector.hpp>
#include <LinAlg/VectorView.hpp>
#include <LinAlg/SpdMatrix.hpp>

namespace BOOM{
  // A dynamic regression state is an element of state parameterized
  // by a set of time varying regression coefficients.
  //
  // The observation matrix at time t is the vector of predictors x[t].
  // The transition matrix is the identity.
  // The RQR matrix is a diagonal matrix of sigma^2 (with a different
  // variance per coefficient).
  //
  // The model is
  //
  //      beta[i, t] ~ N(beta[i, t-1], sigsq[i] / variance_x[i])
  //  1.0 / sigsq[i] ~ Gamma(a / b)
  //
  // That is, each coefficient has its own variance term, which is
  // scaled by the variance of the i'th column of X.  The parameters
  // of the hyperprior are interpretable as: sqrt(b/a) typical amount
  // that a coefficient might change in a single time period, and 'a'
  // is the 'sample size' or 'shrinkage parameter' measuring the
  // degree of similarity in sigma[i] among the arms.
  //
  // In most cases we hope b/a is small, so that sigma[i]'s will be
  // small and the series will be forecastable.  We also hope that 'a'
  // is large because it means that the sigma[i]'s will be similar to
  // one another.
  class DynamicRegressionStateModel
      : public StateModel,
        public CompositeParamPolicy,
        public NullDataPolicy,
        public PriorPolicy
  {
   public:
    // Each row of X is a predictor vector for an observation.
    // I.e. rows of X correspond to time points.
    DynamicRegressionStateModel(const Matrix &X);
    DynamicRegressionStateModel(const DynamicRegressionStateModel &rhs);
    DynamicRegressionStateModel * clone()const override;

    void set_xnames(const std::vector<string> &xnames);
    const std::vector<string> & xnames() const;

    void clear_data() override;
    void observe_state(const ConstVectorView then,
                       const ConstVectorView now,
                       int time_now) override;
    void observe_initial_state(const ConstVectorView &state) override;
    uint state_dimension() const override;
    uint state_error_dimension() const override {
      return state_dimension();
    }

    void update_complete_data_sufficient_statistics(
        int t,
        const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_error_expander(int t) const override;
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override;

    // The observation matrix is row t of the desing matrix.
    SparseVector observation_matrix(int t) const override;

    // The initial state is the value of the regression coefficients
    // at time 0.  Zero with a big variance is a good guess.
    Vector initial_state_mean() const override;
    void set_initial_state_mean(const Vector &mu);

    SpdMatrix initial_state_variance() const override;
    void set_initial_state_variance(const SpdMatrix &sigma);

    const GaussianSuf * suf(int i) const;
    double sigsq(int i) const;
    void set_sigsq(double sigsq, int i);
    const Vector &predictor_variance() const;

    Ptr<UnivParams> Sigsq_prm(int i);
    const Ptr<UnivParams> Sigsq_prm(int i) const;

    void add_forecast_data(const Matrix &predictors);

    void increment_expected_gradient(
        VectorView gradient,
        int t,
        const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

   private:
    void check_size(int n) const;

    uint xdim_;
    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;
    std::vector<string> xnames_;

    // Each model is the prior for the differences in regression
    // coefficients.
    std::vector<Ptr<ZeroMeanGaussianModel> > coefficient_transition_model_;

    // Each column of X is a time point, so it is the transpose of the
    // constructor argument.  X should not contain an intercept.
    std::vector<SparseVector> X_;
    Vector predictor_variance_;

    Ptr<IdentityMatrix> transition_matrix_;
    Ptr<UpperLeftDiagonalMatrix> transition_variance_matrix_;
  };
}
#endif //  BOOM_STATE_SPACE_DYNAMIC_REGRESSION_STATE_MODEL_HPP_
