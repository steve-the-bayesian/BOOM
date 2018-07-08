// Copyright 2018 Google LLC. All Rights Reserved.
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

#ifndef BOOM_STUDENT_LOCAL_LINEAR_TREND_STATE_MODEL_HPP_
#define BOOM_STUDENT_LOCAL_LINEAR_TREND_STATE_MODEL_HPP_

#include "Models/GammaModel.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_4.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/WeightedGaussianSuf.hpp"

namespace BOOM {
  // This is a 'robust' version of the local linear trend model with T
  // errors in place of the usual Gaussian errors.
  //
  //     mu[t+1] = mu[t] + delta[t] + u[t]   u[t] ~ T(sigma0, nu0)
  //  delta[t+1] = delta[t] + v[t]           v[t] ~ T(sigma1, nu1)
  //
  // MCMC for this model uses a latent variable representation of the
  // T distribution: T(nu) = N(0, 1) / sqrt(Gamma(nu/2, nu/2)).  This
  // says that if t ~ T(nu) then it is possible to write t = z/sqrt(w)
  // where z ~ N(0,1) and w ~ Gamma(nu/2, nu/2).  Think of 'w' as a
  // 'weight' in a weighted regression.
  //
  // This class maintains separate vectors of w's for the level and
  // trend components of state, so that conditional on the w's, the
  // state is a standard local linear trend with non-constant
  // variance.  The value of w is updated when "observe_state" is
  // called.
  class StudentLocalLinearTrendStateModel
      : public ParamPolicy_4<UnivParams,   // level variance
                             UnivParams,   // level tail thickness
                             UnivParams,   // slope variance
                             UnivParams>,  // slope tail thickness
        public IID_DataPolicy<DoubleData>,
        public PriorPolicy,
        virtual public StateModel {
   public:
    explicit StudentLocalLinearTrendStateModel(double sigma_level = 1.0,
                                               double nu_level = 1000,
                                               double sigma_slope = 1.0,
                                               double nu_slope = 1000);
    StudentLocalLinearTrendStateModel(
        const StudentLocalLinearTrendStateModel &rhs);
    StudentLocalLinearTrendStateModel *clone() const override;

    void observe_time_dimension(int max_time) override;

    void observe_state(const ConstVectorView &then, const ConstVectorView &now,
                       int time_now) override;

    uint state_dimension() const override { return 2; }
    uint state_error_dimension() const override { return state_dimension(); }

    // This implementation throws, because this model cannot be part
    // of an EM algorithm.
    void update_complete_data_sufficient_statistics(
        int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    // The state error simulation is conditional on the value of the
    // latent variance weights.  It needs to be that way so that
    // latent data imputation can work properly.
    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;
    void simulate_marginal_state_error(RNG &rng, VectorView eta, int t) const;
    void simulate_conditional_state_error(RNG &rng, VectorView eta,
                                          int t) const;

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override;
    Ptr<SparseMatrixBlock> conditional_state_variance_matrix(int t) const;
    Ptr<SparseMatrixBlock> marginal_state_variance_matrix(int t) const;

    Ptr<SparseMatrixBlock> state_error_expander(int t) const override;
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override;

    SparseVector observation_matrix(int t) const override;

    Vector initial_state_mean() const override;
    void set_initial_state_mean(const Vector &v);
    SpdMatrix initial_state_variance() const override;
    void set_initial_state_variance(const SpdMatrix &V);

    // With "marginal" behavior set the model acts like a T
    // distribution when simulating state and exposing parameters.
    // With "mixture" behavior set it acts like a conditionally
    // normal model with unequal variances determined by latent
    // chi-square variables.
    void set_behavior(StateModel::Behavior behavior) override;

    Ptr<UnivParams> SigsqLevel_prm();
    Ptr<UnivParams> NuLevel_prm();
    Ptr<UnivParams> SigsqSlope_prm();
    Ptr<UnivParams> NuSlope_prm();
    const Ptr<UnivParams> SigsqLevel_prm() const;
    const Ptr<UnivParams> NuLevel_prm() const;
    const Ptr<UnivParams> SigsqSlope_prm() const;
    const Ptr<UnivParams> NuSlope_prm() const;

    double sigma_level() const;
    double sigsq_level() const;
    double nu_level() const;
    double sigma_slope() const;
    double sigsq_slope() const;
    double nu_slope() const;

    void set_sigma_level(double sigma);
    void set_sigsq_level(double sigsq);
    void set_nu_level(double nu);
    void set_sigma_slope(double sigma);
    void set_sigsq_slope(double sigsq);
    void set_nu_slope(double nu);

    void clear_data() override;
    const WeightedGaussianSuf &sigma_level_complete_data_suf() const;
    const WeightedGaussianSuf &sigma_slope_complete_data_suf() const;
    const GammaSuf &nu_level_complete_data_suf() const;
    const GammaSuf &nu_slope_complete_data_suf() const;
    const Vector &level_residuals() const {return level_residuals_;}
    const Vector &slope_residuals() const {return slope_residuals_;}
    
    // Posterior draws for the weights in the normal mixture
    // representation of the T distribution.  For Gaussian models the
    // weights will be around 1.  A large outlier has a small weight.
    const Vector &latent_level_weights() const;
    const Vector &latent_slope_weights() const;

   private:
    void check_dim(const ConstVectorView &) const;

    SparseVector observation_matrix_;
    Ptr<LocalLinearTrendMatrix> state_transition_matrix_;
    mutable Ptr<DiagonalMatrixBlock> state_variance_matrix_;
    Ptr<IdentityMatrix> state_error_expander_;

    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;

    Vector latent_level_scale_factors_;
    Vector latent_slope_scale_factors_;

    WeightedGaussianSuf level_complete_data_sufficient_statistics_;
    WeightedGaussianSuf slope_complete_data_sufficient_statistics_;

    GammaSuf level_weight_sufficient_statistics_;
    GammaSuf slope_weight_sufficient_statistics_;

    Vector level_residuals_;
    Vector slope_residuals_;
    
    StateModel::Behavior behavior_;
  };

}  // namespace BOOM
#endif  // BOOM_LOCAL_LINEAR_TREND_STATE_MODEL_HPP_
