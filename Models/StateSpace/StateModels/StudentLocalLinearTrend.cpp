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

#include "Models/StateSpace/StateModels/StudentLocalLinearTrend.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef StudentLocalLinearTrendStateModel SLLTSM;
  }  // namespace

  SLLTSM::StudentLocalLinearTrendStateModel(double sigma_level, double nu_level,
                                            double sigma_slope, double nu_slope)
      : ParamPolicy(new UnivParams(sigma_level), new UnivParams(nu_level),
                    new UnivParams(sigma_slope), new UnivParams(nu_slope)),
        observation_matrix_(2),
        state_transition_matrix_(new LocalLinearTrendMatrix),
        state_variance_matrix_(new DiagonalMatrixBlock(2)),
        state_error_expander_(new IdentityMatrix(2)),
        initial_state_mean_(2, 0.0),
        initial_state_variance_(2),
        behavior_(MIXTURE) {
    observation_matrix_[0] = 1.0;
    // The latent_slope_scale_factors_ and latent_level_scale_factors_
    // are initialized as zero length vectors, and then resized when
    // observe_time_dimension is called.
  }

  SLLTSM::StudentLocalLinearTrendStateModel(
      const StudentLocalLinearTrendStateModel &rhs)
      : Model(rhs),
        StateModel(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        observation_matrix_(rhs.observation_matrix_),
        state_transition_matrix_(new LocalLinearTrendMatrix),
        state_variance_matrix_(rhs.state_variance_matrix_->clone()),
        state_error_expander_(rhs.state_error_expander_->clone()),
        initial_state_mean_(rhs.initial_state_mean_),
        initial_state_variance_(rhs.initial_state_variance_),
        latent_level_scale_factors_(rhs.latent_level_scale_factors_),
        latent_slope_scale_factors_(rhs.latent_slope_scale_factors_),
        level_complete_data_sufficient_statistics_(
            rhs.level_complete_data_sufficient_statistics_),
        slope_complete_data_sufficient_statistics_(
            rhs.slope_complete_data_sufficient_statistics_),
        level_weight_sufficient_statistics_(
            rhs.level_weight_sufficient_statistics_),
        slope_weight_sufficient_statistics_(
            rhs.slope_weight_sufficient_statistics_),
        behavior_(rhs.behavior_) {}

  StudentLocalLinearTrendStateModel *SLLTSM::clone() const {
    return new StudentLocalLinearTrendStateModel(*this);
  }

  void SLLTSM::observe_time_dimension(int max_time) {
    if (latent_level_scale_factors_.size() < max_time) {
      int old_size = latent_level_scale_factors_.size();
      latent_level_scale_factors_.resize(max_time);
      latent_slope_scale_factors_.resize(max_time);
      for (int i = old_size; i < max_time; ++i) {
        latent_slope_scale_factors_[i] = 1.0;
        latent_level_scale_factors_[i] = 1.0;
      }
    }
  }

  void SLLTSM::observe_state(const ConstVectorView &then,
                             const ConstVectorView &now, int time_now) {
    double level_now = now[0];
    double slope_now = now[1];
    double level_then = then[0];
    double slope_then = then[1];

    double level_residual = level_now - (level_then + slope_then);
    // The 'weight' we are about to draw has to do with the variance
    // of the residual in the then -> now transition, so it applies to
    // 'then'.  That means we have to first update the complete data
    // sufficient statistics with the old weight before recording the
    // new weight.
    level_residuals_.push_back(level_residual);
    level_complete_data_sufficient_statistics_.update_raw(
        level_residual, latent_level_scale_factors_[time_now - 1]);
    double level_alpha = .5 * (1 + nu_level());
    double level_beta =
        .5 * (nu_level() + level_residual * level_residual / sigsq_level());
    latent_level_scale_factors_[time_now - 1] = rgamma(level_alpha, level_beta);
    level_weight_sufficient_statistics_.update_raw(
        latent_level_scale_factors_[time_now - 1]);

    double slope_residual = slope_now - slope_then;
    slope_residuals_.push_back(slope_residual);
    slope_complete_data_sufficient_statistics_.update_raw(
        slope_residual, latent_slope_scale_factors_[time_now - 1]);
    double slope_alpha = .5 * (1 + nu_slope());
    double slope_beta =
        .5 * (nu_slope() + slope_residual * slope_residual / sigsq_slope());
    latent_slope_scale_factors_[time_now - 1] = rgamma(slope_alpha, slope_beta);
    slope_weight_sufficient_statistics_.update_raw(
        latent_slope_scale_factors_[time_now - 1]);
  }

  void SLLTSM::update_complete_data_sufficient_statistics(
      int t, const ConstVectorView &, const ConstSubMatrix &) {
    report_error(
        "The StudentLocalLinearTrendStateModel cannot be part "
        "of the EM algorithm.");
  }

  void SLLTSM::simulate_state_error(RNG &rng, VectorView eta, int t) const {
    switch (behavior_) {
      case MIXTURE:
        simulate_conditional_state_error(rng, eta, t);
        break;
      case MARGINAL:
        simulate_marginal_state_error(rng, eta, t);
        break;
      default:
        ostringstream err;
        err << "Cannot handle unknown enumerator: " << behavior_
            << " in SLLTSM::simulate_state_error." << endl;
        report_error(err.str());
    }
  }

  void SLLTSM::simulate_marginal_state_error(RNG &rng, VectorView eta,
                                             int t) const {
    eta[0] = rt_mt(rng, nu_level()) * sigma_level();
    eta[1] = rt_mt(rng, nu_slope()) * sigma_slope();
  }

  void SLLTSM::simulate_conditional_state_error(RNG &rng, VectorView eta,
                                                int t) const {
    double level_weight = latent_level_scale_factors_[t];
    double slope_weight = latent_slope_scale_factors_[t];
    eta[0] = rnorm_mt(rng, 0, sigma_level() / sqrt(level_weight));
    eta[1] = rnorm_mt(rng, 0, sigma_slope() / sqrt(slope_weight));
  }

  Ptr<SparseMatrixBlock> SLLTSM::state_transition_matrix(int t) const {
    return state_transition_matrix_;
  }

  Ptr<SparseMatrixBlock> SLLTSM::state_variance_matrix(int t) const {
    switch (behavior_) {
      case MIXTURE:
        return conditional_state_variance_matrix(t);
      case MARGINAL:
        return marginal_state_variance_matrix(t);
      default:
        ostringstream err;
        err << "Cannot handle unknown enumerator: " << behavior_
            << " in SLLTSM::state_variance_matrix." << endl;
        report_error(err.str());
        return Ptr<SparseMatrixBlock>(NULL);
    }
  }

  Ptr<SparseMatrixBlock> SLLTSM::conditional_state_variance_matrix(
      int t) const {
    state_variance_matrix_->mutable_element(0) =
        sigsq_level() / latent_level_scale_factors_[t];
    state_variance_matrix_->mutable_element(1) =
        sigsq_slope() / latent_slope_scale_factors_[t];
    return state_variance_matrix_;
  }

  Ptr<SparseMatrixBlock> SLLTSM::marginal_state_variance_matrix(int t) const {
    state_variance_matrix_->mutable_element(0) = sigsq_level();
    state_variance_matrix_->mutable_element(1) = sigsq_slope();
    return state_variance_matrix_;
  }

  Ptr<SparseMatrixBlock> SLLTSM::state_error_expander(int t) const {
    return state_error_expander_;
  }

  Ptr<SparseMatrixBlock> SLLTSM::state_error_variance(int t) const {
    return state_variance_matrix(t);
  }

  SparseVector SLLTSM::observation_matrix(int t) const {
    return observation_matrix_;
  }

  Vector StudentLocalLinearTrendStateModel::initial_state_mean() const {
    return initial_state_mean_;
  }

  void StudentLocalLinearTrendStateModel::set_initial_state_mean(
      const Vector &v) {
    initial_state_mean_ = v;
  }

  SpdMatrix StudentLocalLinearTrendStateModel::initial_state_variance() const {
    return initial_state_variance_;
  }

  void SLLTSM::set_initial_state_variance(const SpdMatrix &V) {
    initial_state_variance_ = V;
  }

  //----------------------------------------------------------------------
  // Accessors for parameter storage
  Ptr<UnivParams> SLLTSM::SigsqLevel_prm() { return ParamPolicy::prm1(); }
  const Ptr<UnivParams> SLLTSM::SigsqLevel_prm() const {
    return ParamPolicy::prm1();
  }
  Ptr<UnivParams> SLLTSM::NuLevel_prm() { return ParamPolicy::prm2(); }
  const Ptr<UnivParams> SLLTSM::NuLevel_prm() const {
    return ParamPolicy::prm2();
  }
  Ptr<UnivParams> SLLTSM::SigsqSlope_prm() { return ParamPolicy::prm3(); }
  const Ptr<UnivParams> SLLTSM::SigsqSlope_prm() const {
    return ParamPolicy::prm3();
  }
  Ptr<UnivParams> SLLTSM::NuSlope_prm() { return ParamPolicy::prm4(); }
  const Ptr<UnivParams> SLLTSM::NuSlope_prm() const {
    return ParamPolicy::prm4();
  }

  //----------------------------------------------------------------------
  // Accessors for parameter values
  double SLLTSM::sigma_level() const { return sqrt(sigsq_level()); }
  double SLLTSM::sigsq_level() const { return ParamPolicy::prm1_ref().value(); }
  double SLLTSM::nu_level() const { return ParamPolicy::prm2_ref().value(); }

  double SLLTSM::sigma_slope() const { return sqrt(sigsq_slope()); }
  double SLLTSM::sigsq_slope() const { return ParamPolicy::prm3_ref().value(); }
  double SLLTSM::nu_slope() const { return ParamPolicy::prm4_ref().value(); }

  //----------------------------------------------------------------------
  // Setters for parameter values.
  void SLLTSM::set_sigma_level(double sigma) { set_sigsq_level(sigma * sigma); }
  void SLLTSM::set_sigsq_level(double sigsq) { prm1_ref().set(sigsq); }
  void SLLTSM::set_nu_level(double nu) { prm2_ref().set(nu); }

  void SLLTSM::set_sigma_slope(double sigma) { set_sigsq_slope(sigma * sigma); }
  void SLLTSM::set_sigsq_slope(double sigsq) { prm3_ref().set(sigsq); }
  void SLLTSM::set_nu_slope(double nu) { prm4_ref().set(nu); }

  void SLLTSM::clear_data() {
    DataPolicy::clear_data();
    level_complete_data_sufficient_statistics_.clear();
    level_weight_sufficient_statistics_.clear();
    slope_complete_data_sufficient_statistics_.clear();
    slope_weight_sufficient_statistics_.clear();
    level_residuals_.clear();
    slope_residuals_.clear();
  }

  const WeightedGaussianSuf &SLLTSM::sigma_level_complete_data_suf() const {
    return level_complete_data_sufficient_statistics_;
  }

  const WeightedGaussianSuf &SLLTSM::sigma_slope_complete_data_suf() const {
    return slope_complete_data_sufficient_statistics_;
  }

  const GammaSuf &SLLTSM::nu_level_complete_data_suf() const {
    return level_weight_sufficient_statistics_;
  }

  const GammaSuf &SLLTSM::nu_slope_complete_data_suf() const {
    return slope_weight_sufficient_statistics_;
  }

  const Vector &SLLTSM::latent_level_weights() const {
    return latent_level_scale_factors_;
  }

  const Vector &SLLTSM::latent_slope_weights() const {
    return latent_slope_scale_factors_;
  }

  void SLLTSM::set_behavior(StateModel::Behavior behavior) {
    behavior_ = behavior;
  }

}  // namespace BOOM
